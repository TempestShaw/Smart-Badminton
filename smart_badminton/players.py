"""Player detection and identity tracking for singles and doubles.

Each half of the court holds at most ``players_per_side`` people (1 in singles, 2 in doubles) and nobody crosses the
net, so identities are bounded slots (``near-1``, ``near-2``, ``far-1``, ``far-2``) rather than an open-ended set of
track IDs. A slot follows its player with a constant-velocity prediction in court meters (or normalized image units
without corner calibration), an optimal per-side assignment and a torso colour histogram. When a player is lost, the
slot keeps its appearance so the same person is re-identified instead of receiving a new ID.

Kinematics (wrist speed, swing, lunge, foot speed) are computed per slot, so two different people are never
differenced against each other. :func:`side_features` folds the slots back into the singles feature schema the rally
model was trained on.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import CourtGeometry

SIDES = ("near", "far")
MATCH_FORMATS = {"singles": 1, "doubles": 2}
COURT_LENGTH_METERS = 13.4
COURT_WIDTH_METERS = 6.1
NET_Y_METERS = COURT_LENGTH_METERS / 2
LOST_MATCH_APPEARANCE = 0.5  # Bhattacharyya distance above which a lost slot will not take a new person...
LOST_MATCH_PATIENCE = 4.0  # ...for this many seconds; after that a mismatch only weighs in (capped below)
LOST_MATCH_CAP = 0.6  # so a slot first taken by the wrong person (an umpire, a coach) can still pass to the player
MAXIMUM_PLAYER_SPEED = 6.0  # m/s (or tracking units per second without corner calibration)
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
KEYPOINT_CONFIDENCE = 0.2

PLAYER_TRACK_FIELDS = [
    "time_seconds",
    "frame_index",
    "player_id",
    "side",
    "status",
    "confidence",
    "box_x1_normalized",
    "box_y1_normalized",
    "box_x2_normalized",
    "box_y2_normalized",
    "foot_x_normalized",
    "foot_y_normalized",
    "court_x_meters",
    "court_y_meters",
    "speed_meters_per_second",
    "foot_speed_normalized",
    "swing_score",
    "lunge_score",
    "wrist_x_normalized",
    "wrist_y_normalized",
    "wrist_speed_normalized",
]


def players_per_side(match_format: str) -> int:
    if match_format not in MATCH_FORMATS:
        raise ValueError(f"match format must be one of {', '.join(MATCH_FORMATS)}")
    return MATCH_FORMATS[match_format]


def match_format_from_config(config: Path | dict[str, Any] | None) -> str:
    """The ``analysis.match_format`` of a court config; an absent or unreadable config means singles."""
    if isinstance(config, Path):
        try:
            config = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = None
    data = config if isinstance(config, dict) else {}
    analysis = data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
    value = str(analysis.get("match_format", "singles"))
    return value if value in MATCH_FORMATS else "singles"


@dataclass(eq=False)  # identity semantics: detections hold arrays
class PoseDetection:
    side: str
    box: np.ndarray
    score: float
    keypoints: np.ndarray
    confidences: np.ndarray | None
    foot: tuple[float, float]
    court: tuple[float, float] | None = None
    appearance: np.ndarray | None = None
    spawnable: bool = True
    kinematics: dict[str, Any] = field(default_factory=dict)

    def keypoint_visible(self, index: int) -> bool:
        return self.confidences is None or float(self.confidences[index]) >= KEYPOINT_CONFIDENCE


@dataclass(slots=True)
class PlayerObservation:
    """One player at one sample, detected or briefly predicted through an occlusion."""

    player_id: str
    side: str
    status: str
    time_seconds: float
    confidence: float
    box: np.ndarray
    keypoints: np.ndarray | None
    confidences: np.ndarray | None
    foot: tuple[float, float]
    court: tuple[float, float] | None
    speed_meters_per_second: float | None = None
    foot_speed_normalized: float = 0.0
    stance_width_normalized: float = 0.0
    ankles_visible: bool = False
    swing_score: float = 0.0
    lunge_score: float = 0.0
    wrist: tuple[float, float] | None = None
    wrist_speed_normalized: float = 0.0
    appearance: np.ndarray | None = None

    @property
    def detected(self) -> bool:
        return self.status == "detected"


def torso_histogram(frame: np.ndarray, box: np.ndarray, keypoints: np.ndarray, confidences: np.ndarray | None) -> np.ndarray | None:
    """A 12x4 hue/saturation histogram of the shirt, robust to arm and racket motion."""
    height, width = frame.shape[:2]
    torso = [5, 6, 11, 12]
    if confidences is None or all(float(confidences[index]) >= KEYPOINT_CONFIDENCE for index in torso):
        points = keypoints[torso]
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
    else:
        bx1, by1, bx2, by2 = box
        box_width, box_height = bx2 - bx1, by2 - by1
        x1, x2 = bx1 + box_width * 0.25, bx2 - box_width * 0.25
        y1, y2 = by1 + box_height * 0.18, by1 + box_height * 0.55
    x1, x2 = int(max(0, min(width - 1, x1))), int(max(0, min(width, x2)))
    y1, y2 = int(max(0, min(height - 1, y1))), int(max(0, min(height, y2)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
    total = float(histogram.sum())
    return (histogram / total).astype(np.float32) if total > 0 else None


def appearance_distance(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    coefficient = float(np.sum(np.sqrt(np.maximum(first, 0.0) * np.maximum(second, 0.0))))
    return math.sqrt(max(0.0, 1.0 - min(1.0, coefficient)))  # Bhattacharyya distance in [0, 1]


class CourtFrame:
    """Maps image foot points into the tracking space: court meters, or scaled normalized coordinates."""

    def __init__(self, geometry: CourtGeometry, width: int, height: int):
        self.geometry = geometry
        self.width = width
        self.height = height
        self.matrix = geometry.homography_to_court(width, height)
        self._upright: bool | None = None

    @property
    def metric(self) -> bool:
        return self.matrix is not None

    @property
    def upright_view(self) -> bool:
        """True for cameras behind a baseline, where players stand tall and the far court shrinks in the image.

        From overhead (a drone) or square-on, people's boxes are about as wide as they are tall, so box-shape
        heuristics must not apply. Without corner calibration, an upright view is assumed.
        """
        if self.matrix is None:
            return True
        if self._upright is not None:
            return self._upright
        near = [self.image_point(point) for point in ((0.46, 0.0), (5.64, 0.0))]
        far = [self.image_point(point) for point in ((0.46, 13.4), (5.64, 13.4))]
        self._upright = math.dist(*near) > math.dist(*far) * 1.25
        return self._upright

    def image_point(self, court: tuple[float, float]) -> tuple[float, float]:
        inverse = np.linalg.inv(self.matrix)
        mapped = cv2.perspectiveTransform(np.asarray([[court]], dtype=np.float32), inverse)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def court_point(self, foot: tuple[float, float]) -> tuple[float, float] | None:
        if self.matrix is None:
            return None
        source = np.asarray([[[foot[0] * self.width, foot[1] * self.height]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, self.matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def space(self, foot: tuple[float, float], court: tuple[float, float] | None) -> np.ndarray:
        if court is not None:
            return np.asarray(court, dtype=float)
        # Without corner calibration, ~15 normalized-image units span the frame, close enough to meters for gating.
        return np.asarray(foot, dtype=float) * 15.0

    def side_for(self, foot: tuple[float, float], court: tuple[float, float] | None) -> str | None:
        x, y = foot
        for side in SIDES:
            if self.geometry.contains_normalized(f"{side}_player_zone", x, y):
                return side
        if court is None:
            return None
        # Doubles players regularly stand outside zones drawn for singles; trust calibrated court bounds instead.
        court_x, court_y = court
        if not (-1.2 <= court_x <= COURT_WIDTH_METERS + 1.2 and -2.5 <= court_y <= COURT_LENGTH_METERS + 2.5):
            return None
        if abs(court_y - NET_Y_METERS) < 0.25:
            return None
        return "near" if court_y < NET_Y_METERS else "far"


ANKLE_ABOVE_SOLE = 0.035  # ankle keypoints sit ~7 cm above the floor: ~3.5% of a standing person's box height


def foot_point(box: np.ndarray, keypoints: np.ndarray, confidences: np.ndarray | None) -> np.ndarray:
    """Where the player stands, in pixels: the ankles moved down to the sole, else the bottom of the box.

    Near the far baseline one pixel can be ~0.1 m of depth, so the ankle-to-sole gap alone would place far players
    up to half a metre too deep.
    """
    ankles = [
        keypoints[index]
        for index in (LEFT_ANKLE, RIGHT_ANKLE)
        if confidences is None or float(confidences[index]) >= KEYPOINT_CONFIDENCE
    ]
    if not ankles:
        return np.asarray([(box[0] + box[2]) / 2.0, box[3]])
    point = np.mean(ankles, axis=0).astype(np.float64)
    point[1] = min(float(box[3]), point[1] + ANKLE_ABOVE_SOLE * float(box[3] - box[1]))
    return point


def court_detections(
    frame: np.ndarray,
    people: Iterable[tuple[np.ndarray, float, np.ndarray, np.ndarray | None]],
    court: CourtFrame,
    expected_side: str | None = None,
    doubles: bool = False,
) -> list[PoseDetection]:
    """Keep people standing on this court's halves; ``expected_side`` limits a per-side crop to its own half.

    Singles keeps the original rule (foot inside the active court and the side's player zone). Doubles partners
    cover the full width and often stand behind the baseline, so with calibrated corners they are admitted by court
    position too.
    """
    height, width = frame.shape[:2]
    geometry = court.geometry
    detections = []
    for box, score, keypoints, confidences in people:
        foot = foot_point(box, keypoints, confidences)
        normalized = (float(foot[0] / width), float(foot[1] / height))
        court_point = court.court_point(normalized)
        inside = geometry.contains_active_court(normalized[0], normalized[1], margin=0.015)
        if doubles:
            side = court.side_for(normalized, court_point)
            if side is None or not (inside or court_point is not None):
                continue
        else:
            zones = [side for side in SIDES if geometry.contains_normalized(f"{side}_player_zone", *normalized)]
            side = zones[0] if inside and zones else None
        if side is None or (expected_side is not None and side != expected_side):
            continue
        aspect = float((box[3] - box[1]) / max(1.0, box[2] - box[0]))
        detections.append(
            PoseDetection(
                side=side,
                box=np.asarray(box, dtype=float),
                score=float(score),
                keypoints=np.asarray(keypoints, dtype=float),
                confidences=None if confidences is None else np.asarray(confidences, dtype=float),
                foot=normalized,
                court=court_point,
                appearance=torso_histogram(frame, box, keypoints, confidences),
                # Chairs, bags and seated officials are wider than a standing far player; they may not claim a slot.
                spawnable=not (court.upright_view and side == "far" and aspect < 1.1),
            )
        )
    return detections


@dataclass
class _Track:
    player_id: str
    side: str
    position: np.ndarray
    velocity: np.ndarray
    last_time: float
    appearance: np.ndarray | None
    hits: int = 1
    misses: int = 0
    last_detection: PoseDetection | None = None
    previous_wrists: dict[int, tuple[float, float]] = field(default_factory=dict)
    previous_foot: tuple[float, float] | None = None
    previous_court: tuple[float, float] | None = None
    previous_span: float | None = None
    previous_kinematics_time: float | None = None
    speed_meters_per_second: float = 0.0

    def predicted(self, time_seconds: float) -> np.ndarray:
        # Extrapolate at most 0.3 s: badminton footwork reverses direction faster than that.
        dt = max(0.0, min(0.3, time_seconds - self.last_time))
        return self.position + self.velocity * dt

    def absorb_appearance(self, appearance: np.ndarray | None) -> None:
        if appearance is None:
            return
        self.appearance = appearance if self.appearance is None else (self.appearance * 0.85 + appearance * 0.15)


class PlayerTracker:
    """Online slot tracker. Deterministic, pure NumPy, and cheap: a frame costs microseconds."""

    def __init__(
        self,
        court: CourtFrame,
        players: int = 1,
        coast_seconds: float = 1.2,
        appearance_weight: float = 0.55,
    ):
        if players not in (1, 2):
            raise ValueError("players per side must be 1 or 2")
        self.court = court
        self.players = players
        self.coast_seconds = coast_seconds
        self.appearance_weight = appearance_weight
        self.tracks: dict[str, list[_Track]] = {side: [] for side in SIDES}
        self.diagonal = math.hypot(court.width, court.height)

    def _gate(self, track: _Track, time_seconds: float) -> float:
        dt = max(0.05, time_seconds - track.last_time)
        return min(6.0, 1.4 + 5.0 * dt)  # players move at most ~6 m/s

    def _cost(self, track: _Track, detection: PoseDetection, time_seconds: float) -> float:
        position = self.court.space(detection.foot, detection.court)
        distance = float(np.linalg.norm(track.predicted(time_seconds) - position))
        gate = self._gate(track, time_seconds)
        lost = time_seconds - track.last_time > self.coast_seconds
        if distance > gate and not lost:
            return math.inf
        motion = min(1.0, distance / gate)
        appearance = appearance_distance(track.appearance, detection.appearance)
        if lost:
            # A lost slot is re-identified by appearance; position only breaks ties. A clearly different shirt is
            # someone else (often the partner, whose own slot is momentarily unmatched).
            # After a long absence lighting may have changed the colours, so the slot is reopened to anyone.
            absent = time_seconds - track.last_time
            if appearance is None:
                return 0.5 + motion * 0.15
            if absent < LOST_MATCH_PATIENCE:
                return math.inf if appearance > LOST_MATCH_APPEARANCE else appearance + motion * 0.15
            return min(appearance, LOST_MATCH_CAP) + motion * 0.15
        if appearance is None:
            return motion
        return motion * (1.0 - self.appearance_weight * 0.5) + appearance * self.appearance_weight

    def _assign(self, tracks: list[_Track], detections: list[PoseDetection], time_seconds: float) -> dict[int, int]:
        """Exact minimum-cost assignment; with ≤2 tracks and a handful of candidates brute force is the fastest."""
        if not tracks or not detections:
            return {}
        costs = [[self._cost(track, detection, time_seconds) for detection in detections] for track in tracks]
        unmatched = 0.95
        best: tuple[float, dict[int, int]] = (unmatched * len(tracks), {})
        choices = list(range(len(detections))) + [None]
        for combination in itertools.product(choices, repeat=len(tracks)):
            used = [choice for choice in combination if choice is not None]
            if len(used) != len(set(used)):
                continue
            total = 0.0
            for track_index, choice in enumerate(combination):
                total += unmatched if choice is None else costs[track_index][choice]
            if total < best[0] - 1e-9:
                best = (total, {index: choice for index, choice in enumerate(combination) if choice is not None})
        return best[1]

    def update(self, detections: list[PoseDetection], time_seconds: float) -> list[PlayerObservation]:
        observations: list[PlayerObservation] = []
        for side in SIDES:
            candidates = sorted(
                (detection for detection in detections if detection.side == side),
                key=lambda detection: detection.score * _area(detection.box),
                reverse=True,
            )[:6]
            tracks = self.tracks[side]
            assignment = self._assign(tracks, candidates, time_seconds)
            matched = set(assignment.values())
            for track_index, detection_index in assignment.items():
                self._observe(tracks[track_index], candidates[detection_index], time_seconds)
            for detection_index, detection in enumerate(candidates):
                if detection_index in matched or len(tracks) >= self.players or not detection.spawnable:
                    continue
                if any(_overlap(detection.box, other.box) > 0.55 for other in (candidates[i] for i in matched)):
                    continue  # a duplicate box of a player who is already matched
                tracks.append(self._spawn(side, len(tracks) + 1, detection, time_seconds))
                matched.add(detection_index)
            for track in tracks:
                observation = self._observation(track, time_seconds)
                if observation is not None:
                    observations.append(observation)
        return observations

    def _spawn(self, side: str, slot: int, detection: PoseDetection, time_seconds: float) -> _Track:
        track = _Track(
            player_id=f"{side}-{slot}",
            side=side,
            position=self.court.space(detection.foot, detection.court),
            velocity=np.zeros(2),
            last_time=time_seconds,
            appearance=detection.appearance,
            last_detection=detection,
        )
        self._kinematics(track, detection, time_seconds)
        track.misses = 0
        return track

    def _observe(self, track: _Track, detection: PoseDetection, time_seconds: float) -> None:
        position = self.court.space(detection.foot, detection.court)
        dt = time_seconds - track.last_time
        if 0 < dt <= 0.6:
            measured = (position - track.position) / dt
            speed = float(np.linalg.norm(measured))
            if speed > MAXIMUM_PLAYER_SPEED:
                measured *= MAXIMUM_PLAYER_SPEED / speed
            # A foot taken from the box bottom (ankles hidden) is noisy; let it steer the velocity less.
            weight = 0.6 if _ankles_visible(detection) else 0.25
            track.velocity = track.velocity * (1.0 - weight) + measured * weight
        else:
            track.velocity = np.zeros(2)
        if dt > 0.35:
            # A slot recovered after a gap restarts its kinematics instead of differencing across the gap.
            track.previous_wrists.clear()
            track.previous_foot = None
            track.previous_court = None
            track.previous_span = None
        track.position = position
        track.last_time = time_seconds
        track.hits += 1
        track.misses = 0
        track.last_detection = detection
        track.absorb_appearance(detection.appearance)
        self._kinematics(track, detection, time_seconds)

    def _kinematics(self, track: _Track, detection: PoseDetection, time_seconds: float) -> None:
        previous_time = track.previous_kinematics_time
        delta = time_seconds - previous_time if previous_time is not None else 0.0
        track.previous_kinematics_time = time_seconds
        width, height = self.court.width, self.court.height
        wrists = []
        for index in (LEFT_WRIST, RIGHT_WRIST):
            if not detection.keypoint_visible(index):
                track.previous_wrists.pop(index, None)
                continue
            point = (float(detection.keypoints[index][0]), float(detection.keypoints[index][1]))
            previous = track.previous_wrists.get(index)
            speed = 0.0
            if previous is not None and delta > 0:
                speed = math.hypot(point[0] - previous[0], point[1] - previous[1]) / self.diagonal / delta
            wrists.append((speed, point[0] / width, point[1] / height, previous is not None))
            track.previous_wrists[index] = point
        detection_kinematics: dict[str, Any] = {
            "swing_score": 0.0,
            "wrist": None,
            "wrist_speed": 0.0,
            "foot_speed": 0.0,
            "stance": 0.0,
            "lunge": 0.0,
            "ankles": False,
        }
        if wrists:
            speed, x, y, _measured = max(wrists, key=lambda item: item[0])
            detection_kinematics.update(
                swing_score=float(np.clip(speed / 1.2, 0.0, 1.0)),
                wrist=(x, y),
                wrist_speed=float(np.clip(speed, 0.0, 3.0)),
            )
        ankles = [detection.keypoints[index] for index in (LEFT_ANKLE, RIGHT_ANKLE) if detection.keypoint_visible(index)]
        if len(ankles) == 2:
            span = abs(float(ankles[0][0] - ankles[1][0])) / width
            foot = np.mean(ankles, axis=0)
            normalized = (float(foot[0] / width), float(foot[1] / height))
            if track.previous_foot is not None and delta > 0:
                detection_kinematics["foot_speed"] = float(
                    np.clip(math.dist(normalized, track.previous_foot) / delta, 0.0, 3.0)
                )
            previous_span = track.previous_span if track.previous_span is not None else span
            detection_kinematics["stance"] = span
            detection_kinematics["ankles"] = True
            detection_kinematics["lunge"] = float(np.clip((span + max(0.0, span - previous_span) * 2.0) / 0.18, 0.0, 1.0))
            track.previous_span = span
            track.previous_foot = normalized
        if detection.court is not None:
            if track.previous_court is not None and delta > 0:
                measured = math.dist(detection.court, track.previous_court) / delta
                track.speed_meters_per_second = track.speed_meters_per_second * 0.5 + min(9.0, measured) * 0.5
            track.previous_court = detection.court
        detection.kinematics = detection_kinematics

    def _observation(self, track: _Track, time_seconds: float) -> PlayerObservation | None:
        detection = track.last_detection
        if detection is None:
            return None
        if track.last_time == time_seconds:
            kinematics = detection.kinematics
            return PlayerObservation(
                player_id=track.player_id,
                side=track.side,
                status="detected",
                time_seconds=time_seconds,
                confidence=detection.score,
                box=detection.box,
                keypoints=detection.keypoints,
                confidences=detection.confidences,
                foot=detection.foot,
                court=detection.court,
                speed_meters_per_second=track.speed_meters_per_second if detection.court is not None else None,
                foot_speed_normalized=kinematics.get("foot_speed", 0.0),
                stance_width_normalized=kinematics.get("stance", 0.0),
                ankles_visible=bool(kinematics.get("ankles")),
                swing_score=kinematics.get("swing_score", 0.0),
                lunge_score=kinematics.get("lunge", 0.0),
                wrist=kinematics.get("wrist"),
                wrist_speed_normalized=kinematics.get("wrist_speed", 0.0),
                appearance=detection.appearance,
            )
        track.misses += 1
        if time_seconds - track.last_time > 0.5:
            return None  # long gaps are reported as absent rather than invented
        # Briefly occluded (usually by a partner): carry the box along the predicted path for drawing and stats.
        shift = track.predicted(time_seconds) - track.position
        foot, court_point, box = detection.foot, detection.court, detection.box
        if court_point is not None and self.court.matrix is not None:
            court_point = (court_point[0] + float(shift[0]), court_point[1] + float(shift[1]))
            inverse = np.linalg.inv(self.court.matrix)
            image = cv2.perspectiveTransform(np.asarray([[court_point]], dtype=np.float32), inverse)[0, 0]
            new_foot = (float(image[0] / self.court.width), float(image[1] / self.court.height))
        else:
            new_foot = (foot[0] + float(shift[0]) / 15.0, foot[1] + float(shift[1]) / 15.0)
        dx = (new_foot[0] - foot[0]) * self.court.width
        dy = (new_foot[1] - foot[1]) * self.court.height
        return PlayerObservation(
            player_id=track.player_id,
            side=track.side,
            status="predicted",
            time_seconds=time_seconds,
            confidence=0.0,
            box=box + np.asarray([dx, dy, dx, dy]),
            keypoints=None,
            confidences=None,
            foot=new_foot,
            court=court_point,
            speed_meters_per_second=track.speed_meters_per_second if court_point is not None else None,
        )


def _ankles_visible(detection: PoseDetection) -> bool:
    return detection.keypoint_visible(LEFT_ANKLE) and detection.keypoint_visible(RIGHT_ANKLE)


def _area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over the smaller box, which flags duplicates even when one box is partial."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min(_area(first), _area(second))
    return intersection / smaller if smaller > 0 else 0.0


EMPTY_SIDE_FEATURES = {
    "person_visible": 0.0,
    "swing_score": 0.0,
    "lunge_score": 0.0,
    "foot_x_normalized": 0.0,
    "foot_y_normalized": 0.0,
    "foot_speed_normalized": 0.0,
    "stance_width_normalized": 0.0,
    "active_wrist_x_normalized": 0.0,
    "active_wrist_y_normalized": 0.0,
    "active_wrist_speed_normalized": 0.0,
    "active_wrist_visible": 0.0,
}


def active_player(
    observations: list[PlayerObservation], shuttle: tuple[float, float] | None = None
) -> PlayerObservation | None:
    """The player of one side most likely to be hitting: wrist nearest the shuttle, else the fastest wrist."""
    detected = [observation for observation in observations if observation.detected]
    if not detected:
        return None
    with_wrist = [observation for observation in detected if observation.wrist is not None]
    if shuttle is not None and with_wrist:
        return min(with_wrist, key=lambda observation: math.dist(observation.wrist, shuttle))
    return max(detected, key=lambda observation: (observation.wrist_speed_normalized, observation.confidence))


def side_features(
    observations: list[PlayerObservation], shuttle: tuple[float, float] | None = None
) -> dict[str, float]:
    """Fold per-player observations into the per-side feature columns of the rally model."""
    output: dict[str, float] = {}
    for side in SIDES:
        players = [observation for observation in observations if observation.side == side and observation.detected]
        values = dict(EMPTY_SIDE_FEATURES)
        active = active_player(players, shuttle)
        if active is not None:
            values["person_visible"] = 1.0
            values["swing_score"] = max(player.swing_score for player in players)
            values["lunge_score"] = max(player.lunge_score for player in players)
            if active.ankles_visible:  # the feature schema reports feet only when both ankles are seen
                values["foot_x_normalized"], values["foot_y_normalized"] = active.foot
                values["foot_speed_normalized"] = active.foot_speed_normalized
                values["stance_width_normalized"] = active.stance_width_normalized
            if active.wrist is not None:
                values["active_wrist_x_normalized"], values["active_wrist_y_normalized"] = active.wrist
                values["active_wrist_speed_normalized"] = active.wrist_speed_normalized
                values["active_wrist_visible"] = 1.0
        output.update({f"{side}_{name}": value for name, value in values.items()})
    return output


class PoseEstimator:
    """YOLO pose on one crop per court half; crops keep far players large enough for reliable keypoints."""

    def __init__(
        self,
        model_path: Path,
        geometry: CourtGeometry,
        packages: Path | None = None,
        confidence: float = 0.2,
        image_size: int = 640,
    ):
        if packages is not None and str(packages) not in sys.path:
            sys.path.append(str(packages))
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.cuda = bool(torch.cuda.is_available())
        self.device: int | str = 0 if self.cuda else "cpu"
        self.geometry = geometry
        self.confidence = confidence
        self.image_size = image_size

    def people(self, frame: np.ndarray) -> list[tuple[str, list[tuple[np.ndarray, float, np.ndarray, np.ndarray | None]]]]:
        height, width = frame.shape[:2]
        crops, offsets = [], []
        for side in SIDES:
            x1, y1, x2, y2 = self.geometry.player_inference_bounds(side, width, height)
            crops.append(frame[y1:y2, x1:x2])
            offsets.append((x1, y1))
        # FP16 roughly doubles GPU throughput with no measurable keypoint loss; CPUs stay in FP32.
        precision = {"half": True} if self.cuda else {}
        results = self.model.predict(
            crops, imgsz=self.image_size, conf=self.confidence, device=self.device, verbose=False, **precision
        )
        output = []
        for side, result, (offset_x, offset_y) in zip(SIDES, results, offsets):
            people = []
            if result.boxes is not None and result.keypoints is not None and result.keypoints.xy is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                keypoints = result.keypoints.xy.detach().cpu().numpy()
                confidences = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else None
                offset = np.asarray([offset_x, offset_y], dtype=float)
                for index, box in enumerate(boxes):
                    people.append(
                        (
                            box + np.concatenate([offset, offset]),
                            float(scores[index]),
                            keypoints[index] + offset,
                            None if confidences is None else confidences[index],
                        )
                    )
            output.append((side, people))
        return output

    def detect(self, frame: np.ndarray, court: CourtFrame, doubles: bool = False) -> list[PoseDetection]:
        detections = []
        for side, people in self.people(frame):
            detections.extend(court_detections(frame, people, court, expected_side=side, doubles=doubles))
        return detections


def refine_identities(observations: list[PlayerObservation], players: int) -> list[PlayerObservation]:
    """Offline pass for doubles: undo partner swaps that happened while two players overlapped.

    The online tracker can only follow motion through an occlusion. Afterwards, each stretch between close encounters
    is compared with both partners' long-run shirt colours, and the stretch's labels are exchanged when the swapped
    reading is clearly more consistent.
    """
    if players < 2 or not observations:
        return observations
    by_side: dict[str, dict[float, dict[str, PlayerObservation]]] = {side: {} for side in SIDES}
    for observation in observations:
        by_side[observation.side].setdefault(observation.time_seconds, {})[observation.player_id] = observation
    for side, samples in by_side.items():
        first, second = f"{side}-1", f"{side}-2"
        times = sorted(samples)
        segments: list[list[float]] = [[]]
        for time_seconds in times:
            pair = samples[time_seconds]
            close = first in pair and second in pair and _feet_close(pair[first], pair[second])
            occluded = (first in pair) != (second in pair) or any(not item.detected for item in pair.values())
            if (close or occluded) and segments[-1]:
                segments.append([])
            segments[-1].append(time_seconds)
        references = {first: None, second: None}
        for segment in segments:
            means = {pid: _mean_appearance(samples[t].get(pid) for t in segment) for pid in (first, second)}
            if all(reference is not None for reference in references.values()) and all(
                value is not None for value in means.values()
            ):
                keep = appearance_distance(references[first], means[first]) + appearance_distance(
                    references[second], means[second]
                )
                swap = appearance_distance(references[first], means[second]) + appearance_distance(
                    references[second], means[first]
                )
                if swap + 0.12 < keep:
                    for time_seconds in segment:
                        pair = samples[time_seconds]
                        a, b = pair.get(first), pair.get(second)
                        if a is not None:
                            a.player_id = second
                        if b is not None:
                            b.player_id = first
                        samples[time_seconds] = {item.player_id: item for item in pair.values()}
                    means = {first: means[second], second: means[first]}
            for pid in (first, second):
                if means[pid] is None:
                    continue
                weight = min(0.5, len(segment) / 200.0)
                references[pid] = means[pid] if references[pid] is None else references[pid] * (1 - weight) + means[pid] * weight
    return observations


def _feet_close(first: PlayerObservation, second: PlayerObservation) -> bool:
    if first.court is not None and second.court is not None:
        return math.dist(first.court, second.court) < 0.9
    return math.dist(first.foot, second.foot) < 0.06


def _mean_appearance(observations: Iterable[PlayerObservation | None]) -> np.ndarray | None:
    values = [item.appearance for item in observations if item is not None and item.detected and item.appearance is not None]
    return np.mean(values, axis=0) if values else None


def observation_row(observation: PlayerObservation, frame_index: int, width: int, height: int) -> dict[str, str]:
    def number(value: float | None, digits: int = 6) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    x1, y1, x2, y2 = observation.box
    return {
        "time_seconds": f"{observation.time_seconds:.6f}",
        "frame_index": str(frame_index),
        "player_id": observation.player_id,
        "side": observation.side,
        "status": observation.status,
        "confidence": f"{observation.confidence:.3f}",
        "box_x1_normalized": number(x1 / width),
        "box_y1_normalized": number(y1 / height),
        "box_x2_normalized": number(x2 / width),
        "box_y2_normalized": number(y2 / height),
        "foot_x_normalized": number(observation.foot[0]),
        "foot_y_normalized": number(observation.foot[1]),
        "court_x_meters": number(observation.court[0] if observation.court else None, 3),
        "court_y_meters": number(observation.court[1] if observation.court else None, 3),
        "speed_meters_per_second": number(observation.speed_meters_per_second, 3),
        "foot_speed_normalized": number(observation.foot_speed_normalized),
        "swing_score": number(observation.swing_score, 4),
        "lunge_score": number(observation.lunge_score, 4),
        "wrist_x_normalized": number(observation.wrist[0] if observation.wrist else None),
        "wrist_y_normalized": number(observation.wrist[1] if observation.wrist else None),
        "wrist_speed_normalized": number(observation.wrist_speed_normalized),
    }


def write_player_tracks(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAYER_TRACK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_player_tracks(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def smooth_point(previous: tuple[float, float], current: tuple[float, float], weight: float = 0.5) -> tuple[float, float]:
    """Exponential smoothing of a court position; the foot point jitters by ~10 cm between frames."""
    return (previous[0] + (current[0] - previous[0]) * weight, previous[1] + (current[1] - previous[1]) * weight)


class SwingCounter:
    """Counts distinct swings per player: a peak above ``threshold`` with a refractory period."""

    def __init__(self, threshold: float = 0.5, refractory_seconds: float = 0.45):
        self.threshold = threshold
        self.refractory_seconds = refractory_seconds
        self.last_swing: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, player_id: str, time_seconds: float, swing_score: float) -> bool:
        if swing_score < self.threshold:
            return False
        last = self.last_swing.get(player_id)
        if last is not None and time_seconds - last < self.refractory_seconds:
            return False
        self.last_swing[player_id] = time_seconds
        self.counts[player_id] = self.counts.get(player_id, 0) + 1
        return True


def summarize_player_tracks(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Per-player movement statistics from ``player-tracks.csv``."""
    players: dict[str, dict[str, Any]] = {}
    previous: dict[str, tuple[float, float, float]] = {}
    swings = SwingCounter()
    heat_columns, heat_rows = 6, 14
    for row in rows:
        player_id = row["player_id"]
        summary = players.setdefault(
            player_id,
            {
                "player_id": player_id,
                "side": row["side"],
                "detected_samples": 0,
                "distance_meters": 0.0,
                "max_speed_meters_per_second": 0.0,
                "swings": 0,
                "heatmap": [[0] * heat_columns for _ in range(heat_rows)],
                "metric": False,
            },
        )
        if row["status"] != "detected":
            continue
        time_seconds = float(row["time_seconds"])
        summary["detected_samples"] += 1
        if swings.update(player_id, time_seconds, float(row.get("swing_score") or 0.0)):
            summary["swings"] += 1
        if not row.get("court_x_meters"):
            continue
        summary["metric"] = True
        x, y = float(row["court_x_meters"]), float(row["court_y_meters"])
        column = int(np.clip(x / COURT_WIDTH_METERS * heat_columns, 0, heat_columns - 1))
        line = int(np.clip(y / COURT_LENGTH_METERS * heat_rows, 0, heat_rows - 1))
        summary["heatmap"][line][column] += 1
        last = previous.get(player_id)
        if last is not None and 0 < time_seconds - last[0] <= 0.5:
            x, y = smooth_point(last[1:], (x, y))
            step = math.dist((x, y), last[1:])
            if step <= 9.0 * (time_seconds - last[0]):  # reject homography jumps
                summary["distance_meters"] += step
        if row.get("speed_meters_per_second"):
            summary["max_speed_meters_per_second"] = max(
                summary["max_speed_meters_per_second"], float(row["speed_meters_per_second"])
            )
        previous[player_id] = (time_seconds, x, y)
    for summary in players.values():
        summary["distance_meters"] = round(summary["distance_meters"], 1)
        summary["max_speed_meters_per_second"] = round(summary["max_speed_meters_per_second"], 2)
    return {"players": sorted(players.values(), key=lambda item: item["player_id"])}


COCO_SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
PLAYER_COLORS = {  # BGR
    "near-1": (90, 220, 120),
    "near-2": (40, 175, 255),
    "far-1": (255, 190, 70),
    "far-2": (235, 120, 235),
}


def player_label(player_id: str) -> str:
    side, slot = player_id.split("-")
    return f"{side.upper()} {slot}"


def draw_skeleton(
    frame: np.ndarray, points: np.ndarray, confidences: np.ndarray | None, color: tuple[int, int, int]
) -> None:
    def visible(index: int) -> bool:
        return confidences is None or float(confidences[index]) >= KEYPOINT_CONFIDENCE

    for first, second in COCO_SKELETON:
        if visible(first) and visible(second):
            cv2.line(
                frame,
                tuple(np.rint(points[first]).astype(int)),
                tuple(np.rint(points[second]).astype(int)),
                color,
                3,
                cv2.LINE_AA,
            )
    for index, point in enumerate(points):
        if visible(index):
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 4, color, -1, cv2.LINE_AA)


class PlayerPainter:
    """Drawing state that persists across frames: light box/keypoint smoothing and wrist trails per player."""

    def __init__(self, smoothing: float = 0.4):
        self.smoothing = smoothing
        self.boxes: dict[str, np.ndarray] = {}
        self.keypoints: dict[str, np.ndarray] = {}
        self.trails: dict[str, list[tuple[int, int]]] = {}


def draw_players(frame: np.ndarray, observations: list[PlayerObservation], painter: PlayerPainter) -> int:
    """Draw tracked players with stable colours and labels; returns how many were drawn."""
    drawn = 0
    for observation in observations:
        player_id = observation.player_id
        color = PLAYER_COLORS.get(player_id, (220, 220, 220))
        box = observation.box
        if player_id in painter.boxes:
            box = box * (1.0 - painter.smoothing) + painter.boxes[player_id] * painter.smoothing
        painter.boxes[player_id] = box.copy()
        x1, y1, x2, y2 = (int(value) for value in box)
        thickness = 2 if observation.detected else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = player_label(player_id)
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (8, 12, 10), 1, cv2.LINE_AA)
        drawn += 1
        if not observation.detected or observation.keypoints is None:
            continue
        points = observation.keypoints
        if player_id in painter.keypoints:
            points = points * (1.0 - painter.smoothing) + painter.keypoints[player_id] * painter.smoothing
        painter.keypoints[player_id] = points.copy()
        draw_skeleton(frame, points, observation.confidences, color)
        if observation.wrist is not None:
            height, width = frame.shape[:2]
            trail = painter.trails.setdefault(player_id, [])
            trail.append((round(observation.wrist[0] * width), round(observation.wrist[1] * height)))
            del trail[:-8]
            if len(trail) > 1:
                cv2.polylines(frame, [np.asarray(trail)], False, color, 2, cv2.LINE_AA)
        if observation.swing_score >= 0.75:
            cv2.putText(
                frame,
                "STRONG SWING",
                (x1, min(frame.shape[0] - 8, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (45, 80, 255),
                2,
                cv2.LINE_AA,
            )
    return drawn
