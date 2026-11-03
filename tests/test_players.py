import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from smart_badminton.contacts import attribute_players
from smart_badminton.geometry import CourtGeometry
from smart_badminton.live import derive_regions_from_corners
from smart_badminton.players import (
    CourtFrame,
    PlayerObservation,
    PlayerTracker,
    PoseDetection,
    court_detections,
    match_format_from_config,
    refine_identities,
    side_features,
    summarize_player_tracks,
)

WIDTH, HEIGHT = 1280, 720
CORNERS = [[0.25, 0.95], [0.75, 0.95], [0.64, 0.45], [0.36, 0.45]]


def court_geometry(tmp_path: Path, match_format: str = "singles") -> CourtGeometry:
    regions = derive_regions_from_corners(CORNERS, WIDTH, HEIGHT, match_format)
    config = tmp_path / f"court-{match_format}.json"
    config.write_text(
        json.dumps(
            {
                "reference_frame": {"width": WIDTH, "height": HEIGHT},
                "calibration": {"court_corners_normalized": regions["court_corners_normalized"]},
                "masks_normalized": regions["masks"],
                "analysis": {"match_format": match_format},
            }
        ),
        encoding="utf-8",
    )
    return CourtGeometry.from_json(config)


def colour(hue_bin: int) -> np.ndarray:
    histogram = np.full(48, 0.002, dtype=np.float32)
    histogram[hue_bin * 4 : hue_bin * 4 + 4] = 0.25
    return histogram / histogram.sum()


def person(court: CourtFrame, x: float, y: float, side: str, hue: int | None = None, wrist_dx: float = 0.0) -> PoseDetection:
    """A synthetic pose standing at court position (x, y) metres."""
    inverse = np.linalg.inv(court.matrix)
    foot = cv2.perspectiveTransform(np.asarray([[[x, y]]], dtype=np.float32), inverse)[0, 0]
    height = 180.0 if side == "near" else 90.0
    box = np.asarray([foot[0] - height * 0.25, foot[1] - height, foot[0] + height * 0.25, foot[1]])
    keypoints = np.tile(foot, (17, 1)).astype(float)
    keypoints[:, 1] -= np.linspace(height, 0, 17)
    keypoints[15] = foot + [-8, 0]
    keypoints[16] = foot + [8, 0]
    keypoints[10] = [foot[0] + 30 + wrist_dx, foot[1] - height * 0.7]
    normalized = (float(foot[0] / court.width), float(foot[1] / court.height))
    return PoseDetection(
        side=side,
        box=box,
        score=0.9,
        keypoints=keypoints,
        confidences=np.ones(17),
        foot=normalized,
        court=court.court_point(normalized),
        appearance=None if hue is None else colour(hue),
    )


def test_singles_corners_map_to_the_singles_sidelines(tmp_path: Path) -> None:
    geometry = court_geometry(tmp_path)
    near_left = geometry.image_to_court((CORNERS[0][0] * WIDTH, CORNERS[0][1] * HEIGHT), WIDTH, HEIGHT)
    far_right = geometry.image_to_court((CORNERS[2][0] * WIDTH, CORNERS[2][1] * HEIGHT), WIDTH, HEIGHT)

    assert near_left == pytest.approx((0.46, 0.0), abs=0.02)
    assert far_right == pytest.approx((5.64, 13.4), abs=0.02)


def test_doubles_calibration_widens_the_active_court(tmp_path: Path) -> None:
    singles = derive_regions_from_corners(CORNERS, WIDTH, HEIGHT, "singles")["masks"]["active_court_polygon"]
    doubles = derive_regions_from_corners(CORNERS, WIDTH, HEIGHT, "doubles")["masks"]["active_court_polygon"]

    assert doubles[0][0] < singles[0][0] and doubles[1][0] > singles[1][0]
    with pytest.raises(ValueError, match="in order"):
        derive_regions_from_corners([CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]], WIDTH, HEIGHT)


def test_match_format_defaults_to_singles_for_missing_or_unreadable_configs(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    assert match_format_from_config(None) == "singles"
    assert match_format_from_config(broken) == "singles"
    assert match_format_from_config({"analysis": {"match_format": "doubles"}}) == "doubles"


def test_singles_tracker_keeps_one_player_per_side_despite_bystanders(tmp_path: Path) -> None:
    court = CourtFrame(court_geometry(tmp_path), WIDTH, HEIGHT)
    tracker = PlayerTracker(court, players=1)
    for step in range(20):
        time_seconds = step * 0.1
        player = person(court, 2.5 + 0.08 * step, 3.0, "near", hue=2)
        official = person(court, 6.6, 1.0, "near", hue=8)  # line judge just outside the sideline
        observations = tracker.update([official, player] if step % 3 else [player], time_seconds)
        near = [item for item in observations if item.side == "near"]
        assert [item.player_id for item in near] == ["near-1"]
        assert near[0].court[0] == pytest.approx(2.5 + 0.08 * step, abs=0.05)


def test_doubles_partners_keep_their_ids_when_they_cross_and_one_is_hidden(tmp_path: Path) -> None:
    court = CourtFrame(court_geometry(tmp_path, "doubles"), WIDTH, HEIGHT)
    tracker = PlayerTracker(court, players=2)
    identities: dict[str, list[str]] = {"red": [], "blue": []}
    for step in range(30):
        time_seconds = step * 0.1
        offset = 0.17 * step
        red = person(court, 0.8 + offset, 3.0 + 0.04 * step, "near", hue=0)  # walks left → right
        blue = person(court, 5.6 - offset, 4.2, "near", hue=7)  # walks right → left, passing red
        detections = [red] if 13 <= step <= 15 else [red, blue]  # red hides blue while they pass
        by_position = {}
        for observation in tracker.update(detections, time_seconds):
            if observation.side == "near" and observation.detected:
                by_position[observation.player_id] = observation.court
        for name, detection in (("red", red), ("blue", blue)):
            if detection in detections:
                match = min(by_position, key=lambda pid: np.hypot(*np.subtract(by_position[pid], detection.court)))
                identities[name].append(match)

    assert len(set(identities["red"])) == 1
    assert len(set(identities["blue"])) == 1
    assert identities["red"][0] != identities["blue"][0]


def test_a_lost_doubles_player_is_reidentified_by_shirt_colour(tmp_path: Path) -> None:
    court = CourtFrame(court_geometry(tmp_path, "doubles"), WIDTH, HEIGHT)
    tracker = PlayerTracker(court, players=2)
    for step in range(5):
        tracker.update([person(court, 1.5, 3.0, "near", hue=1), person(court, 4.5, 3.0, "near", hue=9)], step * 0.1)
    ids = {item.court[0] < 3: item.player_id for item in tracker.update([], 0.5) if item.court is not None}
    # The right-hand player leaves the frame for two seconds and returns on the left.
    for step in range(6, 26):
        tracker.update([person(court, 1.5, 3.0, "near", hue=1)], step * 0.1)
    observations = tracker.update([person(court, 1.4, 5.5, "near", hue=9), person(court, 1.5, 3.0, "near", hue=1)], 2.7)
    returning = next(item for item in observations if item.court and item.court[1] > 5)

    assert returning.player_id == ids[False]


def test_identity_refinement_undoes_a_partner_swap(tmp_path: Path) -> None:
    def observation(pid: str, time_seconds: float, x: float, hue: int) -> PlayerObservation:
        return PlayerObservation(
            player_id=pid,
            side="near",
            status="detected",
            time_seconds=time_seconds,
            confidence=0.9,
            box=np.zeros(4),
            keypoints=None,
            confidences=None,
            foot=(x / 10, 0.8),
            court=(x, 3.0),
            appearance=colour(hue),
        )

    observations = []
    for step in range(60):
        t = step * 0.1
        swapped = step >= 30  # the online tracker exchanged labels when the partners met at step 30
        red_x, blue_x = (1.0 + step * 0.05, 5.0 - step * 0.05)
        if step == 30:
            red_x = blue_x = 2.5
        observations.append(observation("near-2" if swapped else "near-1", t, red_x, 0))
        observations.append(observation("near-1" if swapped else "near-2", t, blue_x, 7))
    refine_identities(observations, players=2)
    red_ids = {item.player_id for item in observations if item.appearance[0] > 0.05}

    assert red_ids == {"near-1"}


def test_side_features_follow_the_partner_nearest_the_shuttle(tmp_path: Path) -> None:
    court = CourtFrame(court_geometry(tmp_path, "doubles"), WIDTH, HEIGHT)
    tracker = PlayerTracker(court, players=2)
    tracker.update([person(court, 1.5, 3.0, "near"), person(court, 4.5, 3.0, "near")], 0.0)
    observations = tracker.update([person(court, 1.5, 3.0, "near", wrist_dx=40), person(court, 4.5, 3.0, "near")], 0.1)
    right = next(item for item in observations if item.court[0] > 3)
    features = side_features(observations, shuttle=right.wrist)

    assert features["near_active_wrist_x_normalized"] == pytest.approx(right.wrist[0])
    assert features["near_swing_score"] == max(item.swing_score for item in observations)
    assert features["far_person_visible"] == 0.0


def test_singles_admission_requires_the_side_zone_but_doubles_trusts_court_bounds(tmp_path: Path) -> None:
    geometry = court_geometry(tmp_path)
    court = CourtFrame(geometry, WIDTH, HEIGHT)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    wide = person(court, 6.0, 3.0, "near")  # doubles alley, outside the singles active court
    people = [(wide.box, wide.score, wide.keypoints, wide.confidences)]

    assert court_detections(frame, people, court, "near", doubles=False) == []
    assert [item.side for item in court_detections(frame, people, court, "near", doubles=True)] == ["near"]


def test_player_summary_reports_distance_speed_swings_and_heatmap() -> None:
    rows = [
        {
            "time_seconds": f"{step * 0.1:.1f}",
            "player_id": "far-1",
            "side": "far",
            "status": "detected",
            "court_x_meters": f"{3.0 + step * 0.2:.2f}",
            "court_y_meters": "10.0",
            "speed_meters_per_second": "2.0",
            "swing_score": "0.9" if step == 5 else "0.1",
        }
        for step in range(11)
    ]
    summary = summarize_player_tracks(rows)["players"][0]

    assert summary["distance_meters"] == pytest.approx(2.0, abs=0.25)  # smoothing trails the path slightly
    assert summary["max_speed_meters_per_second"] == 2.0
    assert summary["swings"] == 1
    assert sum(map(sum, summary["heatmap"])) == 11


def test_contacts_name_the_partner_whose_wrist_met_the_shuttle() -> None:
    events = [{"player": "near", "time_seconds": 1.0, "shuttle_x_normalized": 0.6, "shuttle_y_normalized": 0.7}]
    tracks = pd.DataFrame(
        {
            "time_seconds": [1.0, 1.0, 1.0],
            "player_id": ["near-1", "near-2", "far-1"],
            "side": ["near", "near", "far"],
            "status": ["detected"] * 3,
            "wrist_x_normalized": [0.3, 0.61, 0.6],
            "wrist_y_normalized": [0.7, 0.69, 0.7],
            "foot_x_normalized": [0.3, 0.6, 0.6],
            "foot_y_normalized": [0.9, 0.9, 0.5],
        }
    )
    attribute_players(events, tracks)

    assert events[0]["player_id"] == "near-2"


def test_overhead_views_admit_wide_far_players_and_keep_lost_slots_for_their_own_shirt(tmp_path: Path) -> None:
    # A drone looks straight down: the court is an undistorted rectangle and players' boxes are about square.
    overhead = [[0.05, 0.2], [0.05, 0.8], [0.95, 0.8], [0.95, 0.2]]
    regions = derive_regions_from_corners(overhead, WIDTH, HEIGHT, "doubles")
    config = tmp_path / "overhead.json"
    config.write_text(
        json.dumps(
            {
                "reference_frame": {"width": WIDTH, "height": HEIGHT},
                "calibration": {"court_corners_normalized": regions["court_corners_normalized"]},
                "masks_normalized": regions["masks"],
            }
        ),
        encoding="utf-8",
    )
    court = CourtFrame(CourtGeometry.from_json(config), WIDTH, HEIGHT)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    foot = court.image_point((3.0, 10.0))
    box = np.asarray([foot[0] - 30, foot[1] - 25, foot[0] + 30, foot[1] + 25])
    keypoints = np.tile(foot, (17, 1)).astype(float)
    far = court_detections(frame, [(box, 0.9, keypoints, np.ones(17))], court, "far", doubles=True)

    assert court.upright_view is False
    assert len(far) == 1 and far[0].spawnable

    tracker = PlayerTracker(CourtFrame(court_geometry(tmp_path, "doubles"), WIDTH, HEIGHT), players=1)
    behind = tracker.court
    tracker.update([person(behind, 2.0, 3.0, "near", hue=1)], 0.0)
    # Two seconds later a differently dressed person appears: the lost slot must not adopt them.
    assert tracker.update([person(behind, 2.2, 3.1, "near", hue=8)], 2.0) == []
    assert [item.player_id for item in tracker.update([person(behind, 2.2, 3.1, "near", hue=8)], 5.0)] == ["near-1"]


def test_focused_pose_crops_tracked_players_and_rescans_only_when_needed(tmp_path: Path) -> None:
    from smart_badminton.players import FocusedPoseDetector

    geometry = court_geometry(tmp_path)
    court = CourtFrame(geometry, WIDTH, HEIGHT)
    tracker = PlayerTracker(court, 1)
    near, far = person(court, 3.0, 2.5, "near"), person(court, 3.0, 11.0, "far")

    class Estimator:
        image_size = 480

        def __init__(self) -> None:
            self.geometry = geometry
            self.calls: list[tuple[list[str], int]] = []

        def people_in(self, frame, regions, image_size):
            self.calls.append(([side for side, *_ in regions], image_size))
            found = []
            for side, x1, y1, x2, y2 in regions:
                candidate = near if side == "near" else far
                centre = (candidate.box[0] + candidate.box[2]) / 2
                inside = x1 <= centre <= x2 and y1 <= candidate.box[1] and candidate.box[3] <= y2 + 1
                found.append((side, [(candidate.box, 0.9, candidate.keypoints, candidate.confidences)] if inside else []))
            return found

    estimator = Estimator()
    focused = FocusedPoseDetector(estimator, roi_size=256)  # type: ignore[arg-type]
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for step in range(12):
        time_seconds = step * 0.1
        tracker.update(focused.detect(frame, court, tracker, time_seconds), time_seconds)

    assert estimator.calls[0] == (["near", "far"], 480)  # nobody tracked yet: scan both halves
    assert estimator.calls[1:] == [(["near", "far"], 256)] * 11, estimator.calls  # then one crop per player
    regions, size = focused.regions(frame, tracker, 1.2)
    assert size == 256 and [side for side, *_ in regions] == ["near", "far"]
    near_box = regions[0][1:]
    assert near_box[0] <= near.box[0] and near.box[2] <= near_box[2] and near_box[1] <= near.box[1]

    # The far player vanishes: once missing for 0.6 s that half is scanned again, at most every 0.5 s.
    estimator.calls.clear()
    for step in range(12, 30):
        time_seconds = step * 0.1
        far.box = far.box + 4000  # off-frame: never found
        tracker.update(focused.detect(frame, court, tracker, time_seconds), time_seconds)
    scans = [call for call in estimator.calls if call[1] == 480]
    assert scans and all(sides == ["far"] for sides, _ in scans)
    assert len(scans) <= 4
