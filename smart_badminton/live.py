"""Real-time analysis of a live camera, network stream or phone camera.

The live path favours latency over completeness: one capture thread keeps only the newest frame, and one analysis
thread processes whatever is newest. On each processed frame it runs court-region motion, tracked-player pose,
optional shuttle detection and an online rally detector. Stats are published several times a second and an annotated
JPEG is kept ready for MJPEG viewers. Recording keeps the full-rate source so the offline pipeline can produce the
accurate, editable cut afterwards.
"""

from __future__ import annotations

import contextlib
import json
import math
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .audio import StreamingHitDetector
from .features import FEATURE_FIELDS, MotionSignals
from .geometry import CourtGeometry
from .players import (
    SIDES,
    CourtFrame,
    PlayerObservation,
    PlayerPainter,
    PlayerTracker,
    PoseEstimator,
    SwingCounter,
    active_player,
    draw_players,
    player_label,
    players_per_side,
    side_features,
    smooth_point,
)

LIVE_START_THRESHOLD = 0.56
LIVE_KEEP_THRESHOLD = 0.30
FILE_SOURCE_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".lrf"}


def parse_source(source: str) -> int | str:
    value = source.strip()
    if not value:
        raise ValueError("Choose a camera number, a stream URL or a video file")
    return int(value) if value.isdigit() else value


def is_file_source(source: str) -> bool:
    return Path(source).suffix.lower() in FILE_SOURCE_SUFFIXES and Path(source).exists()


class LatestFrame:
    """A one-slot mailbox: writers overwrite, readers wait for a sequence newer than the one they have."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.time_seconds = 0.0
        self.sequence = 0
        self.closed = False

    def put(self, frame: np.ndarray, time_seconds: float) -> None:
        with self.condition:
            self.frame, self.time_seconds = frame, time_seconds
            self.sequence += 1
            self.condition.notify_all()

    def wait(self, after: int, timeout: float = 1.0) -> tuple[np.ndarray | None, float, int]:
        with self.condition:
            if self.sequence <= after and not self.closed:
                self.condition.wait(timeout)
            return self.frame, self.time_seconds, self.sequence

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class CaptureSource:
    """OpenCV capture of a camera index, network stream or file. Files are paced to real time like a camera."""

    def __init__(self, source: str, mailbox: LatestFrame, record_path: Path | None = None, loop: bool = False):
        self.source = source
        self.mailbox = mailbox
        self.file = is_file_source(source)
        self.loop = loop
        self.capture = cv2.VideoCapture(parse_source(source))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open live source: {source}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 1.0 <= self.fps <= 240.0:
            self.fps = 30.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.record_path = record_path
        self.writer: cv2.VideoWriter | None = None
        self.frames = 0
        self.finished = threading.Event()
        self.stopping = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="live-capture", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        started = time.monotonic()
        try:
            while not self.stopping.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    if self.file and self.loop:
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                if self.file:
                    # Pace a file to its frame rate so it behaves exactly like a camera.
                    due = started + self.frames / self.fps
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                capture_time = self.frames / self.fps if self.file else time.monotonic() - started
                self.frames += 1
                if self.record_path is not None:
                    self._record(frame)
                self.mailbox.put(frame, capture_time)
        except Exception as error:  # noqa: BLE001 - surfaced through the session status
            self.error = str(error)
        finally:
            self.capture.release()
            if self.writer is not None:
                self.writer.release()
            self.finished.set()
            self.mailbox.close()

    def _record(self, frame: np.ndarray) -> None:
        if self.writer is None:
            height, width = frame.shape[:2]
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.record_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        self.writer.write(frame)

    def stop(self) -> None:
        self.stopping.set()
        self.thread.join(timeout=3.0)


class PushSource:
    """Frames posted by a browser (a phone's camera), decoded on arrival."""

    def __init__(self, mailbox: LatestFrame):
        self.mailbox = mailbox
        self.fps = 0.0
        self.width = self.height = 0
        self.frames = 0
        self.finished = threading.Event()
        self.error: str | None = None
        self.source = "browser"
        self.started = time.monotonic()
        self.record_path = None

    def start(self) -> None:
        self.started = time.monotonic()

    def push(self, encoded: bytes, capture_time: float | None = None) -> None:
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("The uploaded frame is not a JPEG or PNG image")
        self.height, self.width = frame.shape[:2]
        self.frames += 1
        elapsed = time.monotonic() - self.started
        self.fps = self.frames / elapsed if elapsed > 1 else 0.0
        self.mailbox.put(frame, capture_time if capture_time is not None else elapsed)

    def stop(self) -> None:
        self.finished.set()
        self.mailbox.close()


class AudioStream:
    """Live hit sounds: PCM from an FFmpeg reader of the same stream, or pushed by the browser."""

    def __init__(self, ffmpeg: Path | None = None, input_arguments: list[str] | None = None):
        self.detector = StreamingHitDetector(16000)
        self.lock = threading.Lock()
        self.events: deque[tuple[float, float]] = deque(maxlen=400)
        self.raw_scores: deque[float] = deque(maxlen=400)
        self.process: subprocess.Popen[bytes] | None = None
        self.thread: threading.Thread | None = None
        self.samples = 0
        if ffmpeg is not None and input_arguments:
            command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", *input_arguments, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.thread = threading.Thread(target=self._read, name="live-audio", daemon=True)
            self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while True:
            chunk = self.process.stdout.read(3200)  # 100 ms
            if not chunk:
                break
            self.push(np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0)

    def push(self, samples: np.ndarray) -> None:
        with self.lock:
            self.samples += len(samples)
            for event in self.detector.push(samples):
                self.events.append(event)
                self.raw_scores.append(event[1])

    @property
    def active(self) -> bool:
        return self.samples > 0

    def sample(self, time_seconds: float, window: float = 0.12) -> tuple[float, float]:
        """Normalized hit score near ``time_seconds`` and seconds since the previous hit, as offline features."""
        with self.lock:
            if not self.events:
                return 0.0, 999.0
            scores = np.asarray(self.raw_scores)
            low, high = float(np.percentile(scores, 25)), float(np.percentile(scores, 95))
            scale = max(high - low, 1e-6) if len(scores) >= 4 else max(float(scores.max()), 1e-6)
            score = 0.0
            since = 999.0
            for event_time, raw in self.events:
                if abs(event_time - time_seconds) <= window:
                    normalized = (raw - low) / scale if len(scores) >= 4 else raw / scale
                    score = max(score, float(np.clip(normalized, 0.0, 1.0)))
                if event_time <= time_seconds:
                    since = min(since, time_seconds - event_time)
            return score, since

    def close(self) -> None:
        if self.process is not None:
            with contextlib.suppress(OSError):
                self.process.kill()
            self.process.wait(timeout=2)


class OnlineShuttle:
    """Single-frame YOLO shuttle candidates with the offline filters: flight volume, exclusions, static clutter."""

    def __init__(self, model_path: Path, geometry: CourtGeometry, image_size: int, confidence: float = 0.08):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.cuda = bool(torch.cuda.is_available())
        self.device: int | str = 0 if self.cuda else "cpu"
        self.geometry = geometry
        self.image_size = image_size
        self.confidence = confidence
        self.cells: dict[tuple[int, int], tuple[int, float, float]] = {}
        self.samples = 0
        self.previous: tuple[float, float, float] | None = None
        self.trail: deque[tuple[float, float, float]] = deque(maxlen=12)

    def detect(self, frame: np.ndarray, time_seconds: float) -> dict[str, float]:
        height, width = frame.shape[:2]
        precision = {"half": True} if self.cuda else {}
        result = self.model.predict(
            frame, imgsz=self.image_size, conf=self.confidence, device=self.device, verbose=False, **precision
        )[0]
        self.samples += 1
        candidates = []
        if result.boxes is not None:
            for box, score in zip(result.boxes.xyxy.detach().cpu().numpy(), result.boxes.conf.detach().cpu().numpy()):
                x, y = float((box[0] + box[2]) / 2 / width), float((box[1] + box[3]) / 2 / height)
                if not self.geometry.contains_shuttle_volume(x, y) or self.geometry.excluded_normalized(x, y):
                    continue
                cell = (int(x * 48), int(y * 27))
                count, first, last = self.cells.get(cell, (0, time_seconds, time_seconds))
                if time_seconds - last > 10.0:
                    count, first = 0, time_seconds  # a spot forgets clutter it has not seen for 10 s
                self.cells[cell] = (count + 1, first, time_seconds)
                candidates.append((x, y, float(score), cell))
        # Clutter (lights, logos, a shuttle on the floor) keeps re-appearing in the same cell; flight never does.
        dynamic = [
            item
            for item in candidates
            if not (self.cells[item[3]][0] >= 5 and self.cells[item[3]][2] - self.cells[item[3]][1] >= 1.5)
        ]
        values = {
            "shuttle_candidate_count": float(len(candidates)),
            "shuttle_dynamic_count": float(len(dynamic)),
            "shuttle_best_confidence": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_x_normalized": 0.0,
            "shuttle_y_normalized": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
        if not dynamic:
            return values
        previous = self.previous

        def score(item: tuple[float, float, float, tuple[int, int]]) -> float:
            if previous is None or time_seconds - previous[2] > 0.6:
                return item[2]
            return item[2] + max(0.0, 1.0 - math.dist(item[:2], previous[:2]) / 0.25) * 0.45

        x, y, confidence, _cell = max(dynamic, key=score)
        speed = 0.0
        if previous is not None and 0 < time_seconds - previous[2] <= 0.6:
            speed = math.dist((x, y), previous[:2]) / (time_seconds - previous[2])
        self.previous = (x, y, time_seconds)
        self.trail.append((x, y, time_seconds))
        values.update(
            shuttle_best_confidence=confidence,
            shuttle_visible=1.0,
            shuttle_x_normalized=x,
            shuttle_y_normalized=y,
            shuttle_speed_normalized=speed,
        )
        return values


class RallyModelRunner:
    """Runs the trained rally model over a sliding buffer.

    Its centred 4 s windows need 2 s of future context, so the stable answer is the probability 2 s ago; the newest
    row is reported as provisional.
    """

    def __init__(self, model_path: Path, history_seconds: float = 14.0, lag_seconds: float = 2.0):
        from .model import load_model_bundle

        self.bundle = load_model_bundle(model_path)
        self.history_seconds = history_seconds
        self.lag_seconds = lag_seconds
        self.rows: deque[dict[str, float]] = deque()
        self.last_run = -math.inf
        self.stable: tuple[float, float] | None = None
        self.provisional = 0.0

    def add(self, row: dict[str, float]) -> None:
        self.rows.append(row)
        while self.rows and row["time_seconds"] - self.rows[0]["time_seconds"] > self.history_seconds:
            self.rows.popleft()

    def update(self, now: float) -> tuple[float, float] | None:
        if now - self.last_run < 0.5 or len(self.rows) < 5:
            return self.stable
        import pandas as pd

        from .model import build_feature_matrix

        self.last_run = now
        frame = pd.DataFrame(list(self.rows))
        matrix = build_feature_matrix(frame, float(self.bundle["sample_fps"]))
        for name in self.bundle["feature_names"]:
            if name not in matrix:
                matrix[name] = 0.0
        probability = self.bundle["model"].predict_proba(matrix[self.bundle["feature_names"]])[:, 1]
        times = frame["time_seconds"].to_numpy()
        self.provisional = float(probability[-1])
        target = now - self.lag_seconds
        index = int(np.searchsorted(times, target))
        if index < len(times) and times[0] <= target:
            self.stable = (float(times[index]), float(probability[index]))
        return self.stable


@dataclass
class LiveRally:
    index: int
    start: float
    end: float | None = None
    shots: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, (self.end if self.end is not None else self.start) - self.start)

    def payload(self, now: float | None = None) -> dict[str, Any]:
        end = self.end if self.end is not None else now
        return {
            "index": self.index,
            "start": round(self.start, 2),
            "end": None if self.end is None else round(self.end, 2),
            "duration": round(max(0.0, (end or self.start) - self.start), 2),
            "shots": self.shots,
        }


class LiveRallyDetector:
    """Online hysteresis over fused rally evidence, using the offline segmenter's start/keep thresholds."""

    def __init__(self, start_hold: float = 0.3, end_hold: float = 1.4, preroll: float = 0.35, postroll: float = 0.55):
        self.start_hold = start_hold
        self.end_hold = end_hold
        self.preroll = preroll
        self.postroll = postroll
        self.rallies: list[LiveRally] = []
        self.current: LiveRally | None = None
        self.above_since: float | None = None
        self.last_keep: float | None = None
        self.first_keep: float | None = None

    def update(self, time_seconds: float, evidence: float) -> str | None:
        """Feed evidence in [0, 1]; returns "start" or "end" when the rally state changes."""
        if evidence >= LIVE_KEEP_THRESHOLD:
            self.last_keep = time_seconds
            if self.first_keep is None:
                self.first_keep = time_seconds
        elif self.current is None:
            self.first_keep = None
        if self.current is None:
            if evidence >= LIVE_START_THRESHOLD:
                self.above_since = self.above_since if self.above_since is not None else time_seconds
                if time_seconds - self.above_since >= self.start_hold:
                    start = (self.first_keep if self.first_keep is not None else self.above_since) - self.preroll
                    previous_end = self.rallies[-1].end if self.rallies else None
                    self.current = LiveRally(len(self.rallies) + 1, max(start, previous_end or start))
                    return "start"
            else:
                self.above_since = None
            return None
        if self.last_keep is not None and time_seconds - self.last_keep >= self.end_hold:
            self.current.end = self.last_keep + self.postroll
            self.rallies.append(self.current)
            self.current = None
            self.above_since = self.first_keep = None
            return "end"
        return None


def flight_evidence(shuttle: deque[tuple[float, float, float]], now: float) -> float:
    """1.0 when the shuttle has been flying over the last second, from tracked speed and net-direction travel."""
    recent = [point for point in shuttle if now - point[2] <= 1.0]
    if len(recent) < 2:
        return 0.0
    speeds = []
    for previous, current in zip(recent, recent[1:]):
        dt = current[2] - previous[2]
        if 0 < dt <= 0.6:
            speeds.append(math.dist(previous[:2], current[:2]) / dt)
    if not speeds:
        return 0.0
    moving = sum(speed >= 0.22 for speed in speeds) / len(speeds)  # >≈0.22 frame/s is flight, not a carried shuttle
    coverage = min(1.0, len(recent) / 5.0)
    return float(np.clip(moving * coverage * 1.15, 0.0, 1.0))


@dataclass
class PlayerStats:
    player_id: str
    side: str
    distance_meters: float = 0.0
    rally_distance_meters: float = 0.0
    max_speed_meters_per_second: float = 0.0
    swings: int = 0
    hits: int = 0
    last_court: tuple[float, float] | None = None
    last_time: float | None = None
    observation: PlayerObservation | None = None
    trail: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))


class LiveAnalyzer:
    """Per-frame live analysis. Not thread-safe; the session calls it from one thread."""

    def __init__(
        self,
        config: Path,
        width: int,
        height: int,
        match_format: str = "singles",
        pose_model: Path | None = None,
        shuttle_model: Path | None = None,
        rally_model: Path | None = None,
        packages: Path | None = None,
        audio: AudioStream | None = None,
        pose_image_size: int | None = None,
        shuttle_image_size: int | None = None,
    ):
        cuda = False
        if pose_model or shuttle_model:
            import torch

            cuda = bool(torch.cuda.is_available())
        self.config = config
        self.geometry = CourtGeometry.from_json(config)
        self.width, self.height = width, height
        self.match_format = match_format
        self.court = CourtFrame(self.geometry, width, height)
        self.motion = MotionSignals(self.geometry, width, height, 7.0)
        # CPUs get smaller inference sizes so a laptop keeps several analysed frames per second.
        self.pose = (
            PoseEstimator(pose_model, self.geometry, packages, 0.2, pose_image_size or (640 if cuda else 480))
            if pose_model
            else None
        )
        self.tracker = PlayerTracker(self.court, players_per_side(match_format))
        self.shuttle = (
            OnlineShuttle(shuttle_model, self.geometry, shuttle_image_size or (960 if cuda else 640)) if shuttle_model else None
        )
        self.model = RallyModelRunner(rally_model) if rally_model else None
        self.audio = audio
        self.rallies = LiveRallyDetector()
        # Wrist speed is averaged over longer intervals at live frame rates, so swings peak lower than at 10 fps.
        self.swings = SwingCounter(threshold=0.35)
        self.players: dict[str, PlayerStats] = {}
        self.painter = PlayerPainter()
        self.possession: tuple[str, str, float] | None = None
        self.last_shot: tuple[str, float] | None = None
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.observations: list[PlayerObservation] = []
        # The first inference allocates kernels and buffers; do it now rather than on the first live frame.
        if self.pose is not None:
            self.pose.people(self.frame)
        if self.shuttle is not None:
            self.shuttle.model.predict(self.frame, imgsz=self.shuttle.image_size, device=self.shuttle.device, verbose=False)
        self.evidence = 0.0
        self.model_probability: float | None = None
        self.last_time = 0.0

    def process(self, frame: np.ndarray, time_seconds: float) -> list[PlayerObservation]:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.last_time = time_seconds
        row: dict[str, float] = {"time_seconds": time_seconds, "frame_index": 0.0, **self.motion.sample(frame)}
        shuttle_values = self.shuttle.detect(frame, time_seconds) if self.shuttle else {}
        shuttle_point = (
            (shuttle_values["shuttle_x_normalized"], shuttle_values["shuttle_y_normalized"])
            if shuttle_values.get("shuttle_visible")
            else None
        )
        observations: list[PlayerObservation] = []
        if self.pose is not None:
            observations = self.tracker.update(self.pose.detect(frame, self.court, self.match_format == "doubles"), time_seconds)
        row.update(side_features(observations, shuttle_point))
        audio_score, since_audio = self.audio.sample(time_seconds) if self.audio and self.audio.active else (0.0, 999.0)
        row["audio_hit_score"], row["seconds_since_audio_hit"] = audio_score, since_audio
        row.update({name: shuttle_values.get(name, 0.0) for name in FEATURE_FIELDS if name.startswith("shuttle_")})
        self._update_players(observations, time_seconds, shuttle_point)
        self._update_rally(row, time_seconds)
        self.frame = frame
        self.observations = observations
        return observations

    def _update_players(
        self, observations: list[PlayerObservation], time_seconds: float, shuttle: tuple[float, float] | None
    ) -> None:
        seen = set()
        for observation in observations:
            stats = self.players.setdefault(observation.player_id, PlayerStats(observation.player_id, observation.side))
            stats.observation = observation
            seen.add(observation.player_id)
            if not observation.detected:
                continue
            if observation.court is not None:
                court = observation.court
                if stats.last_court is not None and stats.last_time is not None and 0 < time_seconds - stats.last_time <= 0.6:
                    # Smooth foot jitter first; summing raw jitter would inflate distance by a third or more.
                    court = smooth_point(stats.last_court, observation.court)
                    step = math.dist(court, stats.last_court)
                    if step <= 9.0 * (time_seconds - stats.last_time):
                        stats.distance_meters += step
                        if self.rallies.current is not None:
                            stats.rally_distance_meters += step
                stats.last_court, stats.last_time = court, time_seconds
                stats.trail.append(court)
                if observation.speed_meters_per_second is not None:
                    stats.max_speed_meters_per_second = max(
                        stats.max_speed_meters_per_second, min(9.0, observation.speed_meters_per_second)
                    )
            if self.swings.update(observation.player_id, time_seconds, observation.swing_score):
                stats.swings += 1
        for player_id, stats in self.players.items():
            if player_id not in seen:
                stats.observation = None
        if shuttle is None:
            return
        for hitter in self._contact_candidates(observations, shuttle):
            self._register_shot(hitter, time_seconds)
        switched = self._possession_switch(observations, time_seconds, shuttle)
        if switched is not None:
            self._register_shot(switched, time_seconds)

    def _contact_candidates(
        self, observations: list[PlayerObservation], shuttle: tuple[float, float]
    ) -> list[PlayerObservation]:
        """Players whose racket wrist is at the shuttle while swinging (the contact frame itself)."""
        candidates = []
        for side in SIDES:
            hitter = active_player([item for item in observations if item.side == side], shuttle)
            if hitter is None or hitter.wrist is None:
                continue
            radius = float(np.clip(hitter.stance_width_normalized * 2.2, 0.04, 0.12))
            if math.dist(hitter.wrist, shuttle) <= radius and hitter.swing_score >= 0.2:
                candidates.append(hitter)
        return candidates

    def _possession_switch(
        self, observations: list[PlayerObservation], time_seconds: float, shuttle: tuple[float, float]
    ) -> PlayerObservation | None:
        """The previous holder when the shuttle is now clearly with the other side (it crossed the net)."""
        nearest: dict[str, tuple[float, PlayerObservation]] = {}
        for observation in observations:
            if not observation.detected:
                continue
            x1, y1, x2, y2 = observation.box
            # The racket hand works around head height, so measure to the box's upper-middle.
            anchor = ((x1 + x2) / 2 / self.width, (y1 + (y2 - y1) * 0.3) / self.height)
            distance = math.dist(anchor, shuttle)
            if observation.side not in nearest or distance < nearest[observation.side][0]:
                nearest[observation.side] = (distance, observation)
        if not nearest:
            return None
        side, (distance, player) = min(nearest.items(), key=lambda item: item[1][0])
        other = [value[0] for key, value in nearest.items() if key != side]
        if distance > 0.12 or (other and distance > other[0] * 0.6):
            return None  # in the air between the players: neither side clearly holds the shuttle
        owner, self.possession = self.possession, (side, player.player_id, time_seconds)
        if owner is None or owner[0] == side:
            return None
        previous = next((item for item in observations if item.player_id == owner[1]), None)
        return previous if previous is not None and previous.detected else None

    def _register_shot(self, hitter: PlayerObservation, time_seconds: float) -> None:
        """Deduplicate shot evidence: shots alternate sides, so a same-side repeat must be a separate rally beat."""
        if self.last_shot is not None:
            side, last_time = self.last_shot
            gap = time_seconds - last_time
            if gap < 0.35 or (side == hitter.side and gap < 1.2):
                return
        self.last_shot = (hitter.side, time_seconds)
        if self.rallies.current is None:
            return
        self.rallies.current.shots += 1
        if hitter.player_id in self.players:
            self.players[hitter.player_id].hits += 1

    def _update_rally(self, row: dict[str, float], time_seconds: float) -> None:
        flight = flight_evidence(self.shuttle.trail, time_seconds) if self.shuttle else 0.0
        model = None
        if self.model is not None:
            self.model.add(row)
            stable = self.model.update(time_seconds)
            model = stable[1] if stable is not None else None
            self.model_probability = self.model.provisional
        audio = self.audio is not None and self.audio.active
        if model is None:
            evidence = flight
        elif audio:
            # With sound the trained model is reliable; shuttle flight shortens its 2 s lag at rally starts.
            evidence = max(model, flight * 0.9)
        else:
            # Without sound the model is under-confident (it was trained with racket audio), so flight leads.
            evidence = max(flight, min(1.0, model * 1.25))
        self.evidence = evidence
        change = self.rallies.update(time_seconds, evidence)
        if change == "start":
            for stats in self.players.values():
                stats.rally_distance_meters = 0.0
            self.last_shot = None
        elif change == "end" and self.rallies.rallies[-1].shots == 0:
            self.rallies.rallies[-1].shots = 1  # a rally has at least the serve

    def annotate(self, time_seconds: float) -> np.ndarray:
        """Draw the latest analysis onto the frame it was computed from."""
        output = self.frame.copy()
        if self.geometry.active_court_polygon:
            cv2.polylines(output, [self.geometry.polygon("active_court_polygon", self.width, self.height)], True, (60, 60, 230), 2, cv2.LINE_AA)
        draw_players(output, self.observations, self.painter)
        if self.shuttle is not None:
            trail = [point for point in self.shuttle.trail if time_seconds - point[2] <= 0.8]
            pixels = [(round(x * self.width), round(y * self.height)) for x, y, _ in trail]
            if len(pixels) > 1:
                cv2.polylines(output, [np.asarray(pixels)], False, (40, 230, 255), 2, cv2.LINE_AA)
            if pixels:
                cv2.circle(output, pixels[-1], 6, (40, 230, 255), -1, cv2.LINE_AA)
        current = self.rallies.current
        label = (
            f"RALLY {current.index}  {time_seconds - current.start:4.1f}s  shots {current.shots}"
            if current is not None
            else f"BETWEEN RALLIES  ({len(self.rallies.rallies)} played)"
        )
        color = (40, 200, 90) if current is not None else (150, 150, 150)
        cv2.rectangle(output, (10, 10), (24 + 11 * len(label), 44), (12, 14, 12), -1)
        cv2.circle(output, (24, 27), 7, color, -1, cv2.LINE_AA)
        cv2.putText(output, label, (38, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 236), 1, cv2.LINE_AA)
        return output

    def snapshot(self) -> dict[str, Any]:
        now = self.last_time
        rallies = self.rallies.rallies
        durations = [rally.duration for rally in rallies]
        players = []
        for player_id in sorted(self.players):
            stats = self.players[player_id]
            observation = stats.observation
            players.append(
                {
                    "player_id": player_id,
                    "label": player_label(player_id),
                    "side": stats.side,
                    "visible": bool(observation is not None and observation.detected),
                    "speed_meters_per_second": (
                        round(observation.speed_meters_per_second, 2)
                        if observation is not None and observation.speed_meters_per_second is not None
                        else None
                    ),
                    "max_speed_meters_per_second": round(stats.max_speed_meters_per_second, 2),
                    "distance_meters": round(stats.distance_meters, 1),
                    "rally_distance_meters": round(stats.rally_distance_meters, 1),
                    "swings": stats.swings,
                    "hits": stats.hits,
                    "court": None if observation is None or observation.court is None else [round(v, 2) for v in observation.court],
                    "foot": None if observation is None else [round(v, 4) for v in observation.foot],
                    "trail": [[round(x, 2), round(y, 2)] for x, y in stats.trail],
                }
            )
        return {
            "time_seconds": round(now, 2),
            "match_format": self.match_format,
            "metric": self.court.metric,
            "shuttle_enabled": self.shuttle is not None,
            "pose_enabled": self.pose is not None,
            "model_enabled": self.model is not None,
            "audio_enabled": bool(self.audio is not None and self.audio.active),
            "rally": {
                "active": self.rallies.current is not None,
                "evidence": round(self.evidence, 3),
                "model_probability": None if self.model_probability is None else round(self.model_probability, 3),
                "current": None if self.rallies.current is None else self.rallies.current.payload(now),
            },
            "rallies": [rally.payload() for rally in rallies[-30:]],
            "totals": {
                "rallies": len(rallies),
                "average_duration": round(float(np.mean(durations)), 1) if durations else 0.0,
                "longest_duration": round(max(durations), 1) if durations else 0.0,
                "shots": sum(rally.shots for rally in rallies),
            },
            "players": players,
        }


def derive_regions_from_corners(
    corners: list[list[float]], width: int, height: int, match_format: str = "singles"
) -> dict[str, Any]:
    """Build every calibration region from the four singles-court corners (near left → near right → far right → far left).

    The court is a known rectangle, so its homography places the doubles outline, both halves, the net and the
    flight airspace. Doubles widens the active court to the doubles sidelines; player zones always cover the full
    width plus run-off so partners are never cut off.
    """
    from .geometry import SINGLES_CORNERS_METERS

    if len(corners) != 4:
        raise ValueError("Tap the four singles-court corners: near left, near right, far right, far left")
    image = np.asarray([[float(x) * width, float(y) * height] for x, y in corners], dtype=np.float32)
    if not cv2.isContourConvex(image.astype(np.int32)) or cv2.contourArea(image) < width * height * 0.01:
        raise ValueError("The four corners must outline the court in order: near left, near right, far right, far left")
    inverse = np.linalg.inv(cv2.getPerspectiveTransform(image, SINGLES_CORNERS_METERS))

    def project(points: list[tuple[float, float]]) -> list[list[float]]:
        mapped = cv2.perspectiveTransform(np.asarray([points], dtype=np.float32), inverse)[0]
        return [[round(float(np.clip(x / width, 0.0, 1.0)), 5), round(float(np.clip(y / height, 0.0, 1.0)), 5)] for x, y in mapped]

    left, right = (0.0, 6.1) if match_format == "doubles" else (0.46, 5.64)
    active = project([(left, 0.0), (right, 0.0), (right, 13.4), (left, 13.4)])
    near_zone = project([(-1.2, -2.5), (7.3, -2.5), (7.3, 6.45), (-1.2, 6.45)])
    far_zone = project([(-1.2, 6.95), (7.3, 6.95), (7.3, 15.9), (-1.2, 15.9)])
    net_left, net_right = project([(-0.2, 6.7), (6.3, 6.7)])
    # Net posts are 1.55 m tall; at the net line, the court's pixel width per metre approximates vertical scale.
    pixels_per_meter = math.dist(net_left, net_right) * width / 6.5 / height
    net_top = 1.55 * pixels_per_meter
    net_band = [
        [net_left[0], max(0.0, net_left[1] - net_top)],
        [net_right[0], max(0.0, net_right[1] - net_top)],
        [net_right[0], min(1.0, net_right[1] + 0.012)],
        [net_left[0], min(1.0, net_left[1] + 0.012)],
    ]
    outline = np.asarray(project([(-0.6, -1.0), (6.7, -1.0), (6.7, 14.4), (-0.6, 14.4)]))
    top = 0.0
    airspace = [outline[0].tolist(), outline[1].tolist(), [float(outline[2][0]) + 0.04, top], [float(outline[3][0]) - 0.04, top]]
    airspace = [[round(float(np.clip(x, 0.0, 1.0)), 5), round(float(np.clip(y, 0.0, 1.0)), 5)] for x, y in airspace]
    return {
        "court_corners_normalized": [[round(float(x), 5), round(float(y), 5)] for x, y in corners],
        "masks": {
            "active_court_polygon": active,
            "court_ground_polygon": active,
            "near_player_zone": near_zone,
            "far_player_zone": far_zone,
            "net_band": net_band,
            "shuttle_airspace_polygon": airspace,
            "static_false_positive_polygons": [],
            "background_court_polygons": [],
        },
    }


def write_corner_calibration(
    path: Path, corners: list[list[float]], width: int, height: int, match_format: str
) -> Path:
    from .players import MATCH_FORMATS

    if match_format not in MATCH_FORMATS:
        raise ValueError("match_format must be singles or doubles")
    regions = derive_regions_from_corners(corners, width, height, match_format)
    existing: dict[str, Any] = {}
    if path.exists():
        with contextlib.suppress(ValueError):
            existing = json.loads(path.read_text(encoding="utf-8"))
    analysis = {**existing.get("analysis", {}), "match_format": match_format}
    data = {
        **existing,
        "name": existing.get("name", "Live court"),
        "reference_frame": {"width": width, "height": height, "source_time_seconds": 0.0},
        "calibration": {
            **existing.get("calibration", {}),
            "mode": "four singles corners",
            "court_corners_normalized": regions["court_corners_normalized"],
            "shuttle_perspective_axis_normalized": existing.get("calibration", {}).get("shuttle_perspective_axis_normalized", []),
            "shuttle_vanishing_extension": float(existing.get("calibration", {}).get("shuttle_vanishing_extension", 1.5)),
        },
        "masks_normalized": regions["masks"],
        "analysis": {
            "sample_fps": 10.0,
            "pose_fps": 10.0,
            "pose_confidence": 0.2,
            "audio_event_window_seconds": 0.12,
            "motion_threshold": 7,
            **analysis,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


@dataclass
class LiveSettings:
    source: str
    match_format: str = "singles"
    shuttle: bool = True
    record: bool = False
    analysis_width: int = 1280
    stream_width: int = 960
    jpeg_quality: int = 72
    loop: bool = False
    pose_image_size: int | None = None  # default: 640 on CUDA, 480 on CPU
    shuttle_image_size: int | None = None  # default: 960 on CUDA, 640 on CPU


class LiveSession:
    """Owns one live source and its analysis thread."""

    def __init__(
        self,
        settings: LiveSettings,
        config: Path | None,
        recordings: Path,
        pose_model: Path | None = None,
        shuttle_model: Path | None = None,
        rally_model: Path | None = None,
        packages: Path | None = None,
        ffmpeg: Path | None = None,
        analyzer_factory: Callable[..., LiveAnalyzer] = LiveAnalyzer,
    ):
        players_per_side(settings.match_format)
        self.settings = settings
        self.config = config
        self.recordings = recordings
        self.models = {"pose_model": pose_model, "shuttle_model": shuttle_model if settings.shuttle else None, "rally_model": rally_model}
        self.packages = packages
        self.ffmpeg = ffmpeg
        self.analyzer_factory = analyzer_factory
        self.mailbox = LatestFrame()
        self.output = LatestFrame()
        self.lock = threading.Lock()
        self.state = "starting"
        self.error: str | None = None
        self.started_at = time.time()
        self.processed = 0
        self.analyzed_at: deque[float] = deque(maxlen=30)  # wall-clock times of analysed frames
        self.jpeg: bytes | None = None
        self.raw_jpeg: bytes | None = None
        self.raw_size: tuple[int, int] | None = None
        self.analyzer: LiveAnalyzer | None = None
        self.stats: dict[str, Any] = {}
        self.stopping = threading.Event()
        self.recording_path: Path | None = None
        self.recorder: subprocess.Popen[bytes] | None = None
        browser = settings.source == "browser"
        network = "://" in settings.source
        capture_record = None
        if settings.record and not browser and not is_file_source(settings.source):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            if network and ffmpeg is not None:
                # Stream copy keeps the stream's own audio, which the offline rally analysis needs.
                self.recording_path = recordings / f"live-{stamp}.mkv"
                recordings.mkdir(parents=True, exist_ok=True)
                self.recorder = subprocess.Popen(
                    [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", settings.source, "-map", "0", "-c", "copy", "-y", str(self.recording_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                self.recording_path = capture_record = recordings / f"live-{stamp}.mp4"
        self.source: CaptureSource | PushSource = (
            PushSource(self.mailbox) if browser else CaptureSource(settings.source, self.mailbox, capture_record, settings.loop)
        )
        audio_arguments = None
        if not browser and ffmpeg is not None and (is_file_source(settings.source) or "://" in settings.source):
            audio_arguments = (["-re"] if is_file_source(settings.source) else []) + ["-i", settings.source]
        self.audio = AudioStream(ffmpeg, audio_arguments) if (audio_arguments or browser) else None
        self.thread = threading.Thread(target=self._run, name="live-analysis", daemon=True)

    def start(self) -> None:
        self.source.start()
        self.thread.start()

    def calibrated(self) -> bool:
        return bool(self.config and self.config.exists())

    def reload_calibration(self, config: Path, match_format: str | None = None) -> None:
        with self.lock:
            self.config = config
            if match_format:
                self.settings.match_format = match_format
            self.analyzer = None  # rebuilt with the new court on the next frame

    def _ensure_analyzer(self, frame: np.ndarray) -> LiveAnalyzer | None:
        with self.lock:
            if self.analyzer is not None or not self.calibrated():
                return self.analyzer
            height, width = frame.shape[:2]
            scale = min(1.0, self.settings.analysis_width / width)
            self.analyzer = self.analyzer_factory(
                self.config,
                round(width * scale),
                round(height * scale),
                self.settings.match_format,
                packages=self.packages,
                audio=self.audio,
                pose_image_size=self.settings.pose_image_size,
                shuttle_image_size=self.settings.shuttle_image_size,
                **self.models,
            )
            return self.analyzer

    def _run(self) -> None:
        sequence = 0
        try:
            self.state = "running"
            while not self.stopping.is_set():
                frame, capture_time, current = self.mailbox.wait(sequence, timeout=0.5)
                if current == sequence:
                    if self.source.finished.is_set():
                        break
                    continue
                sequence = current
                if frame is None:
                    continue
                self.raw_size = (frame.shape[1], frame.shape[0])
                self.raw_jpeg = None  # encoded lazily for calibration snapshots
                self._raw_frame = frame
                analyzer = self._ensure_analyzer(frame)
                if analyzer is None:
                    annotated = self._resize_for_stream(frame)
                else:
                    analyzer.process(frame, capture_time)
                    annotated = analyzer.annotate(capture_time)
                    self.stats = analyzer.snapshot()
                self.processed += 1
                if analyzer is not None:
                    self.analyzed_at.append(time.monotonic())
                self._publish(annotated, capture_time)
            if self.source.error:
                raise RuntimeError(self.source.error)
            self.state = "finished"
        except Exception as error:  # noqa: BLE001 - surfaced in the status payload
            self.error = str(error)
            self.state = "error"
        finally:
            self.output.close()

    def _resize_for_stream(self, frame: np.ndarray) -> np.ndarray:
        width = frame.shape[1]
        if width <= self.settings.stream_width:
            return frame
        scale = self.settings.stream_width / width
        return cv2.resize(frame, (self.settings.stream_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    def _publish(self, frame: np.ndarray, capture_time: float) -> None:
        streamed = self._resize_for_stream(frame)
        ok, encoded = cv2.imencode(".jpg", streamed, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
        if ok:
            self.jpeg = encoded.tobytes()
            self.output.put(frame, capture_time)

    def snapshot_jpeg(self) -> bytes | None:
        frame = getattr(self, "_raw_frame", None)
        if frame is None:
            return None
        if self.raw_jpeg is None:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            self.raw_jpeg = encoded.tobytes() if ok else None
        return self.raw_jpeg

    def status(self) -> dict[str, Any]:
        times = self.analyzed_at
        rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
        if times and time.monotonic() - times[-1] > 2.0:
            rate = 0.0  # analysis has stalled or the source stopped sending frames
        return {
            "state": self.state,
            "error": self.error,
            "source": self.settings.source,
            "match_format": self.settings.match_format,
            "calibrated": self.calibrated(),
            "started_at": self.started_at,
            "frames_received": self.source.frames,
            "frames_analyzed": self.processed,
            "analysis_fps": round(rate, 1),
            "source_fps": round(self.source.fps, 1),
            "frame_size": list(self.raw_size) if self.raw_size else None,
            "recording": str(self.recording_path) if self.recording_path else None,
            **({"analysis": self.stats} if self.stats else {}),
        }

    def push_frame(self, encoded: bytes, capture_time: float | None = None) -> None:
        if not isinstance(self.source, PushSource):
            raise ValueError("This live session reads its own camera; frames cannot be uploaded")
        self.source.push(encoded, capture_time)

    def push_audio(self, pcm: bytes) -> None:
        if self.audio is None:
            raise ValueError("This live session has no audio input")
        self.audio.push(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)

    def stop(self) -> dict[str, Any]:
        self.stopping.set()
        self.source.stop()
        self.thread.join(timeout=5.0)
        if self.audio is not None:
            self.audio.close()
        if self.recorder is not None:
            with contextlib.suppress(OSError, ValueError):
                self.recorder.communicate(b"q", timeout=5)  # lets FFmpeg finalize the container
            if self.recorder.poll() is None:
                self.recorder.kill()
        if self.state not in {"error"}:
            self.state = "stopped"
        return self.status()

    def frames(self, boundary: str = "frame"):
        """MJPEG parts for ``multipart/x-mixed-replace`` viewers."""
        sequence = 0
        while not self.stopping.is_set():
            _frame, _time, current = self.output.wait(sequence, timeout=1.0)
            if self.output.closed and current == sequence:
                break
            if current == sequence or self.jpeg is None:
                continue
            sequence = current
            payload = self.jpeg
            yield (
                f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
                + payload
                + b"\r\n"
            )
