from __future__ import annotations

import bisect
import csv
import math
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .geometry import CourtGeometry
from .io import write_rows

FEATURE_FIELDS = [
    "time_seconds",
    "frame_index",
    "court_motion_mean",
    "court_motion_fraction",
    "near_motion_mean",
    "near_motion_fraction",
    "far_motion_mean",
    "far_motion_fraction",
    "near_flow_mean",
    "far_flow_mean",
    "near_person_visible",
    "far_person_visible",
    "near_swing_score",
    "far_swing_score",
    "near_lunge_score",
    "far_lunge_score",
    "near_foot_x_normalized",
    "near_foot_y_normalized",
    "far_foot_x_normalized",
    "far_foot_y_normalized",
    "near_foot_speed_normalized",
    "far_foot_speed_normalized",
    "near_stance_width_normalized",
    "far_stance_width_normalized",
    "near_active_wrist_x_normalized",
    "near_active_wrist_y_normalized",
    "far_active_wrist_x_normalized",
    "far_active_wrist_y_normalized",
    "near_active_wrist_speed_normalized",
    "far_active_wrist_speed_normalized",
    "near_active_wrist_visible",
    "far_active_wrist_visible",
    "audio_hit_score",
    "seconds_since_audio_hit",
    "shuttle_candidate_count",
    "shuttle_dynamic_count",
    "shuttle_best_confidence",
    "shuttle_visible",
    "shuttle_x_normalized",
    "shuttle_y_normalized",
    "shuttle_speed_normalized",
]


def _mask_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask != 0]
    return float(np.mean(selected)) if selected.size else 0.0


def _mask_fraction(values: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    selected = values[mask != 0]
    return float(np.mean(selected >= threshold)) if selected.size else 0.0


class AudioEventIndex:
    def __init__(self, path: Path | None):
        self.times: list[float] = []
        self.scores: list[float] = []
        if path is None or not path.exists():
            return
        raw_scores = []
        rows = []
        with path.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                rows.append(row)
                raw_scores.append(float(row.get("score", 0.0)))
        if not rows:
            return
        low = float(np.percentile(raw_scores, 25))
        high = float(np.percentile(raw_scores, 95))
        scale = max(high - low, 1e-6)
        for row, score in zip(rows, raw_scores):
            self.times.append(float(row["time_seconds"]))
            self.scores.append(float(np.clip((score - low) / scale, 0.0, 1.0)))

    def sample(self, time_seconds: float, window: float) -> tuple[float, float]:
        if not self.times:
            return 0.0, 999.0
        index = bisect.bisect_left(self.times, time_seconds)
        candidates = range(max(0, index - 2), min(len(self.times), index + 3))
        score = 0.0
        for candidate in candidates:
            if abs(self.times[candidate] - time_seconds) <= window:
                score = max(score, self.scores[candidate])
        previous = index - 1
        since = time_seconds - self.times[previous] if previous >= 0 else 999.0
        return score, max(0.0, since)


class ShuttleDetectionIndex:
    def __init__(self, path: Path | None, geometry: CourtGeometry, width: int, height: int):
        self.rows: list[dict[str, float]] = []
        self.times: list[float] = []
        self.persistent_cells: set[tuple[int, int]] = set()
        if path is None or not path.exists():
            return

        cell_counts: Counter[tuple[int, int]] = Counter()
        frame_ids = set()
        with path.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                if row.get("status") and row.get("status") not in {"tracked", "recovered", "manual"}:
                    continue
                if row.get("detection_status") == "inpainted":
                    continue
                # Detection and pose extraction may intentionally use different
                # media resolutions (for example, a 720p detection proxy and a
                # 1080p source master). New trajectory files carry their own
                # coordinate dimensions so normalized positions stay aligned.
                coordinate_width = float(row.get("source_width") or width)
                coordinate_height = float(row.get("source_height") or height)
                x = float(row.get("center_x", row.get("x", 0.0))) / coordinate_width
                y = float(row.get("center_y", row.get("y", 0.0))) / coordinate_height
                if not geometry.contains_shuttle_volume(x, y):
                    continue
                if geometry.excluded_normalized(x, y):
                    continue
                item = {
                    "time": float(row["time_seconds"]),
                    "x": x,
                    "y": y,
                    "confidence": float(row.get("confidence", 0.0)) * float(row.get("evidence_weight") or 1.0),
                    "frame": float(row.get("frame", len(frame_ids))),
                }
                self.rows.append(item)
                frame_ids.add(item["frame"])
                cell_counts[(int(x * 48), int(y * 27))] += 1

        total_frames = max(1, len(frame_ids))
        persistence_threshold = max(20, round(total_frames * 0.18))
        self.persistent_cells = {cell for cell, count in cell_counts.items() if count >= persistence_threshold}
        self.rows.sort(key=lambda item: item["time"])
        self.times = [item["time"] for item in self.rows]

    def sample(
        self,
        time_seconds: float,
        window: float,
        previous_point: tuple[float, float] | None,
        previous_time: float | None,
    ) -> tuple[dict[str, float], tuple[float, float] | None]:
        empty = {
            "shuttle_candidate_count": 0.0,
            "shuttle_dynamic_count": 0.0,
            "shuttle_best_confidence": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_x_normalized": 0.0,
            "shuttle_y_normalized": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
        if not self.times:
            return empty, previous_point
        left = bisect.bisect_left(self.times, time_seconds - window)
        right = bisect.bisect_right(self.times, time_seconds + window)
        candidates = self.rows[left:right]
        dynamic = [
            item for item in candidates if (int(item["x"] * 48), int(item["y"] * 27)) not in self.persistent_cells
        ]
        empty["shuttle_candidate_count"] = float(len(candidates))
        empty["shuttle_dynamic_count"] = float(len(dynamic))
        if not dynamic:
            return empty, previous_point

        def score(item: dict[str, float]) -> float:
            continuity = 0.0
            if previous_point is not None:
                distance = math.hypot(item["x"] - previous_point[0], item["y"] - previous_point[1])
                continuity = max(0.0, 1.0 - distance / 0.25)
            return item["confidence"] + continuity * 0.45

        selected = max(dynamic, key=score)
        speed = 0.0
        if previous_point is not None and previous_time is not None and time_seconds > previous_time:
            speed = math.hypot(selected["x"] - previous_point[0], selected["y"] - previous_point[1]) / (
                time_seconds - previous_time
            )
        empty.update(
            {
                "shuttle_best_confidence": selected["confidence"],
                "shuttle_visible": 1.0,
                "shuttle_x_normalized": selected["x"],
                "shuttle_y_normalized": selected["y"],
                "shuttle_speed_normalized": speed,
            }
        )
        return empty, (selected["x"], selected["y"])


class PoseSignals:
    def __init__(self, model_path: Path | None, packages: Path | None, confidence: float):
        self.model = None
        self.device: int | str = "cpu"
        self.confidence = confidence
        self.previous_wrist: dict[tuple[str, int], tuple[float, float]] = {}
        self.previous_ankle_span: dict[str, float] = {}
        self.previous_foot: dict[str, tuple[float, float]] = {}
        if model_path is None:
            return
        import torch

        if packages is not None:
            sys.path.append(str(packages))
        from ultralytics import YOLO

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path))

    def sample(self, frame: np.ndarray, geometry: CourtGeometry, delta_seconds: float) -> dict[str, float]:
        output = {
            "near_person_visible": 0.0,
            "far_person_visible": 0.0,
            "near_swing_score": 0.0,
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_foot_x_normalized": 0.0,
            "near_foot_y_normalized": 0.0,
            "far_foot_x_normalized": 0.0,
            "far_foot_y_normalized": 0.0,
            "near_foot_speed_normalized": 0.0,
            "far_foot_speed_normalized": 0.0,
            "near_stance_width_normalized": 0.0,
            "far_stance_width_normalized": 0.0,
            "near_active_wrist_x_normalized": 0.0,
            "near_active_wrist_y_normalized": 0.0,
            "far_active_wrist_x_normalized": 0.0,
            "far_active_wrist_y_normalized": 0.0,
            "near_active_wrist_speed_normalized": 0.0,
            "far_active_wrist_speed_normalized": 0.0,
            "near_active_wrist_visible": 0.0,
            "far_active_wrist_visible": 0.0,
        }
        if self.model is None:
            return output
        height, width = frame.shape[:2]
        diagonal = math.hypot(width, height)
        selected: dict[str, tuple[float, np.ndarray, np.ndarray | None]] = {}
        inputs = []
        offsets = []
        for side in ("near", "far"):
            x1, y1, x2, y2 = geometry.player_inference_bounds(side, width, height)
            inputs.append(frame[y1:y2, x1:x2])
            offsets.append((x1, y1))
        results = self.model.predict(inputs, imgsz=640, conf=self.confidence, device=self.device, verbose=False)

        for side, result, (offset_x, offset_y) in zip(("near", "far"), results, offsets):
            if result.boxes is None or result.keypoints is None or result.keypoints.xy is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            box_confidence = result.boxes.conf.detach().cpu().numpy()
            keypoints = result.keypoints.xy.detach().cpu().numpy()
            keypoint_conf = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else None
            for index, (local_box, local_points) in enumerate(zip(boxes, keypoints)):
                box = local_box + np.asarray([offset_x, offset_y, offset_x, offset_y])
                points = local_points + np.asarray([offset_x, offset_y])
                confidences = keypoint_conf[index] if keypoint_conf is not None else None
                ankles = [
                    points[keypoint_index]
                    for keypoint_index in (15, 16)
                    if confidences is None or confidences[keypoint_index] >= 0.2
                ]
                foot = np.mean(ankles, axis=0) if ankles else np.asarray([(box[0] + box[2]) / 2.0, box[3]])
                nx, ny = float(foot[0] / width), float(foot[1] / height)
                if not geometry.contains_active_court(nx, ny, margin=0.015):
                    continue
                if not geometry.contains_normalized(f"{side}_player_zone", nx, ny):
                    continue
                area = float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
                score = area * float(box_confidence[index])
                if side not in selected or score > selected[side][0]:
                    selected[side] = (score, points, confidences)

        for side, (_area, points, confidences) in selected.items():
            output[f"{side}_person_visible"] = 1.0
            wrist_speeds = []
            wrist_observations = []
            for keypoint_index in (9, 10):
                if confidences is None or confidences[keypoint_index] >= 0.2:
                    wrist = points[keypoint_index]
                    key = (side, keypoint_index)
                    previous = self.previous_wrist.get(key)
                    speed = 0.0
                    if previous is not None and delta_seconds > 0:
                        speed = (
                            math.hypot(float(wrist[0]) - previous[0], float(wrist[1]) - previous[1])
                            / diagonal
                            / delta_seconds
                        )
                        wrist_speeds.append(speed)
                    wrist_observations.append((speed, float(wrist[0] / width), float(wrist[1] / height)))
                    self.previous_wrist[key] = (float(wrist[0]), float(wrist[1]))
            if wrist_speeds:
                speed = max(wrist_speeds)
                output[f"{side}_swing_score"] = float(np.clip(speed / 1.2, 0.0, 1.0))
            if wrist_observations:
                active_wrist = max(wrist_observations, key=lambda observation: observation[0])
                output[f"{side}_active_wrist_speed_normalized"] = float(np.clip(active_wrist[0], 0.0, 3.0))
                output[f"{side}_active_wrist_x_normalized"] = active_wrist[1]
                output[f"{side}_active_wrist_y_normalized"] = active_wrist[2]
                output[f"{side}_active_wrist_visible"] = 1.0

            ankles = []
            for keypoint_index in (15, 16):
                if confidences is None or confidences[keypoint_index] >= 0.2:
                    ankles.append(points[keypoint_index])
            if len(ankles) == 2:
                span = abs(float(ankles[0][0] - ankles[1][0])) / width
                foot = np.mean(ankles, axis=0)
                foot_normalized = (float(foot[0] / width), float(foot[1] / height))
                previous_foot = self.previous_foot.get(side)
                foot_speed = 0.0
                if previous_foot is not None and delta_seconds > 0:
                    foot_speed = math.hypot(
                        foot_normalized[0] - previous_foot[0],
                        foot_normalized[1] - previous_foot[1],
                    ) / delta_seconds
                output[f"{side}_foot_x_normalized"] = foot_normalized[0]
                output[f"{side}_foot_y_normalized"] = foot_normalized[1]
                output[f"{side}_foot_speed_normalized"] = float(np.clip(foot_speed, 0.0, 3.0))
                output[f"{side}_stance_width_normalized"] = span
                previous_span = self.previous_ankle_span.get(side, span)
                output[f"{side}_lunge_score"] = float(
                    np.clip((span + max(0.0, span - previous_span) * 2.0) / 0.18, 0.0, 1.0)
                )
                self.previous_ankle_span[side] = span
                self.previous_foot[side] = foot_normalized
        return output


def extract_features(
    video: Path,
    config: Path,
    output_csv: Path,
    audio_events: Path | None = None,
    shuttle_detections: Path | None = None,
    pose_model: Path | None = None,
    packages: Path | None = None,
    sample_fps: float | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    geometry = CourtGeometry.from_json(config)
    config_data = __import__("json").loads(config.read_text(encoding="utf-8"))
    analysis = config_data.get("analysis", {})
    sample_fps = float(sample_fps or analysis.get("sample_fps", 10.0))
    motion_threshold = float(analysis.get("motion_threshold", 7.0))
    audio_window = float(analysis.get("audio_event_window_seconds", 0.12))

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / source_fps
    end_seconds = min(float(end_seconds if end_seconds is not None else duration), duration)
    stride = max(1, round(source_fps / sample_fps))
    start_frame = max(0, round(start_seconds * source_fps))
    end_frame = min(frame_count, round(end_seconds * source_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    analysis_width = 320
    analysis_height = max(1, round(height * analysis_width / width))
    masks = {
        name: geometry.polygon_mask(name, analysis_width, analysis_height)
        for name in ("court_ground_polygon", "near_player_zone", "far_player_zone")
    }
    audio = AudioEventIndex(audio_events)
    shuttle = ShuttleDetectionIndex(shuttle_detections, geometry, width, height)
    pose = PoseSignals(pose_model, packages, float(analysis.get("pose_confidence", 0.2)))

    previous_gray: np.ndarray | None = None
    previous_shuttle: tuple[float, float] | None = None
    previous_shuttle_time: float | None = None
    rows = []
    frame_index = start_frame
    last_sample_time = start_seconds

    while frame_index < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % stride:
            frame_index += 1
            continue
        time_seconds = frame_index / source_fps
        resized = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if previous_gray is None:
            difference = np.zeros_like(gray)
            flow_magnitude = np.zeros_like(gray, dtype=np.float32)
        else:
            difference = cv2.absdiff(gray, previous_gray)
            flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 2, 13, 2, 5, 1.1, 0)
            flow_magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        previous_gray = gray

        row: dict[str, float | int] = {
            "time_seconds": time_seconds,
            "frame_index": frame_index,
            "court_motion_mean": _mask_mean(difference, masks["court_ground_polygon"]),
            "court_motion_fraction": _mask_fraction(difference, masks["court_ground_polygon"], motion_threshold),
            "near_motion_mean": _mask_mean(difference, masks["near_player_zone"]),
            "near_motion_fraction": _mask_fraction(difference, masks["near_player_zone"], motion_threshold),
            "far_motion_mean": _mask_mean(difference, masks["far_player_zone"]),
            "far_motion_fraction": _mask_fraction(difference, masks["far_player_zone"], motion_threshold),
            "near_flow_mean": _mask_mean(flow_magnitude, masks["near_player_zone"]),
            "far_flow_mean": _mask_mean(flow_magnitude, masks["far_player_zone"]),
        }
        delta = max(time_seconds - last_sample_time, 1.0 / sample_fps)
        row.update(pose.sample(frame, geometry, delta))
        audio_score, since_audio = audio.sample(time_seconds, audio_window)
        row["audio_hit_score"] = audio_score
        row["seconds_since_audio_hit"] = since_audio
        shuttle_values, selected = shuttle.sample(
            time_seconds,
            0.55 / sample_fps,
            previous_shuttle,
            previous_shuttle_time,
        )
        row.update(shuttle_values)
        if shuttle_values["shuttle_visible"]:
            previous_shuttle = selected
            previous_shuttle_time = time_seconds
        rows.append(row)
        if progress_callback is not None and len(rows) % 50 == 0:
            progress_callback((frame_index - start_frame) / max(1, end_frame - start_frame))
        last_sample_time = time_seconds
        frame_index += 1

    capture.release()
    if progress_callback is not None:
        progress_callback(1.0)
    formatted = []
    for row in rows:
        formatted.append(
            {
                field: f"{float(row.get(field, 0.0)):.6f}" if field not in ("frame_index",) else str(int(row[field]))
                for field in FEATURE_FIELDS
            }
        )
    write_rows(output_csv, FEATURE_FIELDS, formatted)
    print(
        f"features={len(rows)} source_fps={source_fps:.5f} sample_fps={source_fps / stride:.5f} "
        f"range={start_seconds:.3f}-{end_seconds:.3f} output={output_csv}"
    )
