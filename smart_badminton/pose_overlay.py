from __future__ import annotations

import math
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .encoding import h264_encoding_arguments, run_ffmpeg_with_encoder_fallback
from .geometry import CourtGeometry
from .io import resolve_ffmpeg

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


def _side_for_person(
    box: np.ndarray,
    points: np.ndarray,
    confidences: np.ndarray | None,
    geometry: CourtGeometry,
    width: int,
    height: int,
) -> str | None:
    ankles = [
        points[index]
        for index in (15, 16)
        if confidences is None or float(confidences[index]) >= 0.2
    ]
    foot = np.mean(ankles, axis=0) if ankles else np.asarray([(box[0] + box[2]) / 2.0, box[3]])
    nx, ny = float(foot[0] / width), float(foot[1] / height)
    if geometry.contains_normalized("far_player_zone", nx, ny):
        return "far"
    if geometry.contains_normalized("near_player_zone", nx, ny):
        return "near"
    return None


def _zone_crop(frame: np.ndarray, geometry: CourtGeometry, zone: str) -> tuple[np.ndarray, int, int]:
    height, width = frame.shape[:2]
    if zone.endswith("_player_zone"):
        # Player zones classify a detection by its foot point; they are not body
        # crops. Expand upward by perspective-aware headroom so a near player's
        # torso, wrists and racket hand may cross the far-zone image band.
        side = zone.removesuffix("_player_zone")
        x1, y1, x2, y2 = geometry.player_inference_bounds(side, width, height)
    else:
        x, y, crop_width, crop_height = cv2.boundingRect(geometry.polygon(zone, width, height))
        padding_x, padding_y = round(width * 0.04), round(height * 0.035)
        x1, y1 = max(0, x - padding_x), max(0, y - padding_y)
        x2, y2 = min(width, x + crop_width + padding_x), min(height, y + crop_height + padding_y)
    return frame[y1:y2, x1:x2], x1, y1


def _zone_person(
    result: object,
    offset_x: int,
    offset_y: int,
    side: str,
    geometry: CourtGeometry,
    width: int,
    height: int,
    previous_foot: tuple[float, float] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray] | None:
    if result.boxes is None or result.keypoints is None or result.keypoints.xy is None:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    box_confidence = result.boxes.conf.detach().cpu().numpy()
    keypoints = result.keypoints.xy.detach().cpu().numpy()
    confidence = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else None
    selected = None
    zone_center_x = float(np.mean(np.asarray(getattr(geometry, f"{side}_player_zone"))[:, 0]))
    for index, (local_box, local_points) in enumerate(zip(boxes, keypoints)):
        box = local_box + np.asarray([offset_x, offset_y, offset_x, offset_y])
        points = local_points + np.asarray([offset_x, offset_y])
        confidences = confidence[index] if confidence is not None else None
        detected_side = _side_for_person(box, points, confidences, geometry, width, height)
        if detected_side != side:
            continue
        ankles = [
            points[keypoint_index]
            for keypoint_index in (15, 16)
            if confidences is None or float(confidences[keypoint_index]) >= 0.2
        ]
        foot = np.mean(ankles, axis=0) if ankles else np.asarray([(box[0] + box[2]) / 2.0, box[3]])
        if not geometry.contains_active_court(float(foot[0] / width), float(foot[1] / height), margin=0.015):
            continue
        area = float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
        aspect = float((box[3] - box[1]) / max(1.0, box[2] - box[0]))
        if side == "far" and aspect < 1.1:
            continue
        foot_x = float((box[0] + box[2]) / 2.0 / width)
        centrality = max(0.15, 1.0 - abs(foot_x - zone_center_x) / 0.35)
        continuity = 1.0
        if previous_foot is not None:
            distance = math.hypot(float(foot[0]) - previous_foot[0], float(foot[1]) - previous_foot[1])
            continuity = max(0.08, math.exp(-0.5 * (distance / (width * 0.12)) ** 2))
        selection_score = area * float(box_confidence[index]) * centrality * continuity
        if selected is None or selection_score > selected[0]:
            selected = (selection_score, box, points, confidences, foot)
    return selected


def _draw_skeleton(
    frame: np.ndarray,
    points: np.ndarray,
    confidences: np.ndarray | None,
    color: tuple[int, int, int],
) -> None:
    visible = lambda index: confidences is None or float(confidences[index]) >= 0.2
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


def render_pose_overlay(
    video: Path,
    start_seconds: float,
    end_seconds: float,
    output: Path,
    pose_model: Path,
    config: Path,
    packages: Path | None = None,
    ffmpeg: Path | None = None,
    encoder: str = "auto",
    sample_fps: float = 15.0,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    if end_seconds <= start_seconds:
        raise ValueError("Pose preview end must be after its start")
    if packages is not None and str(packages) not in sys.path:
        sys.path.append(str(packages))
    import torch
    from ultralytics import YOLO

    geometry = CourtGeometry.from_json(config)
    model = YOLO(str(pose_model))
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = min(1280, source_width)
    output_height = round(source_height * output_width / source_width)
    output_width -= output_width % 2
    output_height -= output_height % 2
    stride = max(1, round(source_fps / sample_fps))
    actual_fps = source_fps / stride
    capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    previous_wrists: dict[tuple[str, int], tuple[float, float]] = {}
    wrist_trails: dict[tuple[str, int], list[tuple[int, int]]] = {}
    previous_feet: dict[str, tuple[float, float]] = {}
    previous_boxes: dict[str, np.ndarray] = {}
    previous_keypoints: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="badminton-pose-") as temporary_directory:
        intermediate = Path(temporary_directory) / "pose.avi"
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"MJPG"),
            actual_fps,
            (output_width, output_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create temporary pose video")
        source_index = 0
        written = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if frame_time > end_seconds + 1.0 / source_fps:
                    break
                if source_index % stride:
                    source_index += 1
                    continue
                source_index += 1
                resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                zone_inputs = []
                zone_offsets = []
                for side in ("near", "far"):
                    crop, offset_x, offset_y = _zone_crop(resized, geometry, f"{side}_player_zone")
                    zone_inputs.append(crop)
                    zone_offsets.append((offset_x, offset_y))
                zone_results = model.predict(zone_inputs, conf=0.08, imgsz=640, device=device, verbose=False)
                annotated = resized.copy()
                cv2.polylines(
                    annotated,
                    [geometry.polygon("near_player_zone", output_width, output_height)],
                    True,
                    (90, 220, 120),
                    2,
                    cv2.LINE_AA,
                )
                if geometry.active_court_polygon:
                    cv2.polylines(
                        annotated,
                        [geometry.polygon("active_court_polygon", output_width, output_height)],
                        True,
                        (30, 30, 245),
                        3,
                        cv2.LINE_AA,
                    )
                cv2.polylines(
                    annotated,
                    [geometry.polygon("far_player_zone", output_width, output_height)],
                    True,
                    (255, 190, 70),
                    2,
                    cv2.LINE_AA,
                )
                selected = {}
                for side, result, (offset_x, offset_y) in zip(
                    ("near", "far"), zone_results, zone_offsets
                ):
                    person = _zone_person(
                        result,
                        offset_x,
                        offset_y,
                        side,
                        geometry,
                        output_width,
                        output_height,
                        previous_feet.get(side),
                    )
                    if person is not None:
                        selected[side] = person

                for side, (_score, box, points, confidences, foot) in selected.items():
                    previous_feet[side] = (float(foot[0]), float(foot[1]))
                    if side in previous_boxes:
                        box = box * 0.6 + previous_boxes[side] * 0.4
                    if side in previous_keypoints:
                        points = points * 0.6 + previous_keypoints[side] * 0.4
                    previous_boxes[side] = box.copy()
                    previous_keypoints[side] = points.copy()
                    color = (90, 220, 120) if side == "near" else (255, 190, 70)
                    label = "NEAR PLAYER" if side == "near" else "FAR PLAYER"
                    x1, y1, x2, y2 = [int(value) for value in box]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    _draw_skeleton(annotated, points, confidences, color)
                    cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + 112, y1), color, -1)
                    cv2.putText(
                        annotated, label, (x1 + 5, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (8, 12, 10), 1, cv2.LINE_AA
                    )
                    strong_swing = False
                    for wrist_index in (9, 10):
                        if confidences is not None and float(confidences[wrist_index]) < 0.2:
                            continue
                        wrist = (float(points[wrist_index][0]), float(points[wrist_index][1]))
                        key = (side, wrist_index)
                        previous = previous_wrists.get(key)
                        if previous is not None:
                            normalized_speed = math.hypot(wrist[0] - previous[0], wrist[1] - previous[1])
                            normalized_speed /= math.hypot(output_width, output_height) / actual_fps
                            strong_swing = strong_swing or normalized_speed / 1.2 >= 0.75
                        previous_wrists[key] = wrist
                        trail = wrist_trails.setdefault(key, [])
                        trail.append((round(wrist[0]), round(wrist[1])))
                        del trail[:-8]
                        if len(trail) > 1:
                            cv2.polylines(annotated, [np.asarray(trail)], False, color, 2, cv2.LINE_AA)
                        cv2.circle(annotated, trail[-1], 5, color, -1, cv2.LINE_AA)
                    if strong_swing:
                        cv2.putText(
                            annotated,
                            "STRONG SWING",
                            (x1, min(output_height - 8, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (45, 80, 255),
                            2,
                            cv2.LINE_AA,
                        )
                relative = max(0.0, frame_time - start_seconds)
                cv2.rectangle(annotated, (12, 12), (440, 70), (8, 12, 10), -1)
                cv2.putText(
                    annotated,
                    "YOLO POSE DEBUG - HIT CANDIDATES",
                    (24, 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (225, 255, 235),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"RALLY +{relative:05.2f}s  people {len(selected)}",
                    (24, 59),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (170, 190, 178),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                written += 1
                if progress_callback is not None and (written == 1 or written % 30 == 0):
                    progress_callback(
                        min(0.995, max(0.0, (frame_time - start_seconds) / (end_seconds - start_seconds)))
                    )
        finally:
            capture.release()
            writer.release()
        if written == 0:
            raise RuntimeError("No frames were available for the pose preview")

        duration = end_seconds - start_seconds
        ffmpeg_path = resolve_ffmpeg(ffmpeg)

        def command_for(video_encoder: str) -> list[str]:
            return [
                str(ffmpeg_path),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(intermediate),
                "-ss",
                f"{start_seconds:.6f}",
                "-t",
                f"{duration:.6f}",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                *h264_encoding_arguments(video_encoder, 23, "preview"),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output),
            ]

        run_ffmpeg_with_encoder_fallback(ffmpeg_path, encoder, command_for, output)
        if progress_callback is not None:
            progress_callback(1.0)
    return output
