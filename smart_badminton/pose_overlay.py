from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2

from .encoding import h264_encoding_arguments, run_ffmpeg_with_encoder_fallback
from .geometry import CourtGeometry
from .io import resolve_ffmpeg
from .players import (
    CourtFrame,
    PlayerPainter,
    PlayerTracker,
    PoseEstimator,
    draw_players,
    match_format_from_config,
    players_per_side,
)


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
    match_format: str | None = None,
) -> Path:
    if end_seconds <= start_seconds:
        raise ValueError("Pose preview end must be after its start")
    geometry = CourtGeometry.from_json(config)
    match_format = match_format or match_format_from_config(config)
    doubles = match_format == "doubles"
    # A low threshold keeps partially occluded players; the tracker discards one-frame ghosts.
    estimator = PoseEstimator(pose_model, geometry, packages, confidence=0.08)
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
    court = CourtFrame(geometry, output_width, output_height)
    tracker = PlayerTracker(court, players_per_side(match_format))
    painter = PlayerPainter()
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
                skip = bool(source_index % stride)
                ok, frame = (capture.grab(), None) if skip else capture.read()  # skipped frames are not decoded
                if not ok:
                    break
                frame_time = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if frame_time > end_seconds + 1.0 / source_fps:
                    break
                source_index += 1
                if skip:
                    continue
                resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
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
                detections = estimator.detect(resized, court, doubles)
                observations = tracker.update(detections, frame_time)
                selected = draw_players(annotated, observations, painter)
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
                    f"RALLY +{relative:05.2f}s  players {selected}",
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
