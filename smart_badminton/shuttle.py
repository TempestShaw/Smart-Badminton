from __future__ import annotations

import csv
import sys
from collections.abc import Callable
from pathlib import Path

import cv2

from .geometry import CourtGeometry


def detect_shuttle(
    video: Path,
    config: Path,
    model_path: Path,
    packages: Path | None,
    output_csv: Path,
    sample_fps: float = 15.0,
    confidence: float = 0.04,
    image_size: int = 1280,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, int | float | str]:
    # Import CUDA PyTorch before appending the isolated Ultralytics directory;
    # that directory may contain a CPU-only torch wheel.
    import torch

    if packages is not None:
        sys.path.append(str(packages))
    from ultralytics import YOLO

    geometry = CourtGeometry.from_json(config)
    model = YOLO(str(model_path))
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_seconds = min(end_seconds if end_seconds is not None else frame_count / source_fps, frame_count / source_fps)
    start_frame, end_frame = round(start_seconds * source_fps), round(end_seconds * source_fps)
    stride = max(1, round(source_fps / sample_fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows = []
    sampled = 0
    expected_samples = max(1, (max(0, end_frame - start_frame) + stride - 1) // stride)
    report_stride = max(1, expected_samples // 100)
    frame_index = start_frame
    while frame_index < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % stride == 0:
            sampled += 1
            result = model.predict(frame, imgsz=image_size, conf=confidence, device=device, verbose=False)[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    nx, ny = center_x / width, center_y / height
                    if not geometry.contains_shuttle_volume(nx, ny):
                        continue
                    if geometry.excluded_normalized(nx, ny):
                        continue
                    rows.append(
                        {
                            "time_seconds": f"{frame_index / source_fps:.6f}",
                            "frame": frame_index,
                            "source_width": width,
                            "source_height": height,
                            "confidence": f"{float(score):.6f}",
                            "center_x": f"{center_x:.3f}",
                            "center_y": f"{center_y:.3f}",
                            "width": f"{x2 - x1:.3f}",
                            "height": f"{y2 - y1:.3f}",
                            "source": "yolo",
                            "detection_status": "detected",
                            "evidence_weight": "0.650000",
                        }
                    )
            if progress_callback is not None and (sampled % report_stride == 0 or sampled == expected_samples):
                progress_callback(min(1.0, sampled / expected_samples))
        frame_index += 1
    capture.release()
    if progress_callback is not None:
        progress_callback(1.0)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_seconds",
        "frame",
        "source_width",
        "source_height",
        "confidence",
        "center_x",
        "center_y",
        "width",
        "height",
        "source",
        "detection_status",
        "evidence_weight",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"sampled_frames": sampled, "detections_in_roi": len(rows), "device": str(device), "output": str(output_csv)}
