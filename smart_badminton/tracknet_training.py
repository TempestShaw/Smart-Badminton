from __future__ import annotations

import csv
import random
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .geometry import CourtGeometry
from .shuttle_annotations import load_shuttle_annotations
from .tracknet import (
    TRACKNET_HEIGHT,
    TRACKNET_WIDTH,
    _build_models,
    _crop_bounds,
    _estimate_background,
    _load_checkpoint,
    _prepare_sequence,
    _target_mask,
    _tracknet_input_channels,
    _transform_frame,
)


def export_tracknet_labels(
    video: Path,
    trajectory_csv: Path,
    output_csv: Path,
    annotations_csv: Path | None = None,
) -> dict[str, int | str]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError("Video metadata is incomplete")

    points: dict[int, tuple[float, float, float]] = {}
    with trajectory_csv.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("status") not in {"tracked", "recovered", "manual"}:
                continue
            if row.get("detection_status") == "inpainted":
                continue
            source_width = float(row.get("source_width") or width)
            source_height = float(row.get("source_height") or height)
            frame = int(row["frame"])
            confidence = float(row.get("confidence") or 0.0) * float(row.get("evidence_weight") or 1.0)
            candidate = (
                float(row["center_x"]) / source_width * width,
                float(row["center_y"]) / source_height * height,
                confidence,
            )
            if frame not in points or candidate[2] > points[frame][2]:
                points[frame] = candidate

    annotations = load_shuttle_annotations(annotations_csv)
    for annotation in annotations:
        frame = int(annotation["frame"])
        if annotation["action"] == "reject":
            for nearby in range(max(0, frame - 1), min(frame_count, frame + 2)):
                points.pop(nearby, None)
        else:
            points[frame] = (
                float(annotation["x_normalized"]) * width,
                float(annotation["y_normalized"]) * height,
                1.0,
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["Frame", "Visibility", "X", "Y", "Confidence"])
        writer.writeheader()
        for frame in range(frame_count):
            point = points.get(frame)
            writer.writerow(
                {
                    "Frame": frame,
                    "Visibility": 1 if point else 0,
                    "X": f"{point[0]:.3f}" if point else 0,
                    "Y": f"{point[1]:.3f}" if point else 0,
                    "Confidence": f"{point[2]:.6f}" if point else 0,
                }
            )
    return {
        "frames": frame_count,
        "visible_frames": len(points),
        "manual_points": sum(annotation["action"] == "add" for annotation in annotations),
        "output": str(output_csv),
    }


def _read_labels(path: Path) -> dict[int, tuple[float, float] | None]:
    labels: dict[int, tuple[float, float] | None] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            frame = int(row["Frame"])
            labels[frame] = (float(row["X"]), float(row["Y"])) if int(row["Visibility"]) else None
    return labels


class _FrameCache:
    def __init__(self, video: Path, bounds: tuple[int, int, int, int], capacity: int = 512):
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        self.bounds = bounds
        self.capacity = capacity
        self.frames: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame_index: int) -> np.ndarray:
        if frame_index in self.frames:
            self.frames.move_to_end(frame_index)
            return self.frames[frame_index]
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Could not read training frame {frame_index}")
        transformed = _transform_frame(frame, self.bounds)
        self.frames[frame_index] = transformed
        if len(self.frames) > self.capacity:
            self.frames.popitem(last=False)
        return transformed

    def close(self) -> None:
        self.capture.release()


def _training_starts(labels: dict[int, tuple[float, float] | None], sequence_length: int) -> list[int]:
    if not labels:
        return []
    last_frame = max(labels)
    positive = []
    negative = []
    for start in range(max(0, last_frame - sequence_length + 2)):
        count = sum(labels.get(frame) is not None for frame in range(start, start + sequence_length))
        if count:
            positive.append(start)
        elif start % (sequence_length * 3) == 0:
            negative.append(start)
    return positive + negative


def _target_heatmaps(
    labels: dict[int, tuple[float, float] | None],
    start: int,
    sequence_length: int,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    result = np.zeros((sequence_length, TRACKNET_HEIGHT, TRACKNET_WIDTH), dtype=np.float32)
    x1, y1, x2, y2 = bounds
    crop_width, crop_height = x2 - x1, y2 - y1
    for offset in range(sequence_length):
        point = labels.get(start + offset)
        if point is None:
            continue
        x = round((point[0] - x1) / crop_width * TRACKNET_WIDTH)
        y = round((point[1] - y1) / crop_height * TRACKNET_HEIGHT)
        if 0 <= x < TRACKNET_WIDTH and 0 <= y < TRACKNET_HEIGHT:
            cv2.circle(result[offset], (x, y), 2, 1.0, -1)
    return result


def train_tracknet(
    video: Path,
    config: Path,
    labels_csv: Path,
    base_checkpoint: Path,
    output_checkpoint: Path,
    epochs: int = 8,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    maximum_windows: int = 6000,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, int | float | str]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("TrackNet training requires PyTorch. Install smart-badminton[tracknet].") from error

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    geometry = CourtGeometry.from_json(config)
    bounds = _crop_bounds(geometry, width, height, 0.04, True)
    mask = _target_mask(geometry, width, height, bounds)
    background = _estimate_background(video, bounds, frame_count, 160)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(torch, base_checkpoint, device)
    parameters = checkpoint.get("param_dict", {})
    sequence_length = int(parameters.get("seq_len", 8))
    background_mode = str(parameters.get("bg_mode", "concat"))
    TrackNet, _InpaintNet = _build_models(torch)
    model = TrackNet(_tracknet_input_channels(sequence_length, background_mode), sequence_length).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    labels = _read_labels(labels_csv)
    starts = _training_starts(labels, sequence_length)
    if not starts:
        raise RuntimeError("No visible shuttle labels are available for TrackNet training")
    random.Random(20260821).shuffle(starts)
    starts = starts[:maximum_windows]
    cache = _FrameCache(video, bounds)
    losses = []
    try:
        for epoch in range(epochs):
            random.Random(20260821 + epoch).shuffle(starts)
            for batch_start in range(0, len(starts), batch_size):
                batch_indices = starts[batch_start : batch_start + batch_size]
                inputs = []
                targets = []
                for start in batch_indices:
                    frames = [cache.get(frame) for frame in range(start, start + sequence_length)]
                    inputs.append(_prepare_sequence(frames, background, mask, background_mode))
                    targets.append(_target_heatmaps(labels, start, sequence_length, bounds))
                x = torch.from_numpy(np.stack(inputs)).to(device)
                y = torch.from_numpy(np.stack(targets)).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                weights = 1.0 + y * 7.0
                loss = torch.nn.functional.binary_cross_entropy(prediction, y, weight=weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if progress_callback is not None:
                    completed = epoch * len(starts) + min(len(starts), batch_start + len(batch_indices))
                    progress_callback(completed / max(1, epochs * len(starts)))
    finally:
        cache.close()

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    updated = {
        **checkpoint,
        "model": model.state_dict(),
        "param_dict": {**parameters, "seq_len": sequence_length, "bg_mode": background_mode},
        "smart_badminton_training": {
            "video": str(video),
            "labels": str(labels_csv),
            "epochs": epochs,
            "windows": len(starts),
            "crop": list(bounds),
        },
    }
    torch.save(updated, output_checkpoint)
    return {
        "epochs": epochs,
        "windows": len(starts),
        "final_loss": losses[-1],
        "device": str(device),
        "output": str(output_checkpoint),
    }

