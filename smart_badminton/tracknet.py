"""TrackNetV3-compatible inference adapted from qaz812345/TrackNetV3.

Upstream copyright (c) 2024 qaz812345, used under the MIT License.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import CourtGeometry

TRACKNET_WIDTH = 512
TRACKNET_HEIGHT = 288


@dataclass(frozen=True)
class TrackNetRuntimeConfig:
    batch_size: int = 4
    heatmap_threshold: float = 0.50
    minimum_confidence: float = 0.42
    background_samples: int = 160
    crop_padding: float = 0.04
    use_airspace_crop: bool = True
    maximum_inpaint_gap_frames: int = 18


def _load_checkpoint(torch, path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_models(torch):
    nn = torch.nn

    class Conv2DBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
            self.bn = nn.BatchNorm2d(out_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Double2DConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.conv_1 = Conv2DBlock(in_dim, out_dim)
            self.conv_2 = Conv2DBlock(out_dim, out_dim)

        def forward(self, x):
            return self.conv_2(self.conv_1(x))

    class Triple2DConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.conv_1 = Conv2DBlock(in_dim, out_dim)
            self.conv_2 = Conv2DBlock(out_dim, out_dim)
            self.conv_3 = Conv2DBlock(out_dim, out_dim)

        def forward(self, x):
            return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.down_block_1 = Double2DConv(in_dim, 64)
            self.down_block_2 = Double2DConv(64, 128)
            self.down_block_3 = Triple2DConv(128, 256)
            self.bottleneck = Triple2DConv(256, 512)
            self.up_block_1 = Triple2DConv(768, 256)
            self.up_block_2 = Double2DConv(384, 128)
            self.up_block_3 = Double2DConv(192, 64)
            self.predictor = nn.Conv2d(64, out_dim, (1, 1))
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x1 = self.down_block_1(x)
            x2 = self.down_block_2(nn.functional.max_pool2d(x1, 2))
            x3 = self.down_block_3(nn.functional.max_pool2d(x2, 2))
            x = self.bottleneck(nn.functional.max_pool2d(x3, 2))
            x = self.up_block_1(torch.cat([nn.functional.interpolate(x, scale_factor=2), x3], dim=1))
            x = self.up_block_2(torch.cat([nn.functional.interpolate(x, scale_factor=2), x2], dim=1))
            x = self.up_block_3(torch.cat([nn.functional.interpolate(x, scale_factor=2), x1], dim=1))
            return self.sigmoid(self.predictor(x))

    class Conv1DBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding="same", bias=True)
            self.relu = nn.LeakyReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    class Double1DConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.conv_1 = Conv1DBlock(in_dim, out_dim)
            self.conv_2 = Conv1DBlock(out_dim, out_dim)

        def forward(self, x):
            return self.conv_2(self.conv_1(x))

    class InpaintNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_1 = Conv1DBlock(3, 32)
            self.down_2 = Conv1DBlock(32, 64)
            self.down_3 = Conv1DBlock(64, 128)
            self.buttleneck = Double1DConv(128, 256)
            self.up_1 = Conv1DBlock(384, 128)
            self.up_2 = Conv1DBlock(192, 64)
            self.up_3 = Conv1DBlock(96, 32)
            self.predictor = nn.Conv1d(32, 2, 3, padding="same")
            self.sigmoid = nn.Sigmoid()

        def forward(self, x, mask):
            x = torch.cat([x, mask], dim=2).permute(0, 2, 1)
            x1 = self.down_1(x)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)
            x = self.buttleneck(x3)
            x = self.up_1(torch.cat([x, x3], dim=1))
            x = self.up_2(torch.cat([x, x2], dim=1))
            x = self.up_3(torch.cat([x, x1], dim=1))
            return self.sigmoid(self.predictor(x)).permute(0, 2, 1)

    return TrackNet, InpaintNet


def _tracknet_input_channels(sequence_length: int, background_mode: str) -> int:
    if background_mode == "subtract":
        return sequence_length
    if background_mode == "subtract_concat":
        return sequence_length * 4
    if background_mode == "concat":
        return (sequence_length + 1) * 3
    return sequence_length * 3


def _crop_bounds(
    geometry: CourtGeometry,
    width: int,
    height: int,
    padding: float,
    enabled: bool,
) -> tuple[int, int, int, int]:
    if not enabled:
        return 0, 0, width, height
    polygon = geometry.projected_shuttle_volume_polygon(width, height)
    if len(polygon) < 3:
        return 0, 0, width, height
    x, y, crop_width, crop_height = cv2.boundingRect(polygon)
    pad_x, pad_y = round(width * padding), round(height * padding)
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(width, x + crop_width + pad_x),
        min(height, y + crop_height + pad_y),
    )


def _target_mask(
    geometry: CourtGeometry,
    source_width: int,
    source_height: int,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    source = np.zeros((source_height, source_width), dtype=np.uint8)
    volume = geometry.projected_shuttle_volume_polygon(source_width, source_height)
    if len(volume) >= 3:
        cv2.fillPoly(source, [volume], 255)
    excluded = geometry.exclusion_mask(source_width, source_height)
    source[excluded > 0] = 0
    x1, y1, x2, y2 = bounds
    return cv2.resize(source[y1:y2, x1:x2], (TRACKNET_WIDTH, TRACKNET_HEIGHT), interpolation=cv2.INTER_NEAREST)


def _transform_frame(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, (TRACKNET_WIDTH, TRACKNET_HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _estimate_background(
    video: Path,
    bounds: tuple[int, int, int, int],
    frame_count: int,
    maximum_samples: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    indices = np.linspace(0, max(0, frame_count - 1), min(maximum_samples, max(1, frame_count)), dtype=int)
    samples: list[np.ndarray] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                samples.append(_transform_frame(frame, bounds))
    finally:
        capture.release()
    if not samples:
        raise RuntimeError("Could not sample frames for TrackNet background")
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def _prepare_sequence(
    frames: list[np.ndarray],
    background: np.ndarray,
    mask: np.ndarray,
    background_mode: str,
) -> np.ndarray:
    masked_frames = []
    outside = mask == 0
    for frame in frames:
        prepared = frame.copy()
        prepared[outside] = background[outside]
        masked_frames.append(prepared)
    channels = []
    for frame in masked_frames:
        if background_mode == "subtract":
            diff = np.clip(
                np.sum(np.abs(frame.astype(np.int16) - background.astype(np.int16)), axis=2), 0, 255
            ).astype(np.uint8)
            channels.append(diff[None])
        elif background_mode == "subtract_concat":
            diff = np.clip(
                np.sum(np.abs(frame.astype(np.int16) - background.astype(np.int16)), axis=2), 0, 255
            ).astype(np.uint8)[None]
            channels.append(np.concatenate([np.moveaxis(frame, -1, 0), diff], axis=0))
        else:
            channels.append(np.moveaxis(frame, -1, 0))
    if background_mode == "concat":
        channels.insert(0, np.moveaxis(background, -1, 0))
    return np.concatenate(channels, axis=0).astype(np.float32) / 255.0


def _heatmap_point(heatmap: np.ndarray, threshold: float) -> tuple[float, float, float] | None:
    binary = (heatmap >= threshold).astype(np.uint8)
    components, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if components <= 1:
        return None
    best_label = max(range(1, components), key=lambda label: float(heatmap[labels == label].sum()))
    component = labels == best_label
    weights = heatmap[component].astype(np.float64)
    ys, xs = np.nonzero(component)
    if not len(xs) or float(weights.sum()) <= 0:
        return None
    x = float(np.average(xs, weights=weights))
    y = float(np.average(ys, weights=weights))
    confidence = float(np.max(weights))
    area = int(stats[best_label, cv2.CC_STAT_AREA])
    if area > 220:
        confidence *= 0.45
    return x, y, confidence


def _robust_frame_point(candidates: list[tuple[float, float, float, float]]) -> tuple[float, float, float] | None:
    if not candidates:
        return None
    points = np.asarray([(item[0], item[1]) for item in candidates], dtype=float)
    weights = np.asarray([item[2] * item[3] for item in candidates], dtype=float)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    support = ((distances <= 28.0) * weights[None, :]).sum(axis=1)
    anchor = int(np.argmax(support))
    keep = distances[anchor] <= 28.0
    kept_weights = np.maximum(weights[keep], 1e-6)
    point = np.average(points[keep], axis=0, weights=kept_weights)
    confidence = float(np.average([item[2] for item, selected in zip(candidates, keep) if selected], weights=kept_weights))
    return float(point[0]), float(point[1]), confidence


def _sequence_weights(length: int) -> np.ndarray:
    center = (length - 1) / 2.0
    distance = np.abs(np.arange(length, dtype=float) - center)
    return 1.0 - 0.45 * distance / max(1.0, center)


def _frame_windows(
    video: Path,
    bounds: tuple[int, int, int, int],
    sequence_length: int,
    start_frame: int,
    end_frame: int,
) -> Iterator[tuple[int, list[np.ndarray]]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    window: deque[np.ndarray] = deque(maxlen=sequence_length)
    frame_index = start_frame
    try:
        while frame_index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            window.append(_transform_frame(frame, bounds))
            if len(window) == sequence_length:
                yield frame_index - sequence_length + 1, list(window)
            frame_index += 1
    finally:
        capture.release()


def _inpaint_mask(rows: list[dict[str, float | int | str]], maximum_gap: int) -> np.ndarray:
    visible = np.asarray([row["detection_status"] == "detected" for row in rows], dtype=bool)
    y = np.asarray([float(row["center_y"]) for row in rows], dtype=float)
    mask = np.zeros(len(rows), dtype=np.float32)
    index = 0
    while index < len(rows):
        if visible[index]:
            index += 1
            continue
        end = index
        while end < len(rows) and not visible[end]:
            end += 1
        surrounded = index > 0 and end < len(rows)
        if surrounded and end - index <= maximum_gap and y[index - 1] > 0 and y[end] > 0:
            mask[index:end] = 1.0
        index = end
    return mask


def _run_inpaint(
    torch,
    model,
    rows: list[dict[str, float | int | str]],
    sequence_length: int,
    device,
    maximum_gap: int,
) -> None:
    mask = _inpaint_mask(rows, maximum_gap)
    if not np.any(mask):
        return
    coordinates = np.asarray(
        [[float(row["center_x"]) / float(row["source_width"]), float(row["center_y"]) / float(row["source_height"])] for row in rows],
        dtype=np.float32,
    )
    accumulators: dict[int, list[np.ndarray]] = defaultdict(list)
    for start in range(max(1, len(rows) - sequence_length + 1)):
        end = start + sequence_length
        if end > len(rows) or not np.any(mask[start:end]):
            continue
        coor = torch.from_numpy(coordinates[start:end][None]).to(device)
        repair = torch.from_numpy(mask[start:end][None, :, None]).to(device)
        with torch.inference_mode():
            prediction = model(coor, repair).detach().cpu().numpy()[0]
        for offset in np.flatnonzero(mask[start:end] > 0):
            accumulators[start + int(offset)].append(prediction[int(offset)])
    for index, values in accumulators.items():
        point = np.median(np.asarray(values), axis=0)
        if not (0.0 < point[0] < 1.0 and 0.0 < point[1] < 1.0):
            continue
        rows[index]["center_x"] = float(point[0]) * float(rows[index]["source_width"])
        rows[index]["center_y"] = float(point[1]) * float(rows[index]["source_height"])
        rows[index]["confidence"] = 0.28
        rows[index]["detection_status"] = "inpainted"


def detect_tracknet(
    video: Path,
    config: Path,
    tracknet_model_path: Path,
    output_csv: Path,
    inpaint_model_path: Path | None = None,
    packages: Path | None = None,
    runtime: TrackNetRuntimeConfig | None = None,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, int | float | str]:
    if packages is not None and str(packages) not in sys.path:
        sys.path.insert(0, str(packages))
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("TrackNet requires PyTorch. Install smart-badminton[tracknet].") from error

    runtime = runtime or TrackNetRuntimeConfig()
    geometry = CourtGeometry.from_json(config)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if source_fps <= 0 or frame_count <= 0:
        raise RuntimeError("Video metadata is incomplete")

    start_frame = max(0, round(start_seconds * source_fps))
    end_frame = min(frame_count, round(end_seconds * source_fps) if end_seconds is not None else frame_count)
    bounds = _crop_bounds(geometry, source_width, source_height, runtime.crop_padding, runtime.use_airspace_crop)
    mask = _target_mask(geometry, source_width, source_height, bounds)
    background = _estimate_background(video, bounds, frame_count, runtime.background_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(torch, tracknet_model_path, device)
    parameters = checkpoint.get("param_dict", {})
    sequence_length = int(parameters.get("seq_len", 8))
    background_mode = str(parameters.get("bg_mode", "concat"))
    TrackNet, InpaintNet = _build_models(torch)
    model = TrackNet(_tracknet_input_channels(sequence_length, background_mode), sequence_length).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    candidates: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    batch_starts: list[int] = []
    batch_inputs: list[np.ndarray] = []
    temporal_weights = _sequence_weights(sequence_length)
    total_windows = max(1, end_frame - start_frame - sequence_length + 1)
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not batch_inputs:
            return
        tensor = torch.from_numpy(np.stack(batch_inputs)).to(device)
        with torch.inference_mode():
            predictions = model(tensor).detach().cpu().numpy()
        for start, sequence in zip(batch_starts, predictions):
            for offset, heatmap in enumerate(sequence):
                point = _heatmap_point(heatmap, runtime.heatmap_threshold)
                if point is not None:
                    candidates[start + offset].append((*point, float(temporal_weights[offset])))
        processed += len(batch_inputs)
        batch_inputs.clear()
        batch_starts.clear()
        if progress_callback is not None:
            progress_callback(min(0.88, processed / total_windows * 0.88))

    for start, frames in _frame_windows(video, bounds, sequence_length, start_frame, end_frame):
        batch_starts.append(start)
        batch_inputs.append(_prepare_sequence(frames, background, mask, background_mode))
        if len(batch_inputs) >= runtime.batch_size:
            flush()
    flush()

    x1, y1, x2, y2 = bounds
    crop_width, crop_height = x2 - x1, y2 - y1
    rows: list[dict[str, float | int | str]] = []
    for frame_index in range(start_frame, end_frame):
        selected = _robust_frame_point(candidates.get(frame_index, []))
        detected = selected is not None and selected[2] >= runtime.minimum_confidence
        if detected:
            target_x, target_y, confidence = selected
            center_x = x1 + target_x / TRACKNET_WIDTH * crop_width
            center_y = y1 + target_y / TRACKNET_HEIGHT * crop_height
            normalized_x, normalized_y = center_x / source_width, center_y / source_height
            detected = geometry.contains_shuttle_volume(normalized_x, normalized_y) and not geometry.excluded_normalized(
                normalized_x, normalized_y
            )
        if not detected:
            center_x = center_y = confidence = 0.0
        rows.append(
            {
                "time_seconds": frame_index / source_fps,
                "frame": frame_index,
                "source_width": source_width,
                "source_height": source_height,
                "confidence": confidence,
                "center_x": center_x,
                "center_y": center_y,
                "width": 8.0 if detected else 0.0,
                "height": 8.0 if detected else 0.0,
                "source": "tracknet",
                "detection_status": "detected" if detected else "missing",
                "evidence_weight": 0.90 if detected else 0.0,
            }
        )

    if inpaint_model_path is not None:
        inpaint_checkpoint = _load_checkpoint(torch, inpaint_model_path, device)
        inpaint_length = int(inpaint_checkpoint.get("param_dict", {}).get("seq_len", 16))
        inpaint_model = InpaintNet().to(device)
        inpaint_model.load_state_dict(inpaint_checkpoint["model"])
        inpaint_model.eval()
        _run_inpaint(
            torch,
            inpaint_model,
            rows,
            inpaint_length,
            device,
            min(runtime.maximum_inpaint_gap_frames, inpaint_length - 2),
        )
    if progress_callback is not None:
        progress_callback(0.96)

    visible_rows = [row for row in rows if row["detection_status"] != "missing"]
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
        for row in visible_rows:
            writer.writerow(row)
    if progress_callback is not None:
        progress_callback(1.0)
    return {
        "frames": end_frame - start_frame,
        "detected_points": sum(row["detection_status"] == "detected" for row in rows),
        "inpainted_points": sum(row["detection_status"] == "inpainted" for row in rows),
        "device": str(device),
        "background_mode": background_mode,
        "sequence_length": sequence_length,
        "output": str(output_csv),
    }
