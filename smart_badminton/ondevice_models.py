"""Models for on-device (phone/browser) analysis, derived from the verified YOLO11n-pose checkpoint.

Two ONNX files are exported: a float model for WebGPU and an int8 one for the WebAssembly fallback. Statically
quantised convolutions are ~15% faster on WASM SIMD and a quarter of the download (3.4 MB instead of 12 MB), with
wrist keypoints that stay within the float model's error on player crops. Calibration uses person crops from the
public COCO8-pose sample, so a release build needs no match footage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from .model_assets import DOWNLOADABLE_MODELS, MODEL_FILENAMES, _download, _sha256

ONDEVICE_FLOAT = "yolo11n-pose.onnx"
ONDEVICE_INT8 = "yolo11n-pose-int8.onnx"
ONDEVICE_MODELS = (ONDEVICE_FLOAT, ONDEVICE_INT8)
CROP_SIZE = 192  # the region-of-interest input size used by the on-device engine
CALIBRATION_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip"
CALIBRATION_SHA256 = "468e06839e06eaf317594386967d238af4bdfde670e33d9a2a287412b6a37de5"
EXPORT_RECIPE = "ultralytics YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=320, dynamic=True, simplify=True, opset=17)"
QUANTIZE_RECIPE = (
    "onnxruntime.quantization.quantize_static(QDQ, per-channel int8 weights, uint8 activations, Conv only, MinMax "
    "calibration on 192 px letterboxed person crops from COCO8-pose)"
)


def _letterbox(image, size: int):
    import cv2
    import numpy as np

    scale = size / max(image.shape[:2])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def calibration_crops(root: Path, size: int = CROP_SIZE) -> list:
    """Square person crops (as the engine feeds the model), at three context scales and mirrored."""
    import cv2
    import numpy as np

    crops = []
    for image_path in sorted(root.glob("images/*/*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        labels = image_path.parent.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        if not labels.is_file():
            continue
        for line in labels.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            cx, cy, box_width, box_height = (float(value) for value in values[1:5])
            for context in (1.3, 1.6, 2.0):
                side = max(box_width * width, box_height * height) * context
                x0, y0 = int(max(0, cx * width - side / 2)), int(max(0, cy * height - side / 2))
                crop = image[y0 : int(y0 + side), x0 : int(x0 + side)]
                if min(crop.shape[:2]) < 16:
                    continue
                for mirrored in (crop, crop[:, ::-1]):
                    rgb = _letterbox(np.ascontiguousarray(mirrored), size)[:, :, ::-1]
                    crops.append(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255)
    return crops


def _sidecar(model: Path, name: str, derived_from: str, recipe: str) -> None:
    manifest = {
        "name": name,
        "source_url": "https://github.com/ultralytics/assets/releases/tag/v8.3.0",
        "license": "AGPL-3.0-only",
        "redistributable": True,
        "sha256": _sha256(model),
        "derived_from": derived_from,
        "export": recipe,
    }
    model.with_suffix(model.suffix + ".license.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def export_ondevice_models(directory: Path, calibration: Path | None = None) -> dict[str, object]:
    """Export the float and int8 on-device models next to the verified ``yolo11n-pose.pt``."""
    try:
        import onnx  # noqa: F401
        from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - depends on the optional toolchain
        raise RuntimeError("Exporting on-device models needs: pip install 'smart-badminton[ondevice]'") from error

    directory = directory.expanduser().resolve()
    checkpoint = directory / MODEL_FILENAMES["pose"]
    expected = next(asset.sha256 for asset in DOWNLOADABLE_MODELS if asset.role == "pose")
    if not checkpoint.is_file():
        raise RuntimeError(f"{checkpoint} is missing; run 'smart-badminton install-models' first")
    if _sha256(checkpoint) != expected:
        raise RuntimeError(f"{checkpoint.name} does not match the verified checkpoint")

    with tempfile.TemporaryDirectory(prefix=".smart-badminton-ondevice-", dir=directory) as temporary:
        work = Path(temporary)
        staged = work / MODEL_FILENAMES["pose"]
        shutil.copy2(checkpoint, staged)
        exported = Path(YOLO(str(staged)).export(format="onnx", imgsz=320, dynamic=True, simplify=True, opset=17))
        float_model = directory / ONDEVICE_FLOAT
        shutil.move(str(exported), float_model)
        _sidecar(
            float_model,
            "Ultralytics YOLO11n Pose (ONNX export, dynamic batch and size, opset 17)",
            f"{MODEL_FILENAMES['pose']} sha256 {expected}",
            EXPORT_RECIPE,
        )

        if calibration is None:
            archive = work / "coco8-pose.zip"
            _download(CALIBRATION_URL, archive)
            if _sha256(archive) != CALIBRATION_SHA256:
                raise RuntimeError("Checksum mismatch for the COCO8-pose calibration sample")
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(work / "calibration")
            calibration = work / "calibration" / "coco8-pose"
        crops = calibration_crops(calibration)
        if len(crops) < 16:
            raise RuntimeError(f"Only {len(crops)} calibration crops found in {calibration}")

        class Reader(CalibrationDataReader):
            def __init__(self) -> None:
                self.items = iter(crops)

            def get_next(self):
                item = next(self.items, None)
                return None if item is None else {"images": item}

        prepared = work / "prepared.onnx"
        quant_pre_process(str(float_model), str(prepared), skip_symbolic_shape=True)
        int8_model = directory / ONDEVICE_INT8
        quantize_static(
            str(prepared),
            str(int8_model),
            Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv"],
            calibrate_method=CalibrationMethod.MinMax,
        )
        _sidecar(
            int8_model,
            "Ultralytics YOLO11n Pose (ONNX, int8 QDQ for WebAssembly)",
            f"{ONDEVICE_FLOAT} sha256 {_sha256(float_model)}",
            QUANTIZE_RECIPE,
        )
    return {
        "directory": str(directory),
        "models": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ONDEVICE_MODELS},
        "calibration_crops": len(crops),
    }
