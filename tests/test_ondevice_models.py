from pathlib import Path

import pytest

from smart_badminton.ondevice_models import calibration_crops, export_ondevice_models


def test_calibration_crops_follow_the_engine_input(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    image = np.zeros((240, 320, 3), np.uint8)
    image[60:200, 140:180] = (40, 40, 200)
    cv2.imwrite(str(tmp_path / "images" / "train" / "0001.jpg"), image)
    (tmp_path / "labels" / "train" / "0001.txt").write_text("0 0.5 0.54 0.125 0.58\n", encoding="utf-8")

    crops = calibration_crops(tmp_path)

    assert len(crops) == 6  # three context scales, each mirrored
    assert all(crop.shape == (1, 3, 192, 192) and 0 <= crop.min() and crop.max() <= 1 for crop in crops)


def test_export_requires_the_verified_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        export_ondevice_models(tmp_path)
