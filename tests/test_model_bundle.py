from scripts.build_model_bundle import _bundle_files


def test_model_bundle_has_every_runtime_role_and_license() -> None:
    targets = {target.as_posix() for _source, target in _bundle_files()}

    for name in (
        "rally-state-final-v4-frozen.joblib",
        "yolo11n-pose.pt",
        "yolo11s-ball.pt",
        "TrackNet_best.pt",
        "InpaintNet_best.pt",
    ):
        assert f"models/{name}" in targets
        assert f"models/{name}.license.json" in targets
    assert "THIRD_PARTY_NOTICES.md" in targets
    assert "licenses/AGPL-3.0.txt" in targets
    assert "licenses/TrackNetV3-MIT.txt" in targets


def test_bundle_paths_are_relative() -> None:
    assert all(not target.is_absolute() for _source, target in _bundle_files())
