from pathlib import PurePosixPath

from scripts.repository_guard import SECRET_PATTERNS, is_private_path


def test_repository_guard_rejects_match_runtime_data() -> None:
    assert is_private_path(PurePosixPath("matches/Match5/Metadata/rallies-studio-review.csv"))
    assert is_private_path(PurePosixPath("matches/Match5/Analysis/Shuttle_Annotations/points.csv"))
    assert is_private_path(PurePosixPath("matches/Match5/source.mp4"))


def test_repository_guard_allows_the_synthetic_fixture() -> None:
    assert not is_private_path(PurePosixPath("examples/privacy_safe_sample/Analysis/shuttle-track.csv"))
    assert not is_private_path(PurePosixPath("examples/privacy_safe_sample/Calibration/court-config.json"))


def test_repository_guard_detects_supported_secret_shapes() -> None:
    example = b"token=sk-or-v1-" + b"a" * 32
    assert any(pattern.search(example) for pattern in SECRET_PATTERNS)
