from pathlib import Path

import pytest

from smart_badminton.shuttle_annotations import (
    load_shuttle_annotations,
    save_shuttle_annotations,
    validate_shuttle_annotations,
)


def test_shuttle_annotations_validate_round_trip_and_backup(tmp_path: Path) -> None:
    path = tmp_path / "user-shuttle-points.csv"
    payload = [
        {
            "id": "user-1",
            "time_seconds": 1.25,
            "x_normalized": 0.45,
            "y_normalized": 0.22,
            "action": "add",
            "note": "studio",
        }
    ]
    rows = validate_shuttle_annotations(payload, duration=4.0, fps=60.0, width=1920, height=1080)

    assert rows[0]["frame"] == 75
    assert save_shuttle_annotations(path, rows) is None
    assert load_shuttle_annotations(path)[0]["x_normalized"] == 0.45

    replacement = validate_shuttle_annotations(
        [{**payload[0], "action": "reject", "time_seconds": 2.0}],
        duration=4.0,
        fps=60.0,
        width=1920,
        height=1080,
    )
    backup = save_shuttle_annotations(path, replacement)

    assert backup == path.with_suffix(".csv.bak")
    assert load_shuttle_annotations(backup)[0]["action"] == "add"
    assert load_shuttle_annotations(path)[0]["action"] == "reject"


def test_shuttle_annotations_reject_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="normalized"):
        validate_shuttle_annotations(
            [{"time_seconds": 1.0, "x_normalized": 1.2, "y_normalized": 0.5, "action": "add"}],
            duration=2.0,
            fps=30.0,
            width=1280,
            height=720,
        )
