import csv
from pathlib import Path

from smart_badminton.geometry import CourtGeometry
from smart_badminton.shuttle_annotations import load_shuttle_annotations


def test_privacy_safe_sample_has_valid_geometry_trajectory_and_labels() -> None:
    root = Path(__file__).parents[1] / "examples" / "privacy_safe_sample"
    geometry = CourtGeometry.from_json(root / "Calibration" / "court-config.json")
    with (root / "Analysis" / "shuttle-track.csv").open(newline="", encoding="utf-8") as source:
        trajectory = list(csv.DictReader(source))
    with (root / "Standard_Answers" / "sample_rallies_ground_truth.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        rallies = list(csv.DictReader(source))
    annotations = load_shuttle_annotations(
        root / "Analysis" / "Shuttle_Annotations" / "user-shuttle-points.csv"
    )

    assert geometry.contains_active_court(0.5, 0.75)
    assert {row["status"] for row in trajectory} >= {"tracked", "recovered", "competing", "manual"}
    assert {row["action"] for row in annotations} == {"add", "reject"}
    assert len(rallies) == 2
