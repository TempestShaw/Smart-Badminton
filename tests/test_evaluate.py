from pathlib import Path

from smart_badminton.evaluate import evaluate_rallies


def test_evaluation_detects_complete_coverage(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    predicted = tmp_path / "predicted.csv"
    truth.write_text("rally,start_seconds,end_seconds\n1,2,4\n", encoding="utf-8")
    predicted.write_text("rally,start_seconds,end_seconds\n1,1.5,4.5\n", encoding="utf-8")
    report = evaluate_rallies(predicted, truth)
    assert report["all_truth_rallies_preserved"] is True
    assert report["active_time_recall"] == 1.0
