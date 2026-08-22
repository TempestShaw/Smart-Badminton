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
    assert report["editing_quality"]["premature_cut_seconds"] == 0.0
    assert report["editing_quality"]["missed_rallies"] == 0


def test_editing_loss_prioritizes_premature_cuts_over_extra_context(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    early = tmp_path / "early.csv"
    padded = tmp_path / "padded.csv"
    truth.write_text("rally,start_seconds,end_seconds\n1,2,6\n", encoding="utf-8")
    early.write_text("rally,start_seconds,end_seconds\n1,2,5\n", encoding="utf-8")
    padded.write_text("rally,start_seconds,end_seconds\n1,1,7\n", encoding="utf-8")

    early_loss = evaluate_rallies(early, truth)["editing_quality"]["editing_quality_loss"]
    padded_loss = evaluate_rallies(padded, truth)["editing_quality"]["editing_quality_loss"]

    assert early_loss > padded_loss
