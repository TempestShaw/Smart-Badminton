import csv
import hashlib
import json
from pathlib import Path

from smart_badminton.score_learning import fit_score_evidence_model, load_score_evidence_model
from smart_badminton.scoring import calculate_score_state, load_score_corrections, validate_score_corrections


def _project(library: Path, name: str, winners: list[str], landing_side: str = "near") -> Path:
    root = library / name
    metadata = root / "Metadata"
    analysis = root / "Analysis"
    metadata.mkdir(parents=True)
    analysis.mkdir(parents=True)
    corrections = metadata / "score-corrections.csv"
    corrections.write_text(
        "rally,winner,server_override,note\n"
        + "".join(f"{index},{winner},unknown,human\n" for index, winner in enumerate(winners, 1)),
        encoding="utf-8",
    )
    with (analysis / "rally-events.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["rally", "event", "confidence", "last_hitter", "landing_side", "score_usable"],
        )
        writer.writeheader()
        for index in range(1, len(winners) + 1):
            writer.writerow(
                {
                    "rally": index,
                    "event": "landing_in_candidate",
                    "confidence": 0.82,
                    "last_hitter": "unknown",
                    "landing_side": landing_side,
                    "score_usable": "yes",
                }
            )
    return corrections


def test_manual_results_disable_a_repeatedly_wrong_terminal_rule(tmp_path: Path) -> None:
    corrections = _project(tmp_path, "Match1", ["near", "near", "far"])
    original_hash = hashlib.sha256(corrections.read_bytes()).hexdigest()
    model_path = tmp_path / ".smart-badminton" / "models" / "score-evidence-v1.json"

    model = fit_score_evidence_model(tmp_path, model_path)
    rows = calculate_score_state(
        1,
        events=[
            {
                "rally": "1",
                "event": "landing_in_candidate",
                "confidence": "0.82",
                "last_hitter": "unknown",
                "landing_side": "near",
                "score_usable": "yes",
            }
        ],
        evidence_model=model,
        evidence_project="Match1",
    )

    assert model["rules"]["landing_in:near"]["status"] == "candidate"
    assert model["project_rules"]["Match1"]["landing_in:near"]["status"] == "disabled"
    assert model["manual_examples"] == 3
    assert model["comparable_examples"] == 3
    assert model_path.with_name("score-evidence-v1-training.csv").exists()
    assert rows[0]["winner"] == "unknown"
    assert hashlib.sha256(corrections.read_bytes()).hexdigest() == original_hash


def test_trusted_manual_evidence_can_expand_a_consistent_rule(tmp_path: Path) -> None:
    _project(tmp_path, "Match1", ["far", "far", "far", "far"])
    model_path = tmp_path / "score-model.json"
    fit_score_evidence_model(tmp_path, model_path)
    model = load_score_evidence_model(model_path)

    rows = calculate_score_state(
        1,
        events=[
            {
                "rally": "1",
                "event": "landing_in_candidate",
                "confidence": "0.60",
                "last_hitter": "unknown",
                "landing_side": "near",
                "score_usable": "no",
            }
        ],
        evidence_model=model,
        evidence_project="Match1",
    )

    assert model is not None
    assert model["rules"]["landing_in:near"]["status"] == "candidate"
    assert model["project_rules"]["Match1"]["landing_in:near"]["status"] == "trusted"
    assert rows[0]["winner"] == "far"
    assert rows[0]["winner_source"] == "automatic-terminal"
    assert rows[0]["note"].startswith("learned evidence:")

    new_project_rows = calculate_score_state(
        1,
        events=[
            {
                "rally": "1",
                "event": "landing_in_candidate",
                "confidence": "0.60",
                "last_hitter": "unknown",
                "landing_side": "near",
                "score_usable": "no",
            }
        ],
        evidence_model=model,
        evidence_project="Match2",
    )
    assert new_project_rows[0]["winner"] == "unknown"


def test_structured_score_corrections_are_backward_compatible(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    old.write_text("rally,winner,server_override,note\n1,near,unknown,human\n", encoding="utf-8")

    loaded = load_score_corrections(old)
    validated = validate_score_corrections(
        [
            {
                **loaded[0],
                "server": "near",
                "last_hitter": "near",
                "terminal_event": "landing_in",
                "landing_side": "far",
                "post_rally_event": "handoff",
            }
        ],
        1,
    )

    assert loaded[0]["terminal_event"] == "unknown"
    assert validated[0]["server"] == "near"
    assert validated[0]["post_rally_event"] == "handoff"


def test_global_rule_requires_consistent_examples_from_two_projects(tmp_path: Path) -> None:
    _project(tmp_path, "Match1", ["far", "far"])
    _project(tmp_path, "Match2", ["far", "far"])

    model = fit_score_evidence_model(tmp_path, tmp_path / "score-model.json")

    assert model["rules"]["landing_in:near"]["status"] == "trusted"
    assert model["rules"]["landing_in:near"]["projects"] == ["Match1", "Match2"]


def test_machine_consensus_is_low_weight_and_project_local(tmp_path: Path) -> None:
    _project(tmp_path, "Match1", ["far", "far"])
    corrections = tmp_path / "Match1" / "Metadata" / "score-corrections.csv"
    corrections.write_text("rally,winner,server_override,note\n1,far,unknown,human\n", encoding="utf-8")
    machine = tmp_path / "Match1" / "Analysis" / "Score_Labeling" / "machine-score-labels.json"
    machine.parent.mkdir(parents=True)
    machine.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "rally": 2,
                        "status": "consensus",
                        "suggestion": {
                            "winner": "far",
                            "last_hitter": "far",
                            "terminal_event": "landing_in",
                            "landing_side": "near",
                            "post_rally_event": "none",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    model = fit_score_evidence_model(tmp_path, tmp_path / "score-model.json")

    assert model["manual_examples"] == 1
    assert model["pseudo_examples"] == 1
    assert model["rules"]["landing_in:near"]["samples"] == 1
    assert model["project_rules"]["Match1"]["landing_in:near"]["effective_samples"] == 1.25
