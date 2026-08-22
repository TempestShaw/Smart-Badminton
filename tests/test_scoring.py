from pathlib import Path

from smart_badminton.scoring import analyze_score, calculate_score_state, evaluate_score


def test_uncertain_events_never_change_score_but_manual_winner_does() -> None:
    events = [
        {
            "rally": "1",
            "event": "landing_in_candidate",
            "confidence": "0.95",
            "last_hitter": "near",
            "score_usable": "no",
        }
    ]

    unresolved = calculate_score_state(1, events)
    corrected = calculate_score_state(1, events, [{"rally": 1, "winner": "far", "server_override": "unknown"}])

    assert unresolved[0]["near_score"] == unresolved[0]["far_score"] == 0
    assert unresolved[0]["winner_source"] == "unresolved"
    assert corrected[0]["far_score"] == 1
    assert corrected[0]["server_next"] == "far"
    assert corrected[0]["winner_source"] == "manual"


def test_next_formal_serve_resolves_previous_rally_without_inventing_last_point() -> None:
    rows = calculate_score_state(
        3,
        serve_observations=[
            {"rally": 1, "server": "far", "confidence": 0.92},
            {"rally": 2, "server": "near", "confidence": 0.90},
            {"rally": 3, "server": "far", "confidence": 0.88},
        ],
    )

    assert rows[0]["winner"] == "near"
    assert rows[0]["winner_source"] == "automatic-next-serve"
    assert rows[1]["winner"] == "far"
    assert rows[1]["winner_source"] == "automatic-next-serve"
    assert rows[1]["near_score"] == rows[1]["far_score"] == 1
    assert rows[2]["winner"] == "unknown"
    assert rows[2]["winner_source"] == "unresolved"
    assert rows[2]["score_complete"] is False


def test_low_confidence_next_serve_never_changes_score() -> None:
    rows = calculate_score_state(
        2,
        serve_observations=[{"rally": 2, "server": "near", "confidence": 0.69}],
    )

    assert all(row["winner"] == "unknown" for row in rows)
    assert rows[-1]["near_score"] == rows[-1]["far_score"] == 0


def test_near_court_landing_awards_far_player_without_last_hitter() -> None:
    rows = calculate_score_state(
        1,
        events=[
            {
                "rally": "1",
                "event": "landing_in_candidate",
                "confidence": "0.86",
                "last_hitter": "unknown",
                "landing_side": "near",
                "score_usable": "yes",
            }
        ],
    )

    assert rows[0]["winner"] == "far"
    assert rows[0]["far_score"] == 1
    assert rows[0]["winner_source"] == "automatic-terminal"


def test_score_analysis_and_evaluation_are_separate_from_rally_timeline(tmp_path: Path) -> None:
    rallies = tmp_path / "rallies.csv"
    rallies.write_text("rally,start_seconds,end_seconds\n1,1,2\n2,3,4\n", encoding="utf-8")
    original = rallies.read_text(encoding="utf-8")
    corrections = tmp_path / "corrections.csv"
    corrections.write_text(
        "rally,winner,server_override,note\n1,near,unknown,human\n2,far,unknown,human\n",
        encoding="utf-8",
    )
    output = tmp_path / "score.csv"

    summary = analyze_score(rallies, output, corrections_csv=corrections)
    evaluation = evaluate_score(output, corrections)

    assert summary["resolved"] == 2
    assert evaluation["coverage"] == 1.0
    assert evaluation["accuracy_on_covered"] == 1.0
    assert evaluation["editing_metrics_included"] is False
    assert rallies.read_text(encoding="utf-8") == original
