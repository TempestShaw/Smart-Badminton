from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from smart_badminton.score_labeling import (
    consensus_label,
    label_score_evidence,
    load_machine_labels,
    prepare_score_evidence,
    save_machine_label_review,
    validate_machine_label,
)


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    assert writer.isOpened()
    for index in range(30):
        frame = np.full((180, 320, 3), index * 5, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_prepare_score_evidence_extracts_chronological_terminal_frames(tmp_path: Path) -> None:
    video = tmp_path / "match.mp4"
    _video(video)
    rallies = tmp_path / "rallies.csv"
    rallies.write_text("rally,start_seconds,end_seconds\n1,0.2,2.2\n", encoding="utf-8")

    manifest = prepare_score_evidence(video, rallies, tmp_path / "labels", [1])

    frames = manifest["rallies"][0]["frames"]
    assert len(frames) == 8
    assert [row["time"] for row in frames] == sorted(row["time"] for row in frames)
    assert all(Path(row["path"]).exists() for row in frames)
    assert (tmp_path / "labels" / "manifest.json").exists()


def test_consensus_requires_agreement_and_confidence() -> None:
    label = validate_machine_label(
        {
            "terminal_event": "net",
            "last_hitter": "near",
            "landing_side": "near",
            "winner": "far",
            "post_rally_event": "none",
            "confidence": 0.91,
            "evidence_frames": [6, 7],
        }
    )
    status, suggestion = consensus_label([label, {**label, "confidence": 0.87}])
    assert status == "consensus"
    assert suggestion["winner"] == "far"

    status, suggestion = consensus_label([label, {**label, "winner": "near"}])
    assert status == "review"
    assert suggestion["winner"] == "unknown"


def test_machine_labels_are_saved_separately_and_reviewed(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame), np.zeros((24, 24, 3), dtype=np.uint8))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rallies": [{"rally": 3, "frames": [{"time": 4.0, "path": str(frame)}]}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "machine-score-labels.json"

    def requester(_endpoint, _key, _payload, _timeout):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "terminal_event": "landing_in",
                                "last_hitter": "far",
                                "landing_side": "near",
                                "winner": "far",
                                "post_rally_event": "none",
                                "confidence": 0.93,
                                "evidence_frames": [1],
                                "reason": "lands near",
                            }
                        )
                    }
                }
            ]
        }

    result = label_score_evidence(manifest, output, "vision-model", "https://example.test", "secret", requester=requester)
    assert result["labels"][0]["status"] == "consensus"
    assert load_machine_labels(output)[0]["suggestion"]["winner"] == "far"

    reviewed = save_machine_label_review(output, 3, "accepted")
    assert reviewed[0]["status"] == "accepted"
