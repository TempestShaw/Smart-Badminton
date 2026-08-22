from __future__ import annotations

import csv
import json
from pathlib import Path

from smart_badminton.hybrid import DETECTION_FIELDS, fuse_shuttle_detections


def _write_detection(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=DETECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(time: float, frame: int, x: float, y: float, source: str, status: str = "detected") -> dict[str, object]:
    return {
        "time_seconds": time,
        "frame": frame,
        "source_width": 1000,
        "source_height": 500,
        "confidence": 0.8,
        "center_x": x,
        "center_y": y,
        "width": 8,
        "height": 8,
        "source": source,
        "detection_status": status,
        "evidence_weight": 0.9,
    }


def test_hybrid_fuses_agreement_and_keeps_inpaint_low_weight(tmp_path: Path) -> None:
    config = tmp_path / "court.json"
    config.write_text(
        json.dumps(
            {
                "reference_frame": {"width": 1000, "height": 500},
                "masks_normalized": {
                    "court_ground_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "near_player_zone": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "far_player_zone": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "net_band": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "shuttle_airspace_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "static_false_positive_polygons": [],
                    "background_court_polygons": [],
                },
            }
        ),
        encoding="utf-8",
    )
    yolo = tmp_path / "yolo.csv"
    tracknet = tmp_path / "tracknet.csv"
    output = tmp_path / "hybrid.csv"
    _write_detection(yolo, [_row(1.0, 30, 500, 250, "yolo")])
    _write_detection(
        tracknet,
        [
            _row(1.01, 30, 510, 252, "tracknet"),
            _row(1.04, 31, 530, 260, "tracknet", "inpainted"),
        ],
    )

    result = fuse_shuttle_detections(yolo, tracknet, config, output)
    with output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    assert result["fused"] == 1
    assert rows[0]["source"] == "hybrid"
    assert rows[0]["detection_status"] == "fused"
    assert rows[1]["detection_status"] == "inpainted"
    assert float(rows[1]["evidence_weight"]) <= 0.15

