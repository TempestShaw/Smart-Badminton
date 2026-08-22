import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_badminton.regression import _verify_truth_locks, train_guarded_candidate


def _write_source(root: Path, source_id: str) -> tuple[Path, Path, str]:
    times = np.arange(0.0, 5.0, 0.1)
    features = root / f"{source_id}-features.csv"
    truth = root / f"{source_id}-truth.csv"
    pd.DataFrame({"time_seconds": times, "frame_index": np.arange(len(times))}).to_csv(features, index=False)
    truth.write_text("rally,start_seconds,end_seconds\n1,1,3\n", encoding="utf-8")
    return features, truth, hashlib.sha256(truth.read_bytes()).hexdigest()


def test_truth_lock_rejects_changed_manual_answer(tmp_path: Path) -> None:
    first = _write_source(tmp_path, "a")
    second = _write_source(tmp_path, "b")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "sources": [
                    {"id": "a", "features": first[0].name, "rallies": first[1].name, "truth_sha256": first[2]},
                    {"id": "b", "features": second[0].name, "rallies": second[1].name, "truth_sha256": second[2]},
                ]
            }
        ),
        encoding="utf-8",
    )
    first[1].write_text("rally,start_seconds,end_seconds\n1,1,4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Truth lock mismatch"):
        _verify_truth_locks(dataset)


def test_guarded_training_cannot_overwrite_baseline(tmp_path: Path) -> None:
    path = tmp_path / "baseline.joblib"
    with pytest.raises(ValueError, match="must not overwrite"):
        train_guarded_candidate(tmp_path / "dataset.json", path, path, tmp_path / "gate.json")
