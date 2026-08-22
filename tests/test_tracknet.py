from __future__ import annotations

import numpy as np

from smart_badminton.tracknet import _heatmap_point, _inpaint_mask, _robust_frame_point
from smart_badminton.tracknet_training import _target_heatmaps, _training_starts


def test_heatmap_point_uses_weighted_component_center() -> None:
    heatmap = np.zeros((288, 512), dtype=np.float32)
    heatmap[100:103, 200:204] = 0.8
    heatmap[101, 202] = 0.95

    point = _heatmap_point(heatmap, 0.5)

    assert point is not None
    assert 201.0 < point[0] < 203.0
    assert 100.0 < point[1] < 102.5
    assert abs(point[2] - 0.95) < 1e-6


def test_robust_frame_point_rejects_distant_window_vote() -> None:
    point = _robust_frame_point(
        [
            (100.0, 80.0, 0.9, 1.0),
            (102.0, 81.0, 0.8, 0.9),
            (420.0, 230.0, 0.99, 0.5),
        ]
    )

    assert point is not None
    assert point[0] < 105
    assert point[1] < 85


def test_inpaint_mask_only_repairs_short_surrounded_gap() -> None:
    rows = [
        {"detection_status": status, "center_y": y}
        for status, y in [
            ("detected", 10),
            ("missing", 0),
            ("missing", 0),
            ("detected", 20),
            ("missing", 0),
            ("missing", 0),
            ("missing", 0),
        ]
    ]

    mask = _inpaint_mask(rows, maximum_gap=2)

    assert mask.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_training_windows_and_targets_follow_point_labels() -> None:
    labels = {frame: ((50.0, 25.0) if frame == 4 else None) for frame in range(12)}

    starts = _training_starts(labels, sequence_length=4)
    targets = _target_heatmaps(labels, start=2, sequence_length=4, bounds=(0, 0, 100, 50))

    assert {1, 2, 3, 4}.issubset(starts)
    assert targets.shape == (4, 288, 512)
    assert targets[2].max() == 1.0
    assert sum(frame.max() > 0 for frame in targets) == 1
