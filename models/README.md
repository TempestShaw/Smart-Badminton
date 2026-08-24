# Models

Each GitHub Release contains a separate `smart-badminton-models-<version>.zip`. Extract it beside the repository or
installed launcher; Studio discovers the `models/` directory automatically. The same verified files can be installed
with:

```bash
smart-badminton install-models --directory models
```

| File | Purpose | License |
| --- | --- | --- |
| `rally-state-final-v4-frozen.joblib` | Rally activity | Apache-2.0 |
| `yolo11n-pose.pt` | Player pose | AGPL-3.0-only |
| `yolo11s-ball.pt` | Shuttle detection | AGPL-3.0-only |
| `TrackNet_best.pt` | Shuttle sequence tracking | MIT |
| `InpaintNet_best.pt` | Short trajectory repair | MIT |

Every checkpoint has a checksum-bound `.license.json`. See [MODEL_CARD.md](MODEL_CARD.md) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The Python wheel stays small; the complete model ZIP is the
offline runtime bundle.
