# Rally State v4 Frozen

`rally-state-final-v4-frozen.joblib` estimates rally activity from fixed-camera badminton features. The segmenter
combines that probability with motion, pose, audio and optional shuttle evidence to produce editable clip boundaries.

## Details

- Model family: scikit-learn histogram gradient boosting
- Input rate: 10 feature samples per second
- Training data: five human-reviewed fixed-camera recordings
- License: Apache-2.0
- SHA-256: `baa9f5c6bf1c82f5f46b37a6ebf71de14cec3b013f07ff792556f01f5ca295d5`

The private source videos and annotations are not included. The artifact metadata contains no local file paths or
reviewed timestamps.

## Hardware

The checkpoint runs on CPU and has no CUDA dependency. CUDA only accelerates optional TrackNet, YOLO and pose feature
extraction; those components automatically fall back to CPU.

## Usage

```bash
smart-badminton predict \
  --features output/features.csv \
  --model models/rally-state-final-v4-frozen.joblib \
  --output output/probabilities.csv
```

For Studio, pass the same path with `--model`. Keep the adjacent adapter file to use the release boundary profile.

## Scope

The model was trained for fixed, full-court singles footage. New camera angles, doubles, heavy occlusion and poor court
visibility may require manual timeline adjustment or camera-specific training.
