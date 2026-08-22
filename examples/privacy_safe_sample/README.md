# Privacy-safe sample project

This metadata-only fixture contains no recorded person, venue, audio or model output. All coordinates and times are
synthetic. It demonstrates the project layout and the hand-off between calibration, trajectory evidence, user
corrections and a reviewed timeline.

```text
privacy_safe_sample/
  Calibration/court-config.json
  Analysis/shuttle-track.csv
  Analysis/Shuttle_Annotations/user-shuttle-points.csv
  Standard_Answers/sample_rallies_ground_truth.csv
```

The trajectory includes a primary flight, one endpoint-validated recovered point, and a simultaneous competing-court
candidate. The user annotation file shows one added missed point and one rejected candidate. This is intentionally not
a playable Studio project because no video is included; point Studio at your own authorized recording and use its
Court Calibration and Shuttle Annotation controls to create equivalent local files.
