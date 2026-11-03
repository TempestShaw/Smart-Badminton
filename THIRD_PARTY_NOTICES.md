# Third-party notices

The Smart Badminton source is Apache-2.0. The optional complete vision bundle contains components under their own
licenses:

- Ultralytics and `yolo11n-pose.pt`: AGPL-3.0-only. Copyright Ultralytics. Source and license:
  <https://github.com/ultralytics/ultralytics>.
- `yolo11s-ball.pt`: released by Good-Badminton. The release declares Apache-2.0, while the embedded Ultralytics
  checkpoint metadata declares AGPL-3.0. Smart Badminton distributes it under the more restrictive AGPL-3.0 terms.
  Source: <https://github.com/yo-WASSUP/Good-Badminton/releases/tag/v0.1.0>.
- TrackNetV3 runtime, `TrackNet_best.pt`, and `InpaintNet_best.pt`: MIT. Copyright (c) 2024 qaz812345. Source:
  <https://github.com/qaz812345/TrackNetV3>.
- `@ffmpeg/ffmpeg` and `@ffmpeg/util`: MIT. Source: <https://github.com/ffmpegwasm/ffmpeg.wasm>.
- `yolo11n-pose.onnx` and `yolo11n-pose-int8.onnx`: AGPL-3.0-only, exported from `yolo11n-pose.pt` (the int8 model is
  quantised with person crops from Ultralytics' COCO8-pose sample, used only for calibration and not distributed).
  They are published with releases and with the web build's on-device mode.
- `onnxruntime-web`: MIT. Copyright Microsoft Corporation. Source: <https://github.com/microsoft/onnxruntime>. It is
  downloaded as an npm dependency and published only with the web build.
- `@ffmpeg/core` WebAssembly runtime: GPL-2.0-or-later. Source and corresponding build scripts:
  <https://github.com/ffmpegwasm/ffmpeg.wasm>. It is downloaded as an npm dependency and published only with the
  Browser Quick web build.

Copies of the applicable AGPL-3.0 and MIT license texts are in `licenses/`. Checksum-bound manifests next to every
model identify the exact artifact and source. Browser runtime package versions are pinned in `studio-web/package-lock.json`.
