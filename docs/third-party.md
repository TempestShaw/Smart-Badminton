# Third-party software and models

The source repository does not vendor upstream source trees. The GitHub Release model bundle contains verified model
weights with checksum-bound manifests and license copies.

## Ultralytics

Player pose and shuttle detection use the AGPL-3.0 Ultralytics runtime. The complete model bundle includes
`yolo11n-pose.pt` and the Good-Badminton shuttle checkpoint with AGPL-3.0 notices. Smart Badminton publishes its full
source and does not provide an Enterprise-licensed build.

## Good-Badminton

The [Good-Badminton release](https://github.com/yo-WASSUP/Good-Badminton/releases/tag/v0.1.0) declares
`yolo11s-ball.pt` Apache-2.0. Its embedded Ultralytics metadata declares AGPL-3.0, so this project applies the more
restrictive AGPL-3.0 terms to the bundled copy.

## Checkpoint and dataset gate

`smart-badminton doctor` computes every configured checkpoint SHA-256 and reports a model as release-safe only when a matching sidecar declares name, source URL, license and redistribution status. `audit-shuttle-dataset` separately requires those rights fields for every camera source. These checks prevent accidental packaging; they are engineering safeguards, not legal advice.

## TrackNetV3

[TrackNetV3](https://github.com/qaz812345/TrackNetV3) is MIT licensed. This project contains an adapted,
checkpoint-compatible runtime and bundles the two checkpoints linked by its official README. The original copyright
and MIT license are retained.
