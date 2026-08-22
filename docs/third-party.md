# Third-party software and models

This repository does not vendor upstream source trees or model weights.

## Ultralytics

Player pose and the optional shuttle detector dynamically import the `ultralytics` package only when those features are requested. [Ultralytics documents AGPL-3.0 and Enterprise options](https://github.com/ultralytics/ultralytics/blob/main/docs/en/index.md). Users are responsible for choosing compatible terms. Do not copy an Ultralytics checkpoint into a release without reviewing its terms and adding the checksum-bound `<checkpoint>.license.json` sidecar checked by `smart-badminton doctor`.

## Good-Badminton

The exploratory [Good-Badminton repository](https://github.com/yo-WASSUP/Good-Badminton) carries Apache-2.0 source licensing. Its README points to separately released OpenMMLab-ecosystem weights, so the repository license must not be assumed to prove every checkpoint's redistribution terms. This project does not copy its source or weights.

## Checkpoint and dataset gate

`smart-badminton doctor` computes every configured checkpoint SHA-256 and reports a model as release-safe only when a matching sidecar declares name, source URL, license and redistribution status. `audit-shuttle-dataset` separately requires those rights fields for every camera source. These checks prevent accidental packaging; they are engineering safeguards, not legal advice.

## TrackNetV3

[TrackNetV3](https://github.com/qaz812345/TrackNetV3) is MIT licensed. This project contains an adapted, checkpoint-compatible runtime and records that attribution in source and documentation. Upstream checkpoints remain local downloads under ignored model or `ThirdParty` directories; verify checkpoint redistribution rights before publishing them.
