# Model files

Model weights are optional and are not included by default.

To publish a checkpoint:

1. Add `<checkpoint>.license.json` with its source, license, SHA-256 and `redistributable: true`.
2. Confirm it with `smart-badminton doctor`.
3. Add a named `.gitignore` exception for that checkpoint.
4. Commit it through Git LFS.

The Python wheel never bundles model weights.
