# Contributing

1. Create a virtual environment and install `python -m pip install -e ".[dev]"`.
2. Keep videos, frames, human labels and venue-specific output outside Git.
3. Run `python scripts/repository_guard.py`, `pytest` and `ruff check .` before proposing a change.
4. Include a synthetic regression test for boundary changes.
5. For detection improvements, report per-rally coverage as well as aggregate precision; do not optimize away quiet endings.
6. For Studio UI changes, use Node 22 in `studio-web/`, run `npm run check` and `npm run build`, and include the generated `smart_badminton/studio_static/` update.
7. Build release artifacts with `python scripts/build_wheel.py --output-directory dist`. The verifier rejects stale React assets, private media, model weights and a missing license.
8. Publish a model only through Git LFS with a checksum-matched, redistributable license sidecar.

Bug reports should include the command, platform, Python version, camera geometry config and a minimal privacy-safe reproduction where possible.
