from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from smart_badminton.doctor import inspect_model_license
from smart_badminton.model_assets import DOWNLOADABLE_MODELS, MODEL_FILENAMES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _version() -> str:
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not read project version from pyproject.toml")
    return match.group(1)


def _bundle_files() -> list[tuple[Path, Path]]:
    model_names = [MODEL_FILENAMES["rally"], *(asset.filename for asset in DOWNLOADABLE_MODELS)]
    files: list[tuple[Path, Path]] = []
    for name in model_names:
        model = MODELS / name
        license_manifest = model.with_suffix(model.suffix + ".license.json")
        files.extend(((model, Path("models") / model.name), (license_manifest, Path("models") / license_manifest.name)))
    for name in ("rally-state-final-v4-frozen.adapter.json", "MODEL_CARD.md", "README.md"):
        files.append((MODELS / name, Path("models") / name))
    files.append((ROOT / "THIRD_PARTY_NOTICES.md", Path("THIRD_PARTY_NOTICES.md")))
    for license_file in sorted((ROOT / "licenses").iterdir()):
        if license_file.is_file():
            files.append((license_file, Path("licenses") / license_file.name))
    return files


def build_model_bundle(output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    version = _version()
    root_name = f"smart-badminton-models-{version}"
    files = _bundle_files()
    missing = [str(source) for source, _target in files if not source.is_file()]
    if missing:
        raise RuntimeError(f"Model bundle inputs are missing: {missing}")
    for source, target in files:
        if target.parts[:1] == ("models",) and source.suffix in {".pt", ".joblib"}:
            inspection = inspect_model_license(source)
            if inspection.get("redistributable") is not True:
                raise RuntimeError(f"Model is not release-safe: {source.name}: {inspection.get('status')}")

    manifest = {
        "version": version,
        "models": [
            {
                "file": target.as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            for source, target in files
            if source.suffix in {".pt", ".joblib"}
        ],
    }
    destination = output_directory.resolve() / f"{root_name}.zip"
    with zipfile.ZipFile(destination, "w") as archive:
        for source, target in files:
            compression = zipfile.ZIP_STORED if source.suffix in {".pt", ".joblib"} else zipfile.ZIP_DEFLATED
            archive.write(source, (Path(root_name) / target).as_posix(), compress_type=compression)
        archive.writestr(
            (Path(root_name) / "models" / "manifest.json").as_posix(),
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            compress_type=zipfile.ZIP_DEFLATED,
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the complete open-source vision model bundle")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    bundle = build_model_bundle(args.output_directory.expanduser().resolve())
    print(f"Verified model bundle: {bundle}")


if __name__ == "__main__":
    main()
