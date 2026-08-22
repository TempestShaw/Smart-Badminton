from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "smart_badminton"
STATIC_ROOT = PACKAGE_ROOT / "studio_static"
BUILD_ROOT = REPOSITORY_ROOT / "build"


def _clean_generated_build() -> None:
    resolved = BUILD_ROOT.resolve()
    if resolved.parent != REPOSITORY_ROOT.resolve() or resolved.name != "build":
        raise RuntimeError(f"Refusing to clean unexpected build directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _expected_static_files() -> set[str]:
    return {
        (Path("smart_badminton") / "studio_static" / path.relative_to(STATIC_ROOT)).as_posix()
        for path in STATIC_ROOT.rglob("*")
        if path.is_file()
    }


def verify_wheel(wheel: Path) -> None:
    expected_static = _expected_static_files()
    with zipfile.ZipFile(wheel) as archive:
        names = {name.rstrip("/") for name in archive.namelist() if not name.endswith("/")}
        packaged_static = {name for name in names if name.startswith("smart_badminton/studio_static/")}
        if packaged_static != expected_static:
            stale = sorted(packaged_static - expected_static)
            missing = sorted(expected_static - packaged_static)
            raise RuntimeError(
                "Wheel Studio assets do not match the published source tree. "
                f"Stale={stale[:8]} Missing={missing[:8]}"
            )
        if "smart_badminton/studio_static/index.html" not in names:
            raise RuntimeError("Wheel is missing the Studio entrypoint")
        if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
            raise RuntimeError("Wheel is missing the Apache-2.0 license file")
        forbidden_suffixes = (".joblib", ".mp4", ".mov", ".lrf", ".pt", ".pth", ".onnx")
        forbidden = sorted(name for name in names if name.lower().endswith(forbidden_suffixes))
        if forbidden:
            raise RuntimeError(f"Wheel contains private media or model artifacts: {forbidden[:8]}")


def build_wheel(output_directory: Path) -> Path:
    if not STATIC_ROOT.joinpath("index.html").exists():
        raise RuntimeError("Build and publish studio-web before creating the Python wheel")
    _clean_generated_build()
    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smart-badminton-wheel-") as temporary:
        temporary_directory = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(REPOSITORY_ROOT),
                "--no-deps",
                "--wheel-dir",
                str(temporary_directory),
            ],
            check=True,
        )
        wheels = list(temporary_directory.glob("smart_badminton-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one wheel, found {len(wheels)}")
        verify_wheel(wheels[0])
        destination = output_directory.resolve() / wheels[0].name
        shutil.copy2(wheels[0], destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify a reproducible Smart Badminton wheel")
    parser.add_argument("--output-directory", type=Path, default=REPOSITORY_ROOT / "dist")
    args = parser.parse_args()
    wheel = build_wheel(args.output_directory.expanduser().resolve())
    print(f"Verified wheel: {wheel}")


if __name__ == "__main__":
    main()
