from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath

from smart_badminton.doctor import inspect_model_license

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FIXTURE = PurePosixPath("examples/privacy_safe_sample")
PRIVATE_DIRECTORIES = {".smart-badminton", "Analysis", "Calibration", "Edited", "Metadata", "Original", "Proxy"}
PRIVATE_EXTENSIONS = {".avi", ".lrf", ".mkv", ".mov", ".mp4", ".wav"}
MODEL_EXTENSIONS = {".joblib", ".onnx", ".pt", ".pth"}
SECRET_PATTERNS = (
    re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def tracked_files() -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        relative
        for value in result.stdout.split(b"\0")
        if value
        for relative in (PurePosixPath(value.decode("utf-8")),)
        if (ROOT / relative).is_file()
    ]


def is_private_path(path: PurePosixPath) -> bool:
    if path == SYNTHETIC_FIXTURE or SYNTHETIC_FIXTURE in path.parents:
        return False
    return bool(PRIVATE_DIRECTORIES.intersection(path.parts)) or path.suffix.lower() in PRIVATE_EXTENSIONS


def lfs_managed(path: PurePosixPath) -> bool:
    result = subprocess.run(
        ["git", "check-attr", "filter", "--", path.as_posix()],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip().endswith(": lfs")


def audit_repository() -> list[str]:
    errors: list[str] = []
    tracked = tracked_files()
    tracked_names = {path.as_posix() for path in tracked}
    for relative in tracked:
        if is_private_path(relative):
            errors.append(f"private runtime file is tracked: {relative}")
            continue
        path = ROOT / relative
        if relative.suffix.lower() in MODEL_EXTENSIONS:
            sidecar = f"{relative.as_posix()}.license.json"
            if sidecar not in tracked_names:
                errors.append(f"model license sidecar is missing: {relative}")
            elif not inspect_model_license(path).get("redistributable"):
                errors.append(f"model is not verified for redistribution: {relative}")
            if not lfs_managed(relative):
                errors.append(f"model is not managed by Git LFS: {relative}")
            continue
        try:
            content = path.read_bytes()
        except OSError as error:
            errors.append(f"cannot read tracked file {relative}: {error}")
            continue
        if b"\0" in content:
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            errors.append(f"credential-like value found in tracked file: {relative}")
    return errors


def main() -> None:
    errors = audit_repository()
    if errors:
        raise SystemExit("Repository publication guard failed:\n- " + "\n- ".join(errors))
    print(f"Repository publication guard passed ({len(tracked_files())} tracked files).")


if __name__ == "__main__":
    main()
