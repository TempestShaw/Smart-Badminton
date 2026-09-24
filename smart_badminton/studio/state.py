from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..project_layout import LibraryLayout, ProjectLayout

API_SCHEMA_VERSION = 8


@dataclass
class StudioState:
    video: Path
    rallies: Path
    output: Path
    proxy: Path | None = None
    ffmpeg: Path | None = None
    encoder: str = "auto"
    quality: int = 21
    library: Path | None = None
    output_directory: Path | None = None
    config: Path | None = None
    model: Path | None = None
    pose_model: Path | None = None
    shuttle_model: Path | None = None
    tracknet_model: Path | None = None
    inpaint_model: Path | None = None
    shuttle_mode: str = "hybrid"
    packages: Path | None = None
    tracknet_packages: Path | None = None
    analysis_options: dict[str, float | bool] = field(
        default_factory=lambda: {
            "preroll": 0.35,
            "postroll": 0.55,
            "end_pending": 0.55,
            "maximum_internal_gap": 1.2,
            "suppress_handoffs": True,
        }
    )
    render_status: dict[str, Any] = field(default_factory=lambda: {"state": "idle"})
    analysis_status: dict[str, Any] = field(default_factory=lambda: {"state": "idle"})
    render_lock: threading.Lock = field(default_factory=threading.Lock)
    analysis_lock: threading.Lock = field(default_factory=threading.Lock)
    project_lock: threading.Lock = field(default_factory=threading.Lock)
    pose_lock: threading.Lock = field(default_factory=threading.Lock)
    runtime_cache: dict[str, Any] = field(default_factory=dict)
    payload_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=dict)
    payload_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    status_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def library_root(self) -> Path:
        return self.library or self.video.parent

    def layout(self, video: Path | None = None) -> ProjectLayout:
        return ProjectLayout.for_video(self.library_root, video or self.video)

    def config_ready(self) -> bool:
        return bool(self.config and self.config.exists())


def _path_signature(paths: tuple[Path | None, ...]) -> tuple[Any, ...]:
    signature: list[Any] = []
    for path in paths:
        if path is None or not path.exists():
            signature.append(str(path))
            continue
        stat = path.stat()
        signature.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def cached_payload(
    state: StudioState,
    key: str,
    paths: tuple[Path | None, ...],
    builder: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild a derived payload only when one of its input files changes.

    A failing builder degrades to an unavailable payload so one bad artifact cannot take down the whole project view.
    """
    signature = _path_signature(paths)
    with state.payload_cache_lock:
        cached = state.payload_cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
    try:
        payload = builder()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        payload = {"available": False, "reason": str(error)}
    with state.payload_cache_lock:
        state.payload_cache[key] = (_path_signature(paths), payload)
    return payload


def _analysis_status_path(state: StudioState) -> Path:
    return LibraryLayout(state.library_root).analysis_status


def _publish_analysis_status(state: StudioState, status: dict[str, Any], force: bool = False) -> None:
    """Save a status, then expose it, so a client that sees a state can rely on the file holding it too."""
    with state.status_lock:
        now = time.monotonic()
        terminal = status.get("state") in {"complete", "error"}
        if force or terminal or now - float(state.runtime_cache.get("analysis_status_write", 0.0)) >= 0.5:
            path = _analysis_status_path(state)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
            state.runtime_cache["analysis_status_write"] = now
        state.analysis_status = status


# Status payloads carry their own "state" key, so the Studio argument is named ``studio_state`` here.
def set_analysis_status(studio_state: StudioState, **values: Any) -> None:
    _publish_analysis_status(studio_state, {**studio_state.analysis_status, **values, "updated_at": time.time()})


def begin_analysis_status(studio_state: StudioState, **values: Any) -> dict[str, Any]:
    now = time.time()
    status = {
        "job_id": uuid.uuid4().hex[:12],
        "state": "running",
        "background": True,
        "started_at": now,
        "updated_at": now,
        **values,
    }
    _publish_analysis_status(studio_state, status, force=True)
    return studio_state.analysis_status


def restore_analysis_status(state: StudioState) -> None:
    path = _analysis_status_path(state)
    if not path.exists():
        return
    try:
        restored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return  # A truncated status file must not stop Studio from starting.
    if not isinstance(restored, dict):
        return
    if restored.get("state") == "running":
        restored = {
            **restored,
            "state": "error",
            "label": "上次后台任务因 Studio 服务停止而中断",
            "message": "重新打开对应视频并再次启动分析；已完成的缓存步骤会被复用。",
            "updated_at": time.time(),
        }
    state.analysis_status = restored
