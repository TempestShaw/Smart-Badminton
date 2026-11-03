"""HTTP API for live sessions, shared by Studio and the stand-alone ``smart-badminton live`` server."""

from __future__ import annotations

import ipaddress
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..live import LiveSession, LiveSettings, write_corner_calibration
from ..players import MATCH_FORMATS, match_format_from_config

try:  # FastAPI resolves route annotations from module globals; Studio extras stay optional for the CLI.
    from fastapi import Request
except ImportError:  # pragma: no cover - only without the studio extra
    Request = Any  # type: ignore[assignment,misc]

MAXIMUM_UPLOAD_BYTES = 8 * 1024**3
MAXIMUM_FRAME_BYTES = 8 * 1024**2


class LiveManager:
    """At most one live session per server; it owns the camera."""

    def __init__(
        self,
        library: Path,
        pose_model: Path | None = None,
        shuttle_model: Path | None = None,
        rally_model: Path | None = None,
        packages: Path | None = None,
        ffmpeg: Path | None = None,
        studio_config: Callable[[], Path | None] | Path | None = None,
        session_factory: type[LiveSession] = LiveSession,
    ):
        self.library = library
        self.pose_model = pose_model
        self.shuttle_model = shuttle_model
        self.rally_model = rally_model
        self.packages = packages
        self.ffmpeg = ffmpeg
        self._studio_config = studio_config
        self.session_factory = session_factory
        self.session: LiveSession | None = None
        self.lock = threading.Lock()

    @property
    def studio_config(self) -> Path | None:
        """Studio's court calibration, read on demand because saving a calibration can change it."""
        return self._studio_config() if callable(self._studio_config) else self._studio_config

    @property
    def config(self) -> Path:
        return self.library / "Calibration" / "live-court-config.json"

    @property
    def recordings(self) -> Path:
        return self.library / "Live"

    def _available(self, path: Path | None) -> bool:
        return bool(path and path.exists())

    def idle_payload(self) -> dict[str, Any]:
        calibrated = self.config.exists()
        return {
            "state": "idle",
            "calibrated": calibrated,
            "match_format": match_format_from_config(self.config) if calibrated else "singles",
            "studio_calibration_available": self._available(self.studio_config),
            "models": {
                "pose": self._available(self.pose_model),
                "shuttle": self._available(self.shuttle_model),
                "rally": self._available(self.rally_model),
            },
            "recordings": str(self.recordings),
        }

    def status(self) -> dict[str, Any]:
        session = self.session
        if session is None:
            return self.idle_payload()
        return {**self.idle_payload(), **session.status()}

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = str(payload.get("source", "")).strip()
        match_format = str(payload.get("match_format") or "singles")
        if match_format not in MATCH_FORMATS:
            raise ValueError("match_format must be singles or doubles")
        flags = {name: payload.get(name, default) for name, default in (("shuttle", True), ("record", False), ("loop", False))}
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be true or false")
        if not source:
            raise ValueError("Choose a camera number, a stream URL, a video file or the browser camera")
        if source != "browser" and not source.isdigit() and "://" not in source and not Path(source).exists():
            raise ValueError(f"Live source not found: {source}")
        if flags["shuttle"] and not self._available(self.shuttle_model):
            raise ValueError("Shuttle tracking needs the YOLO shuttle model; install the model bundle or turn it off")
        with self.lock:
            if self.session is not None and self.session.state in {"starting", "running"}:
                raise RuntimeError("A live session is already running; stop it first")
            if self.config.exists() and match_format_from_config(self.config) != match_format:
                self._set_match_format(match_format)
            self.session = self.session_factory(
                LiveSettings(source=source, match_format=match_format, shuttle=flags["shuttle"], record=flags["record"], loop=flags["loop"]),
                self.config if self.config.exists() else None,
                self.recordings,
                pose_model=self.pose_model if self._available(self.pose_model) else None,
                shuttle_model=self.shuttle_model if flags["shuttle"] else None,
                rally_model=self.rally_model if self._available(self.rally_model) else None,
                packages=self.packages,
                ffmpeg=self.ffmpeg,
            )
            self.session.start()
        return self.status()

    def _set_match_format(self, match_format: str) -> None:
        import json

        data = json.loads(self.config.read_text(encoding="utf-8"))
        data.setdefault("analysis", {})["match_format"] = match_format
        self.config.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def stop(self) -> dict[str, Any]:
        with self.lock:
            session = self.session
            if session is None:
                return self.idle_payload()
            result = session.stop()
        return {**self.idle_payload(), **result}

    def require_session(self) -> LiveSession:
        if self.session is None or self.session.state not in {"starting", "running"}:
            raise RuntimeError("No live session is running")
        return self.session

    def calibrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        match_format = str(payload.get("match_format") or "singles")
        if payload.get("copy_from_studio"):
            if not self._available(self.studio_config):
                raise ValueError("Studio has no court calibration to copy")
            self.config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.studio_config, self.config)
            self._set_match_format(match_format)
        else:
            corners = payload.get("corners")
            if not isinstance(corners, list) or len(corners) != 4:
                raise ValueError("corners must list the four singles-court corners")
            session = self.require_session()
            if session.raw_size is None:
                raise RuntimeError("Wait for the first camera frame before calibrating")
            width, height = session.raw_size
            points = [[float(value) for value in corner] for corner in corners]
            if not all(len(point) == 2 and 0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0 for point in points):
                raise ValueError("Each corner must be a normalized [x, y] point inside the frame")
            write_corner_calibration(self.config, points, width, height, match_format)
        if self.session is not None:
            self.session.reload_calibration(self.config, match_format)
        return self.status()


def _safe_recording_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).stem).strip(".-") or "phone"
    suffix = Path(name).suffix.lower()
    return f"{stem[:60]}{suffix if suffix in {'.mp4', '.webm', '.mov', '.mkv'} else '.webm'}"


def add_live_routes(app: Any, manager: LiveManager) -> None:
    from fastapi.responses import Response, StreamingResponse

    @app.get("/api/live")
    def live_status():
        return manager.status()

    @app.post("/api/live/start")
    def live_start(payload: dict[str, Any]):
        return manager.start(payload)

    @app.post("/api/live/stop")
    def live_stop():
        return manager.stop()

    @app.put("/api/live/calibration")
    def live_calibration(payload: dict[str, Any]):
        return manager.calibrate(payload)

    @app.get("/api/live/stream.mjpg")
    def live_stream():
        session = manager.require_session()
        return StreamingResponse(
            session.frames("frame"),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/live/snapshot.jpg")
    def live_snapshot():
        session = manager.require_session()
        image = session.snapshot_jpeg()
        if image is None:
            raise RuntimeError("No camera frame has arrived yet")
        return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/live/frame")
    async def live_frame(request: Request):
        body = await request.body()
        if not body or len(body) > MAXIMUM_FRAME_BYTES:
            raise ValueError("A frame must be a JPEG under 8 MB")
        capture_time = request.headers.get("x-capture-time")
        manager.require_session().push_frame(body, float(capture_time) if capture_time else None)
        return {"accepted": True}

    @app.post("/api/live/audio")
    async def live_audio(request: Request):
        body = await request.body()
        if len(body) % 2 or len(body) > MAXIMUM_FRAME_BYTES:
            raise ValueError("Audio must be 16 kHz mono 16-bit PCM")
        manager.require_session().push_audio(body)
        return {"accepted": True}

    @app.post("/api/live/recording")
    async def live_recording(request: Request):
        """A browser's own recording (video with sound), streamed to the library for full offline analysis."""
        name = _safe_recording_name(request.headers.get("x-filename") or f"phone-{time.strftime('%Y%m%d-%H%M%S')}.webm")
        manager.recordings.mkdir(parents=True, exist_ok=True)
        destination = manager.recordings / name
        if destination.exists():
            destination = destination.with_name(f"{destination.stem}-{time.strftime('%H%M%S')}{destination.suffix}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        size = 0
        with temporary.open("wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAXIMUM_UPLOAD_BYTES:
                    handle.close()
                    temporary.unlink(missing_ok=True)
                    raise ValueError("Recordings larger than 8 GB must be copied into the library directly")
                handle.write(chunk)
        if size == 0:
            temporary.unlink(missing_ok=True)
            raise ValueError("The recording is empty")
        temporary.replace(destination)
        return {"path": str(destination), "bytes": size}


def create_live_app(manager: LiveManager, allowed_hosts: list[str] | None = None, allow_local_ips: bool = False):
    """A server with only the live page and API, for sessions that have no recorded library yet."""
    from importlib.resources import files

    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from .app import LOOPBACK_HOSTS, HostCheckMiddleware

    static_root = Path(str(files("smart_badminton").joinpath("studio_static")))
    app = FastAPI(title="Smart Badminton Live", docs_url=None, redoc_url=None)
    app.add_middleware(HostCheckMiddleware, allowed_hosts=allowed_hosts or LOOPBACK_HOSTS, allow_local_ips=allow_local_ips)
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    async def bad_request(_request: Any, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    for error_type in (KeyError, OSError, RuntimeError, TypeError, ValueError):
        app.add_exception_handler(error_type, bad_request)

    @app.get("/")
    def index():
        return FileResponse(static_root / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "mode": "live"}

    add_live_routes(app, manager)
    return app


def ensure_certificate(directory: Path, hosts: list[str]) -> tuple[Path, Path]:
    """A self-signed certificate for LAN HTTPS, which phone browsers require before they share a camera."""
    directory.mkdir(parents=True, exist_ok=True)
    certificate, key = directory / "studio-cert.pem", directory / "studio-key.pem"
    if certificate.exists() and key.exists():
        return certificate, key
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("HTTPS needs OpenSSL to create a local certificate; install OpenSSL or use a camera URL")
    names = []
    for host in hosts:
        bare = host.strip("[]")
        try:
            ipaddress.ip_address(bare)
            names.append(f"IP:{bare}")
        except ValueError:
            names.append(f"DNS:{bare}")
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "825",
            "-subj",
            "/CN=Smart Badminton Studio",
            "-addext",
            f"subjectAltName={','.join(dict.fromkeys(names))}",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    return certificate, key


def serve(app: Any, host: str, port: int, https_directory: Path | None, hosts: list[str]) -> None:
    import uvicorn

    options: dict[str, Any] = {}
    scheme = "http"
    if https_directory is not None:
        certificate, key = ensure_certificate(https_directory, hosts)
        options = {"ssl_certfile": str(certificate), "ssl_keyfile": str(key)}
        scheme = "https"
    shown = "localhost" if host in {"0.0.0.0", "::", ""} else host
    print(f"Smart Badminton: {scheme}://{shown}:{port}")
    for address in hosts:
        if address[0].isdigit() and not address.startswith("127."):
            print(f"  on your network: {scheme}://{address}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info", **options)


def run_live(
    library: Path,
    host: str = "0.0.0.0",
    port: int = 8765,
    source: str | None = None,
    match_format: str = "singles",
    shuttle: bool = True,
    record: bool = False,
    https: bool = False,
    pose_model: Path | None = None,
    shuttle_model: Path | None = None,
    rally_model: Path | None = None,
    packages: Path | None = None,
    ffmpeg: Path | None = None,
    open_browser: bool = False,
) -> None:
    """Serve only the live page; with ``source`` the session starts immediately."""
    import webbrowser

    from ..io import resolve_ffmpeg
    from ..model_assets import resolve_model
    from .app import WILDCARD_BINDS, allowed_hosts_for

    library = library.expanduser().resolve()
    library.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg_path: Path | None = resolve_ffmpeg(ffmpeg)
    except (FileNotFoundError, RuntimeError):
        ffmpeg_path = None
    manager = LiveManager(
        library,
        resolve_model(pose_model, "pose", library),
        resolve_model(shuttle_model, "shuttle", library),
        resolve_model(rally_model, "rally", library),
        packages,
        ffmpeg_path,
    )
    if source is not None:
        manager.start({"source": source, "match_format": match_format, "shuttle": shuttle and manager._available(manager.shuttle_model), "record": record})
    hosts = allowed_hosts_for(host)
    app = create_live_app(manager, hosts, allow_local_ips=host in WILDCARD_BINDS)
    if open_browser:
        scheme = "https" if https else "http"
        threading.Timer(0.8, lambda: webbrowser.open(f"{scheme}://127.0.0.1:{port}/?view=live")).start()
    try:
        serve(app, host, port, library / ".smart-badminton" / "tls" if https else None, hosts)
    finally:
        manager.stop()
