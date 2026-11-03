import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from smart_badminton.audio import StreamingHitDetector, analyze_audio
from smart_badminton.live import LiveRallyDetector, flight_evidence, write_corner_calibration
from smart_badminton.players import match_format_from_config
from smart_badminton.studio.live_routes import LiveManager, create_live_app

CORNERS = [[0.25, 0.95], [0.75, 0.95], [0.64, 0.45], [0.36, 0.45]]


def synthetic_video(path: Path, seconds: float = 2.0, fps: float = 20.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    for index in range(round(seconds * fps)):
        frame = np.full((180, 320, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (20 + index * 3 % 280, 90), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not reached")
        time.sleep(0.05)


def test_live_rally_detector_uses_offline_thresholds_with_hysteresis() -> None:
    detector = LiveRallyDetector(start_hold=0.3, end_hold=1.0, preroll=0.2, postroll=0.5)
    changes = []
    for step in range(80):
        t = step * 0.1
        evidence = 0.4 if 1.0 <= t < 1.4 else 0.9 if 1.4 <= t < 4.0 else 0.35 if 4.0 <= t < 4.6 else 0.05
        change = detector.update(t, evidence)
        if change:
            changes.append((change, round(t, 1)))

    assert changes == [("start", 1.7), ("end", 5.5)]
    rally = detector.rallies[0]
    assert rally.start == pytest.approx(1.0 - 0.2)  # backtracks to where evidence first passed the keep level
    assert rally.end == pytest.approx(4.5 + 0.5)


def test_flight_evidence_separates_flight_from_a_carried_shuttle() -> None:
    flying = deque((0.2 + 0.08 * step, 0.5, step * 0.15) for step in range(7))
    carried = deque((0.5 + 0.004 * step, 0.7, step * 0.15) for step in range(7))

    assert flight_evidence(flying, 0.9) > 0.8
    assert flight_evidence(carried, 0.9) == 0.0


def test_streaming_hit_detector_matches_the_offline_detector(tmp_path: Path) -> None:
    import wave

    rng = np.random.default_rng(3)
    rate, hits = 16000, [1.5, 2.6, 4.1, 6.0, 7.7]
    samples = rng.normal(0, 0.01, rate * 9).astype(np.float32)
    for hit in hits:
        start = int(hit * rate)
        samples[start : start + 400] += (rng.normal(0, 0.5, 400) * np.exp(-np.arange(400) / 80)).astype(np.float32)
    wav = tmp_path / "hits.wav"
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    analyze_audio(wav, tmp_path / "hits.csv")
    offline = [float(line.split(",")[0]) for line in (tmp_path / "hits.csv").read_text().splitlines()[1:]]
    detector = StreamingHitDetector(rate)
    streamed = []
    for start in range(0, len(samples), 3200):
        streamed += [event[0] for event in detector.push(samples[start : start + 3200])]
    streamed += [event[0] for event in detector.flush()]

    assert offline == pytest.approx(hits, abs=0.02)
    assert streamed == pytest.approx(offline, abs=0.02)


def test_corner_calibration_writes_a_complete_config(tmp_path: Path) -> None:
    config = write_corner_calibration(tmp_path / "live.json", CORNERS, 1280, 720, "doubles")
    data = json.loads(config.read_text(encoding="utf-8"))

    assert match_format_from_config(config) == "doubles"
    for region in ("active_court_polygon", "near_player_zone", "far_player_zone", "net_band", "shuttle_airspace_polygon"):
        assert len(data["masks_normalized"][region]) >= 4
    near_y = [point[1] for point in data["masks_normalized"]["near_player_zone"]]
    far_y = [point[1] for point in data["masks_normalized"]["far_player_zone"]]
    assert min(near_y) > max(far_y) - 0.02  # the halves meet at the net


def test_live_api_streams_a_file_source_and_calibrates_from_four_corners(tmp_path: Path) -> None:
    video = synthetic_video(tmp_path / "court.mp4")
    manager = LiveManager(tmp_path / "library")
    client = TestClient(create_live_app(manager), base_url="http://127.0.0.1")

    assert client.get("/api/health").json()["mode"] == "live"
    assert client.get("/api/live").json()["state"] == "idle"
    rejected = client.post("/api/live/start", json={"source": str(tmp_path / "missing.mp4"), "shuttle": False})
    assert rejected.status_code == 400
    no_model = client.post("/api/live/start", json={"source": str(video)})
    assert "shuttle model" in no_model.json()["detail"]

    started = client.post("/api/live/start", json={"source": str(video), "shuttle": False, "loop": True})
    assert started.status_code == 200 and started.json()["calibrated"] is False
    wait_for(lambda: manager.session.raw_size is not None and manager.session.jpeg is not None)
    snapshot = client.get("/api/live/snapshot.jpg")
    assert snapshot.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(snapshot.content, np.uint8), cv2.IMREAD_COLOR).shape == (180, 320, 3)
    assert client.post("/api/live/start", json={"source": str(video), "shuttle": False}).status_code == 400

    calibrated = client.put("/api/live/calibration", json={"corners": CORNERS, "match_format": "doubles"})
    assert calibrated.status_code == 200 and calibrated.json()["calibrated"] is True
    wait_for(lambda: "analysis" in manager.status())
    analysis = manager.status()["analysis"]
    assert analysis["match_format"] == "doubles" and analysis["metric"] is True
    assert manager.status()["frames_analyzed"] > 0

    part = next(manager.session.frames("frame"))
    assert part.startswith(b"--frame\r\nContent-Type: image/jpeg")
    stopped = client.post("/api/live/stop").json()
    assert stopped["state"] == "stopped"


def test_live_api_accepts_browser_camera_frames_and_recordings(tmp_path: Path) -> None:
    manager = LiveManager(tmp_path / "library")
    client = TestClient(create_live_app(manager), base_url="http://127.0.0.1")
    client.post("/api/live/start", json={"source": "browser", "shuttle": False})
    frame = cv2.imencode(".jpg", np.full((90, 160, 3), 128, np.uint8))[1].tobytes()

    assert client.post("/api/live/frame", content=frame, headers={"x-capture-time": "0.5"}).json() == {"accepted": True}
    assert client.post("/api/live/frame", content=b"not an image").status_code == 400
    assert client.post("/api/live/audio", content=np.zeros(1600, "<i2").tobytes()).status_code == 200
    wait_for(lambda: manager.session.raw_size == (160, 90))

    upload = client.post("/api/live/recording", content=b"\x1aE\xdf\xa3 fake webm", headers={"x-filename": "../../evil name.webm"})
    saved = Path(upload.json()["path"])
    assert saved.parent == tmp_path / "library" / "Live" and saved.name == "evil-name.webm"
    client.post("/api/live/stop")


def test_live_routes_reject_foreign_hosts(tmp_path: Path) -> None:
    client = TestClient(create_live_app(LiveManager(tmp_path)), base_url="http://attacker.example")

    assert client.get("/api/live").status_code == 400
