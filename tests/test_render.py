import subprocess
from pathlib import Path

from smart_badminton.render import render_rallies
from smart_badminton.trajectory_overlay import create_trajectory_ass


def test_render_retries_cpu_when_compiled_hardware_encoder_cannot_start(tmp_path: Path, monkeypatch) -> None:
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,0,1\n", encoding="utf-8")
    commands: list[list[str]] = []

    def run(command: list[str], check: bool) -> None:
        assert check is True
        commands.append(command)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("smart_badminton.render.resolve_ffmpeg", lambda _path=None: Path("ffmpeg"))
    monkeypatch.setattr(
        "smart_badminton.encoding.available_ffmpeg_encoders",
        lambda _path=None: {"h264_nvenc", "h264_qsv"},
    )
    monkeypatch.setattr("smart_badminton.encoding.subprocess.run", run)

    used = render_rallies(tmp_path / "source.mp4", timeline, tmp_path / "edited.mp4", encoder="auto")

    assert used == "h264_qsv"
    assert "h264_nvenc" in commands[0]
    assert "h264_qsv" in commands[1]


def test_trajectory_ass_draws_only_connected_visible_points(tmp_path: Path) -> None:
    trajectory = tmp_path / "shuttle-track.csv"
    trajectory.write_text(
        "time_seconds,frame,center_x,center_y,confidence,source_width,source_height,status,flight_id\n"
        "1.00,30,100,100,0.9,1280,720,tracked,flight-1\n"
        "1.05,32,130,110,0.9,1280,720,tracked,flight-1\n"
        "1.10,33,160,120,0.9,1280,720,competing,flight-2\n"
        "1.15,35,190,130,0.9,1280,720,tracked,flight-1\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "trajectory.ass"

    event_count = create_trajectory_ass(trajectory, overlay)
    content = overlay.read_text(encoding="utf-8")

    assert event_count == 2
    assert "PlayResX: 1280" in content
    assert "Dialogue: 0,0:00:01.05" in content
    assert "Dialogue: 0,0:00:01.15" in content
    assert "flight-2" not in content


def test_render_adds_ass_filter_when_trajectory_is_requested(tmp_path: Path, monkeypatch) -> None:
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n2,3,4\n", encoding="utf-8")
    trajectory = tmp_path / "shuttle-track.csv"
    trajectory.write_text(
        "time_seconds,frame,center_x,center_y,confidence,source_width,source_height,status,flight_id\n"
        "1.00,30,100,100,0.9,1280,720,tracked,flight-1\n"
        "1.05,32,130,110,0.9,1280,720,tracked,flight-1\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def run(command: list[str], check: bool) -> None:
        assert check is True
        commands.append(command)

    monkeypatch.setattr("smart_badminton.render.resolve_ffmpeg", lambda _path=None: Path("ffmpeg"))
    monkeypatch.setattr("smart_badminton.encoding.available_ffmpeg_encoders", lambda _path=None: {"libx264"})
    monkeypatch.setattr("smart_badminton.encoding.subprocess.run", run)

    used = render_rallies(
        tmp_path / "source.mp4",
        timeline,
        tmp_path / "edited.mp4",
        encoder="libx264",
        trajectory_csv=trajectory,
    )

    assert used == "libx264"
    filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "ass=filename=" in filter_graph
    assert "split=2[marked0][marked1]" in filter_graph
    assert not list(tmp_path.glob(".trajectory-*.ass"))
