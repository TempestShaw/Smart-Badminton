import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import imageio_ffmpeg
import pytest

from smart_badminton.encoding import has_audio_stream
from smart_badminton.render import render_rallies
from smart_badminton.score_overlay import create_score_ass
from smart_badminton.trajectory_overlay import create_trajectory_ass


def test_render_retries_cpu_when_compiled_hardware_encoder_cannot_start(tmp_path: Path, monkeypatch) -> None:
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,0,1\n", encoding="utf-8")
    commands: list[list[str]] = []

    def run(command: list[str]) -> None:
        commands.append(command)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("smart_badminton.render.resolve_ffmpeg", lambda _path=None: Path("ffmpeg"))
    monkeypatch.setattr("smart_badminton.render.has_audio_stream", lambda _ffmpeg, _video: True)
    monkeypatch.setattr(
        "smart_badminton.encoding.available_ffmpeg_encoders",
        lambda _path=None: {"h264_nvenc", "h264_qsv"},
    )
    monkeypatch.setattr("smart_badminton.encoding.run_ffmpeg", run)

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

    monkeypatch.setattr("smart_badminton.render.resolve_ffmpeg", lambda _path=None: Path("ffmpeg"))
    monkeypatch.setattr("smart_badminton.render.has_audio_stream", lambda _ffmpeg, _video: True)
    monkeypatch.setattr("smart_badminton.encoding.available_ffmpeg_encoders", lambda _path=None: {"libx264"})
    monkeypatch.setattr("smart_badminton.encoding.run_ffmpeg", commands.append)

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


def test_score_ass_updates_at_the_end_of_each_rally(tmp_path: Path) -> None:
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,3\n2,4,6\n", encoding="utf-8")
    score = tmp_path / "score.csv"
    score.write_text(
        "rally,winner,near_score,far_score,game_finished\n"
        "1,near,1,0,\n"
        "2,far,1,1,\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "score.ass"

    event_count = create_score_ass(timeline, score, overlay)
    content = overlay.read_text(encoding="utf-8")

    assert event_count == 4
    assert "0:00:01.00,0:00:02.35" in content
    assert "近场  0  :  0  远场" in content
    assert "0:00:02.35,0:00:03.00" in content
    assert "近场  1  :  0  远场" in content
    assert "0:00:05.35,0:00:06.00" in content
    assert "近场  1  :  1  远场" in content


def test_render_filters_winner_and_adds_score_overlay(tmp_path: Path, monkeypatch) -> None:
    timeline = tmp_path / "rallies.csv"
    timeline.write_text(
        "rally,start_seconds,end_seconds\n1,1,2\n2,3,4\n3,5,6\n",
        encoding="utf-8",
    )
    score = tmp_path / "score.csv"
    score.write_text(
        "rally,winner,near_score,far_score,game_finished\n"
        "1,near,1,0,\n"
        "2,far,1,1,\n"
        "3,unknown,1,1,\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    monkeypatch.setattr("smart_badminton.render.resolve_ffmpeg", lambda _path=None: Path("ffmpeg"))
    monkeypatch.setattr("smart_badminton.render.has_audio_stream", lambda _ffmpeg, _video: True)
    monkeypatch.setattr("smart_badminton.encoding.available_ffmpeg_encoders", lambda _path=None: {"libx264"})
    monkeypatch.setattr("smart_badminton.encoding.run_ffmpeg", commands.append)

    render_rallies(
        tmp_path / "source.mp4",
        timeline,
        tmp_path / "near-winners.mp4",
        encoder="libx264",
        score_csv=score,
        winner_filter="near",
        include_score=True,
    )

    filter_graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "score.ass" in filter_graph
    assert "trim=start=1.000:end=2.000" in filter_graph
    assert "trim=start=3.000:end=4.000" not in filter_graph
    assert "concat=n=1" in filter_graph


def test_render_cuts_video_only_when_the_source_has_no_audio(tmp_path: Path, monkeypatch) -> None:
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n2,3,4\n", encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr("smart_badminton.render.resolve_ffmpeg", lambda _path=None: Path("ffmpeg"))
    monkeypatch.setattr("smart_badminton.render.has_audio_stream", lambda _ffmpeg, _video: False)
    monkeypatch.setattr("smart_badminton.encoding.available_ffmpeg_encoders", lambda _path=None: {"libx264"})
    monkeypatch.setattr("smart_badminton.encoding.run_ffmpeg", commands.append)

    render_rallies(tmp_path / "silent.mp4", timeline, tmp_path / "edited.mp4", encoder="libx264")

    command = commands[0]
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "[0:a:0]" not in filter_graph
    assert filter_graph.endswith("[v0][v1]concat=n=2:v=1:a=0[vout]")
    assert "-an" in command
    assert "[aout]" not in command


def _synthetic_clip(path: Path, with_audio: bool) -> Path:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=2"]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "-shortest"]
    subprocess.run([*command, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_render_real_ffmpeg_with_and_without_source_audio(tmp_path: Path) -> None:
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,0.1,0.9\n2,1.2,1.8\n", encoding="utf-8")

    for with_audio in (False, True):
        source = _synthetic_clip(tmp_path / f"source-{with_audio}.mp4", with_audio)
        output = tmp_path / f"edited-{with_audio}.mp4"
        assert has_audio_stream(ffmpeg, source) is with_audio

        render_rallies(source, timeline, output, ffmpeg=ffmpeg, encoder="libx264")

        assert output.stat().st_size > 0
        assert has_audio_stream(ffmpeg, output) is with_audio


@pytest.mark.parametrize("rate", ["30000/1001", "30", "60"])
def test_render_keeps_the_source_frame_rate_without_dropping_frames(tmp_path: Path, rate: str) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source = tmp_path / "source.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=320x180:rate={rate}:duration=3"]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,0.5,1.5\n2,2.0,2.5\n", encoding="utf-8")
    output = tmp_path / "edited.mp4"

    render_rallies(source, timeline, output, ffmpeg=Path(ffmpeg), encoder="libx264")

    source_fps = float(Fraction(rate))
    capture = cv2.VideoCapture(str(output))
    output_fps, frames = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    assert output_fps == pytest.approx(source_fps, abs=0.01)
    assert frames == pytest.approx(1.5 * source_fps, abs=2)
