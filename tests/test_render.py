import subprocess
from pathlib import Path

from smart_badminton.render import render_rallies


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
