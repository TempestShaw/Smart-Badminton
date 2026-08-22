import subprocess
from pathlib import Path

from smart_badminton import encoding


def test_qsv_arguments_do_not_silently_switch_to_libx264() -> None:
    arguments = encoding.h264_encoding_arguments("h264_qsv", 24, "preview")

    assert arguments[:2] == ["-c:v", "h264_qsv"]
    assert "-global_quality" in arguments
    assert "libx264" not in arguments


def test_openh264_is_a_real_software_fallback(monkeypatch) -> None:
    monkeypatch.setattr(encoding, "available_ffmpeg_encoders", lambda _path=None: {"libopenh264"})

    candidates, warning = encoding.video_encoder_candidates("auto")

    assert candidates == ["libopenh264"]
    assert warning is None
    assert encoding.h264_encoding_arguments("libopenh264", 25, "proxy") == [
        "-c:v",
        "libopenh264",
        "-b:v",
        "4M",
    ]


def test_working_encoder_probe_skips_compiled_but_unusable_hardware(monkeypatch) -> None:
    monkeypatch.setattr(
        encoding,
        "video_encoder_candidates",
        lambda _requested, _ffmpeg=None: (["h264_nvenc", "h264_qsv", "libx264"], None),
    )
    monkeypatch.setattr(
        encoding,
        "probe_video_encoder",
        lambda candidate, _ffmpeg=None: (candidate == "h264_qsv", "device unavailable"),
    )

    selected, warning, failures = encoding.choose_working_video_encoder("auto", Path("ffmpeg"))

    assert selected == "h264_qsv"
    assert failures == {"h264_nvenc": "device unavailable"}
    assert "could not start" in str(warning)


def test_all_runtime_encoder_failures_are_reported_and_partial_output_removed(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "partial.mp4"
    commands: list[list[str]] = []
    monkeypatch.setattr(
        encoding,
        "video_encoder_candidates",
        lambda _requested, _ffmpeg=None: (["h264_nvenc", "h264_qsv"], None),
    )

    def fail(command: list[str], check: bool) -> None:
        assert check is True
        commands.append(command)
        output.write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(encoding.subprocess, "run", fail)

    try:
        encoding.run_ffmpeg_with_encoder_fallback(
            Path("ffmpeg"),
            "auto",
            lambda candidate: ["ffmpeg", "-c:v", candidate, str(output)],
            output,
        )
    except RuntimeError as error:
        assert "h264_nvenc, h264_qsv" in str(error)
    else:
        raise AssertionError("Expected every encoder to fail")

    assert len(commands) == 2
    assert not output.exists()
