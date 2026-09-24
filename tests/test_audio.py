import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

from smart_badminton.audio import analyze_audio


def test_audio_analysis_stops_before_work_when_the_video_has_no_audio(tmp_path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=1"]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )
    output = tmp_path / "audio-events.csv"

    with pytest.raises(ValueError, match="silent.mp4 has no audio track"):
        analyze_audio(silent, output, ffmpeg=Path(ffmpeg))
    assert not output.exists()
