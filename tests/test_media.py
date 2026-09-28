# SPDX-License-Identifier: Apache-2.0
import subprocess
import wave
from pathlib import Path

import pytest

from conftest import RecordingReporter, write_wav
from transcribe_cli import media
from transcribe_cli.errors import MediaError, PrerequisiteError

FIXTURES = Path(__file__).parent / "fixtures"


def test_engine_ready_wav_detection(tmp_path: Path) -> None:
    assert media.is_engine_ready_wav(write_wav(tmp_path / "ok.wav", 1))
    assert not media.is_engine_ready_wav(write_wav(tmp_path / "rate.wav", 1, rate=48000))
    assert not media.is_engine_ready_wav(write_wav(tmp_path / "stereo.wav", 1, channels=2))
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not a wav")
    assert not media.is_engine_ready_wav(broken)
    assert not media.is_engine_ready_wav(FIXTURES / "sample_15s.aac")


def test_normalize_passes_ready_wav_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_subprocess(*_: object, **__: object) -> None:
        raise AssertionError("no subprocess expected")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    reporter = RecordingReporter()
    source = write_wav(tmp_path / "ok.wav", 1)
    assert media.normalize(source, tmp_path, reporter) == source
    assert reporter.calls == []


def test_normalize_converts_and_reports(tmp_path: Path) -> None:
    reporter = RecordingReporter()
    ticks = iter([10.0, 12.5])
    source = write_wav(tmp_path / "stereo.wav", 1, rate=44100, channels=2)
    wav = media.normalize(source, tmp_path, reporter, lambda: next(ticks))
    assert wav == tmp_path / "audio.wav"
    assert media.read_frame_count(wav) == 16000
    assert reporter.calls == [("conversion_started", source), ("conversion_finished", 2.5)]


def test_describe_and_convert_aac_fixture(tmp_path: Path) -> None:
    assert media.describe(FIXTURES / "sample_15s.aac").startswith("AAC audio, ")
    wav = media.normalize(FIXTURES / "sample_15s.aac", tmp_path, RecordingReporter())
    with wave.open(str(wav), "rb") as reader:
        assert reader.getnframes() > 15 * 16000


def test_describe_wav(tmp_path: Path) -> None:
    assert media.describe(write_wav(tmp_path / "a.wav", 1)) == "16-bit PCM WAV, 16 kHz, mono"
    assert media.describe(write_wav(tmp_path / "b.wav", 1, rate=44100, channels=2)) == \
        "16-bit PCM WAV, 44.1 kHz, stereo"


def test_ffmpeg_failure_is_a_media_error(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.mp3"
    garbage.write_bytes(b"definitely not audio")
    with pytest.raises(MediaError, match="ffprobe failed|No audio stream"):
        media.describe(garbage)


def test_missing_tool_is_a_prerequisite_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_: object, **__: object) -> None:
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(PrerequisiteError, match="ffprobe not found on PATH"):
        media.describe(tmp_path / "x.wav")


def test_empty_audio_is_rejected(tmp_path: Path) -> None:
    empty = write_wav(tmp_path / "empty.wav", 0)
    with pytest.raises(MediaError, match="Decoded audio was empty"):
        media.normalize(empty, tmp_path, RecordingReporter())
