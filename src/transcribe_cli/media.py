# SPDX-License-Identifier: Apache-2.0
"""FFmpeg/ffprobe wrapper: describe inputs and normalise them to engine-ready WAV."""

from __future__ import annotations

import json
import subprocess
import time
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from transcribe_cli.errors import MediaError, PrerequisiteError
from transcribe_cli.segments import SAMPLE_RATE

if TYPE_CHECKING:
    from transcribe_cli.reporting import Reporter

WAV_LAYOUT = (1, SAMPLE_RATE, 2)  # channels, rate, sample width in bytes


def _run(command: Sequence[str], source: Path) -> subprocess.CompletedProcess[str]:
    tool = command[0]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise PrerequisiteError(f"{tool} not found on PATH") from error
    if result.returncode:
        raise MediaError(f"{tool} failed for {source}: {result.stderr.strip()}")
    return result


def describe(path: Path) -> str:
    result = _run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample",
         "-of", "json", str(path)],
        path,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise MediaError(f"No audio stream found: {path}")
    stream = streams[0]
    codec = stream.get("codec_name", "unknown").upper()
    if codec.startswith("PCM_"):
        codec = "PCM"
    bits = stream.get("bits_per_raw_sample")
    if not bits or not str(bits).isdigit() or int(bits) == 0:
        bits = stream.get("bits_per_sample")
    if bits and str(bits).isdigit() and int(bits) > 0:
        codec = f"{bits}-bit {codec}"
    channels = stream.get("channels")
    layout = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels} channels"
    rate = int(stream["sample_rate"]) / 1000
    container = path.suffix[1:].upper()
    format_name = f"{codec} audio" if codec == container else f"{codec} {container}"
    return f"{format_name}, {rate:g} kHz, {layout}"


def is_engine_ready_wav(path: Path) -> bool:
    """True for 16 kHz mono 16-bit PCM WAV, which the engine reads without conversion."""
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as reader:
            layout = (reader.getnchannels(), reader.getframerate(), reader.getsampwidth())
            return layout == WAV_LAYOUT and reader.getcomptype() == "NONE"
    except (wave.Error, EOFError):
        return False


def read_frame_count(wav: Path) -> int:
    with wave.open(str(wav), "rb") as reader:
        return reader.getnframes()


def normalize(source: Path, workdir: Path, reporter: Reporter,
              clock: Callable[[], float] = time.perf_counter) -> Path:
    """Return `source` if it is engine-ready, otherwise a converted `audio.wav` in `workdir`."""
    if is_engine_ready_wav(source):
        wav = source
    else:
        wav = workdir / "audio.wav"
        reporter.conversion_started(source)
        started = clock()
        _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
              "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(wav)], source)
        reporter.conversion_finished(clock() - started)
    with wave.open(str(wav), "rb") as reader:
        if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth()) != WAV_LAYOUT:
            raise MediaError(f"ffmpeg produced an invalid WAV: {source}")
        if reader.getnframes() == 0:
            raise MediaError(f"Decoded audio was empty: {source}")
    return wav
