# SPDX-License-Identifier: Apache-2.0
"""Black-box tests: the real CLI process against a fake transcribe.cpp binary."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import FakeEngineBinary

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ENTRY = os.environ.get("CLI_ENTRY", "-m transcribe_cli").split()


def run_cli(engine: FakeEngineBinary, model_file: Path, *args: str | Path,
            model_id: bool = True) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **engine.env(), "PYTHONPATH": str(ROOT / "src")}
    argv = [sys.executable, *ENTRY, *map(str, args)]
    if model_id:
        argv += ["--model-id", str(model_file)]
    return subprocess.run(argv, capture_output=True, text=True, env=env, cwd=ROOT)


def command_of(call: dict[str, list[str]]) -> list[str]:
    return call["argv"]


def option(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_passthrough_wav_writes_transcript(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                           wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(default="Hello world.")
    wav = wav_factory("talk.wav", 5)
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode == 0, result.stderr
    assert wav.with_suffix(".txt").read_bytes() == b"Hello world.\n"
    lines = result.stdout.splitlines()
    assert lines[0] == f"Processing file: {wav}"
    assert lines[1] == "Input format: 16-bit PCM WAV, 16 kHz, mono"
    assert lines[2] == "Running transcribe.cpp with cohere on 1 audio chunk(s)..."
    assert lines[3] == f"Wrote transcript to {wav.with_suffix('.txt')} with {dummy_gguf} using Vulkan GPU"
    metrics = rf"Metrics for {re.escape(str(wav))}: 5\.00s audio; batch completed in \d+\.\d{{2}}s \(\d+\.\d{{2}}x\)"
    assert re.fullmatch(metrics, lines[4])
    assert not any(line.startswith("Converting") for line in lines)
    assert result.stderr.splitlines() == ["Chunks processed: 1/1"]
    (call,) = fake_engine.calls()
    argv = command_of(call)
    assert argv[0:1] == ["-m"] and argv[1] == str(dummy_gguf)
    assert argv[2:4] == ["--backend", "vulkan"]
    assert "--batch-jsonl" in argv and option(argv, "--timestamps") == "none"
    assert option(argv, "--batch-size") == "1"
    assert option(argv, "-l") == "en"
    assert call["paths"] == [str(wav)]


def test_non_engine_format_is_converted(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                        wav_factory: Callable[..., Path]) -> None:
    wav = wav_factory("stereo.wav", 2, rate=44100, channels=2)
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[1] == "Input format: 16-bit PCM WAV, 44.1 kHz, stereo"
    assert lines[2] == f"Converting to 16-bit PCM WAV, 16 kHz, mono: {wav}"
    assert re.fullmatch(r"Conversion completed in \d+\.\d{2}s", lines[3])
    assert lines[4] == "Running transcribe.cpp with cohere on 1 audio chunk(s)..."
    (call,) = fake_engine.calls()
    assert call["paths"][0].endswith("audio.wav")


def test_aac_fixture_is_converted(fake_engine: FakeEngineBinary, dummy_gguf: Path, tmp_path: Path) -> None:
    source = tmp_path / "sample.aac"
    source.write_bytes((FIXTURES / "sample_15s.aac").read_bytes())
    result = run_cli(fake_engine, dummy_gguf, source)
    assert result.returncode == 0, result.stderr
    assert "Input format: AAC audio" in result.stdout
    assert source.with_suffix(".txt").read_text() == "ok\n"
    # 15 s of audio with the cohere 30 s limit is a single chunk.
    assert len(fake_engine.calls()[0]["paths"]) == 1


def test_long_file_is_chunked_and_stitched(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                           wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(texts={
        "chunk-0000.wav": "Please review the guest room door.",
        "chunk-0001.wav": "the guest room door. Please find the exits.",
        "chunk-0002.wav": "Please find the exits. Then leave quickly.",
        "chunk-0003.wav": "Then leave quickly. Goodbye.",
    })
    wav = wav_factory("long.wav", 95)
    result = run_cli(fake_engine, dummy_gguf, wav, "--batch-size", "2")
    assert result.returncode == 0, result.stderr
    assert wav.with_suffix(".txt").read_text() == (
        "Please review the guest room door. Please find the exits. Then leave quickly. Goodbye.\n")
    assert "on 4 audio chunk(s)" in result.stdout
    (call,) = fake_engine.calls()
    assert len(call["paths"]) == 4
    # chunks are 30/30/30/11 s long, so lengths differ and batching is disabled.
    assert option(command_of(call), "--batch-size") == "1"
    assert result.stderr.splitlines() == [f"Chunks processed: {n}/4" for n in (1, 2, 3, 4)]


def test_equal_length_chunks_keep_batch_size(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                             wav_factory: Callable[..., Path]) -> None:
    # 30 s chunks, 3 s overlap: 57 s -> starts 0 and 27 -> lengths 30 and 30.
    wav = wav_factory("even.wav", 57)
    result = run_cli(fake_engine, dummy_gguf, wav, "--batch-size", "2")
    assert result.returncode == 0, result.stderr
    (call,) = fake_engine.calls()
    assert len(call["paths"]) == 2
    assert option(command_of(call), "--batch-size") == "2"


def test_unaligned_overlap_warns(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                 wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(texts={"chunk-0000.wav": "Please open the door.",
                                 "chunk-0001.wav": "Do not open the door. There is smoke."})
    wav = wav_factory("warn.wav", 40)
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode == 0, result.stderr
    assert (f"Warning: could not align 1 chunk overlap(s) for {wav}; check for repeated words"
            in result.stderr)
    assert wav.with_suffix(".txt").read_text() == "Please open the door. Do not open the door. There is smoke.\n"


def test_truncated_chunk_is_split_and_only_it_retried(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                                      wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(truncate_longer_than=20, texts={
        "chunk-0000-part-0.wav": "one two three four five six",
        "chunk-0000-part-1.wav": "four five six seven eight nine",
        "chunk-0001.wav": "seven eight nine ten eleven",
    })
    wav = wav_factory("trunc.wav", 40)  # chunks: 30 s (truncated) and 13 s
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode == 0, result.stderr
    first, retry = fake_engine.calls()
    assert [Path(p).name for p in first["paths"]] == ["chunk-0000.wav", "chunk-0001.wav"]
    assert [Path(p).name for p in retry["paths"]] == ["chunk-0000-part-0.wav", "chunk-0000-part-1.wav"]
    assert option(command_of(retry), "--batch-size") == "1"
    assert "Retrying truncated audio in shorter pieces:" in result.stderr
    assert wav.with_suffix(".txt").read_text() == "one two three four five six seven eight nine ten eleven\n"


def test_truncation_below_two_seconds_is_an_error(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                                  wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(truncate_longer_than=0.5)
    wav = wav_factory("tiny.wav", 3)
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode != 0
    assert "shorter than two seconds" in result.stderr
    assert not wav.with_suffix(".txt").exists()


def test_multiple_files_report_each_as_completed(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                                 wav_factory: Callable[..., Path]) -> None:
    first, second = wav_factory("a.wav", 2), wav_factory("b.wav", 3)
    result = run_cli(fake_engine, dummy_gguf, first, second)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    metrics = [i for i, line in enumerate(lines) if line.startswith("Metrics for")]
    assert len(metrics) == 2
    assert lines[metrics[0] + 1] == f"Processing file: {second}"
    assert first.with_suffix(".txt").exists() and second.with_suffix(".txt").exists()


@pytest.mark.parametrize(("model", "language", "expected"), [
    ("cohere", None, "en"),
    ("cohere", "auto", "auto"),
    ("cohere", "fr", "fr"),
    ("nemotron", None, None),
    ("nemotron", "auto", None),
    ("nemotron", "de", "de"),
    ("parakeet", None, None),
    ("parakeet", "en", None),
    ("qwen", None, None),
])
def test_language_policy(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                         wav_factory: Callable[..., Path], model: str,
                         language: str | None, expected: str | None) -> None:
    wav = wav_factory("lang.wav", 1)
    extra = ["--language", language] if language else []
    result = run_cli(fake_engine, dummy_gguf, wav, "--model", model, *extra)
    assert result.returncode == 0, result.stderr
    assert option(command_of(fake_engine.calls()[0]), "-l") == expected


@pytest.mark.parametrize(("model", "language"), [("qwen", "zh"), ("parakeet", "fr")])
def test_rejected_language(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                           wav_factory: Callable[..., Path], model: str, language: str) -> None:
    wav = wav_factory("lang.wav", 1)
    result = run_cli(fake_engine, dummy_gguf, wav, "--model", model, "--language", language)
    assert result.returncode != 0
    assert fake_engine.calls() == []


@pytest.mark.parametrize("model", ["parakeet", "nemotron"])
def test_serial_models_force_batch_size_one(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                            wav_factory: Callable[..., Path], model: str) -> None:
    wav = wav_factory("serial.wav", 1)
    result = run_cli(fake_engine, dummy_gguf, wav, "--model", model, "--batch-size", "4")
    assert result.returncode == 0, result.stderr
    assert option(command_of(fake_engine.calls()[0]), "--batch-size") == "1"


@pytest.mark.parametrize(("model", "seconds", "chunks"), [
    ("cohere", 31, 2), ("qwen", 61, 2), ("nemotron", 61, 2), ("parakeet", 301, 2),
    ("cohere", 30, 1), ("qwen", 60, 1), ("parakeet", 300, 1),
])
def test_model_chunk_limits(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                            wav_factory: Callable[..., Path], model: str, seconds: int, chunks: int) -> None:
    wav = wav_factory("limit.wav", seconds)
    result = run_cli(fake_engine, dummy_gguf, wav, "--model", model)
    assert result.returncode == 0, result.stderr
    assert len(fake_engine.calls()[0]["paths"]) == chunks


def test_no_vulkan_gpu_fails_before_processing(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                               wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(devices="CPU0: kind=cpu type=cpu")
    wav = wav_factory("gpu.wav", 1)
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode != 0
    assert "CPU fallback is disabled" in result.stderr
    assert "Processing file" not in result.stdout
    assert fake_engine.calls() == []


def test_engine_failure_reports_stderr(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                       wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(exit=3)
    wav = wav_factory("boom.wav", 1)
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode != 0
    assert "progress boom.wav" in result.stderr


def test_engine_tolerates_junk_and_partial_lines(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                                 wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(junk_line=True, partial_last_line=True)
    wav = wav_factory("junk.wav", 1)
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode == 0, result.stderr
    assert wav.with_suffix(".txt").read_text() == "ok\n"


def test_per_file_engine_error_is_reported(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                           wav_factory: Callable[..., Path]) -> None:
    fake_engine.configure(errors={"bad.wav": "unsupported language"})
    wav = wav_factory("bad.wav", 1)
    result = run_cli(fake_engine, dummy_gguf, wav)
    assert result.returncode != 0
    assert "unsupported language" in result.stderr


def test_missing_input_and_bad_extension(fake_engine: FakeEngineBinary, dummy_gguf: Path, tmp_path: Path) -> None:
    result = run_cli(fake_engine, dummy_gguf, tmp_path / "nope.wav")
    assert result.returncode != 0 and "Input file not found" in result.stderr
    text = tmp_path / "notes.txt"
    text.touch()
    result = run_cli(fake_engine, dummy_gguf, text)
    assert result.returncode != 0 and "Unsupported input format" in result.stderr


def test_missing_gguf_is_an_error(fake_engine: FakeEngineBinary, wav_factory: Callable[..., Path],
                                  tmp_path: Path) -> None:
    wav = wav_factory("x.wav", 1)
    result = run_cli(fake_engine, tmp_path / "missing.gguf", wav)
    assert result.returncode != 0 and "GGUF file not found" in result.stderr


def test_batch_size_must_be_positive(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                     wav_factory: Callable[..., Path]) -> None:
    result = run_cli(fake_engine, dummy_gguf, wav_factory("x.wav", 1), "--batch-size", "0")
    assert result.returncode != 0
    assert "batch-size" in result.stderr


def test_model_choices_and_default(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                   wav_factory: Callable[..., Path]) -> None:
    result = run_cli(fake_engine, dummy_gguf, wav_factory("x.wav", 1), "--model", "whisper")
    assert result.returncode != 0
    for name in ("cohere", "qwen", "parakeet", "nemotron"):
        assert name in result.stderr


def test_colliding_outputs_fail_before_any_work(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                                wav_factory: Callable[..., Path], tmp_path: Path) -> None:
    first = wav_factory("a.wav", 1)
    second = tmp_path / "a.mp3"
    second.write_bytes(b"x")
    result = run_cli(fake_engine, dummy_gguf, first, second)
    assert result.returncode == 2
    assert "would both write" in result.stderr
    assert fake_engine.calls() == [] and "Processing file" not in result.stdout
