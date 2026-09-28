# SPDX-License-Identifier: Apache-2.0
import io
import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from transcribe_cli.pipeline import FileResult
from transcribe_cli.reporting import ChunkProgress, ConsoleReporter


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_progress_resizes_with_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = iter([os.terminal_size((160, 24)), os.terminal_size((70, 24)), os.terminal_size((25, 24))])
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): next(sizes))
    terminal = Terminal()
    terminal.write("Running transcribe.cpp on 65 chunks...\n")
    with redirect_stderr(terminal):
        progress = ChunkProgress(65)
        progress.render()  # unchanged state does not consume a size... it re-reads the terminal
        progress.done = 32
        progress.advance()
        progress.close()
    output = terminal.getvalue()
    assert output.startswith("Running transcribe.cpp on 65 chunks...\n")
    frames = output.split("\r\033[K")[1:]
    assert [len(frame.rstrip("\n")) for frame in frames] == [158, 68, 23]
    assert all(frame.startswith("Chunks ") for frame in frames)
    assert "#" in frames[-1] and "-" in frames[-1]
    assert output.endswith("\n")


def test_progress_skips_redraw_when_nothing_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((80, 24)))
    terminal = Terminal()
    with redirect_stderr(terminal):
        progress = ChunkProgress(4)
        progress.render()
        progress.render()
    assert terminal.getvalue().count("\r") == 1


def test_progress_prints_lines_when_not_a_tty() -> None:
    stream = io.StringIO()
    with redirect_stderr(stream):
        progress = ChunkProgress(2)
        progress.render()
        progress.advance()
        progress.advance()
        progress.close()
    assert stream.getvalue().splitlines() == ["Chunks processed: 1/2", "Chunks processed: 2/2"]


def test_console_reporter_output() -> None:
    out, err = io.StringIO(), io.StringIO()
    reporter = ConsoleReporter(Path("model.gguf"))
    result = FileResult(Path("a.wav"), Path("a.txt"), 10.0, 2.0, 0)
    with redirect_stdout(out), redirect_stderr(err):
        reporter.file_started(Path("a.wav"), "AAC audio, 44.1 kHz, stereo")
        reporter.conversion_started(Path("a.wav"))
        reporter.conversion_finished(1.234)
        reporter.inference_started("cohere", 1)
        reporter.tick()
        reporter.chunk_finished()
        reporter.inference_finished()
        reporter.truncation_retry(Path("chunk.wav"))
        reporter.unaligned_overlaps(Path("a.wav"), 2, retry=False)
        reporter.unaligned_overlaps(Path("chunk.wav"), 1, retry=True)
        reporter.file_finished(result)
    assert out.getvalue().splitlines() == [
        "Processing file: a.wav",
        "Input format: AAC audio, 44.1 kHz, stereo",
        "Converting to 16-bit PCM WAV, 16 kHz, mono: a.wav",
        "Conversion completed in 1.23s",
        "Running transcribe.cpp with cohere on 1 audio chunk(s)...",
        "Wrote transcript to a.txt with model.gguf using Vulkan GPU",
        "Metrics for a.wav: 10.00s audio; batch completed in 2.00s (5.00x)",
    ]
    assert err.getvalue().splitlines() == [
        "Chunks processed: 1/1",
        "Retrying truncated audio in shorter pieces: chunk.wav",
        "Warning: could not align 2 chunk overlap(s) for a.wav; check for repeated words",
        "Warning: could not align 1 retry overlap(s) for chunk.wav",
    ]


def test_speedup_with_zero_elapsed_is_infinite() -> None:
    assert FileResult(Path("a"), Path("b"), 1.0, 0.0, 0).speedup == float("inf")
