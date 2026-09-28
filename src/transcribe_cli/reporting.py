# SPDX-License-Identifier: Apache-2.0
"""Everything the user sees while a run is in progress."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from transcribe_cli.pipeline import FileResult


class Reporter(Protocol):
    def file_started(self, path: Path, audio_format: str) -> None: ...
    def conversion_started(self, path: Path) -> None: ...
    def conversion_finished(self, elapsed: float) -> None: ...
    def inference_started(self, model: str, chunks: int) -> None: ...
    def tick(self) -> None: ...
    def chunk_finished(self) -> None: ...
    def inference_finished(self) -> None: ...
    def truncation_retry(self, wav: Path) -> None: ...
    def unaligned_overlaps(self, path: Path, count: int, *, retry: bool) -> None: ...
    def file_finished(self, result: FileResult) -> None: ...


class ChunkProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.tty = sys.stderr.isatty()
        self.width = 0
        self.rendered = -1
        self.render()

    def render(self) -> None:
        if not self.tty:
            if self.done:
                print(f"Chunks processed: {self.done}/{self.total}", file=sys.stderr, flush=True)
            return
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        if width == self.width and self.done == self.rendered:
            return
        self.width = width
        self.rendered = self.done
        label = f"Chunks {self.done}/{self.total} "
        space = max(0, width - len(label) - 4)
        if space >= 4:
            filled = space * self.done // self.total
            line = f"{label}[{'#' * filled}{'-' * (space - filled)}]"
        else:
            line = label.rstrip()[:max(1, width - 1)]
        sys.stderr.write("\r\033[K" + line)
        sys.stderr.flush()

    def advance(self) -> None:
        self.done += 1
        self.render()

    def close(self) -> None:
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


class ConsoleReporter:
    """Status lines on stdout; progress and warnings on stderr."""

    def __init__(self, model_file: Path | None = None) -> None:
        self._model_file = model_file
        self._progress: ChunkProgress | None = None

    def file_started(self, path: Path, audio_format: str) -> None:
        print(f"Processing file: {path}", flush=True)
        print(f"Input format: {audio_format}", flush=True)

    def conversion_started(self, path: Path) -> None:
        print(f"Converting to 16-bit PCM WAV, 16 kHz, mono: {path}", flush=True)

    def conversion_finished(self, elapsed: float) -> None:
        print(f"Conversion completed in {elapsed:.2f}s", flush=True)

    def inference_started(self, model: str, chunks: int) -> None:
        print(f"Running transcribe.cpp with {model} on {chunks} audio chunk(s)...", flush=True)
        self._progress = ChunkProgress(chunks)

    def tick(self) -> None:
        if self._progress:
            self._progress.render()

    def chunk_finished(self) -> None:
        if self._progress:
            self._progress.advance()

    def inference_finished(self) -> None:
        if self._progress:
            self._progress.close()
            self._progress = None

    def truncation_retry(self, wav: Path) -> None:
        print(f"Retrying truncated audio in shorter pieces: {wav}", file=sys.stderr, flush=True)

    def unaligned_overlaps(self, path: Path, count: int, *, retry: bool) -> None:
        if retry:
            message = f"Warning: could not align {count} retry overlap(s) for {path}"
        else:
            message = f"Warning: could not align {count} chunk overlap(s) for {path}; check for repeated words"
        print(message, file=sys.stderr, flush=True)

    def file_finished(self, result: FileResult) -> None:
        suffix = f" with {self._model_file} using Vulkan GPU" if self._model_file else ""
        print(f"Wrote transcript to {result.output}{suffix}")
        print(f"Metrics for {result.source}: {result.audio_seconds:.2f}s audio; "
              f"batch completed in {result.inference_seconds:.2f}s ({result.speedup:.2f}x)", flush=True)
