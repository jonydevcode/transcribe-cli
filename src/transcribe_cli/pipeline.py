# SPDX-License-Identifier: Apache-2.0
"""Per-file orchestration: normalise, chunk, transcribe, retry truncations, stitch."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from transcribe_cli import media
from transcribe_cli.engine import Engine, SegmentResult
from transcribe_cli.errors import EngineError, OutputError, TranscriptionError
from transcribe_cli.models import ModelSpec
from transcribe_cli.reporting import Reporter
from transcribe_cli.segments import (
    MIN_SPLIT_FRAMES,
    OVERLAP_FRAMES,
    SAMPLE_RATE,
    plan_chunks,
    plan_halves,
    write_span,
)
from transcribe_cli.stitch import stitch_chunks


@dataclass(frozen=True)
class FileResult:
    source: Path
    output: Path
    audio_seconds: float
    inference_seconds: float
    unaligned_overlaps: int

    @property
    def speedup(self) -> float:
        return self.audio_seconds / self.inference_seconds if self.inference_seconds > 0 else float("inf")


class Transcriber:
    def __init__(self, engine: Engine, model: ModelSpec, *, language: str | None, batch_size: int,
                 reporter: Reporter, clock: Callable[[], float] = time.perf_counter) -> None:
        self._engine = engine
        self._model = model
        self._language = language
        self._batch_size = batch_size if model.supports_batching else 1
        self._reporter = reporter
        self._clock = clock

    def transcribe_file(self, source: Path, output: Path) -> FileResult:
        reporter = self._reporter
        with tempfile.TemporaryDirectory(prefix="transcribe-cli-") as temporary:
            workdir = Path(temporary)
            reporter.file_started(source, media.describe(source))
            wav = media.normalize(source, workdir, reporter, self._clock)
            total = media.read_frame_count(wav)
            spans = plan_chunks(total, self._model.chunk_seconds * SAMPLE_RATE, OVERLAP_FRAMES)
            chunks = [wav] if len(spans) == 1 else [
                write_span(wav, span, workdir / f"chunk-{index:04d}.wav") for index, span in enumerate(spans)]
            reporter.inference_started(self._model.name, len(chunks))
            started = self._clock()
            try:
                results = self._run(chunks, self._batch_size, workdir, advance=True)
            finally:
                reporter.inference_finished()
            texts, retry_unaligned = self._texts(results, workdir)
            transcript, unaligned = stitch_chunks(texts)
            elapsed = self._clock() - started
            try:
                output.write_text(transcript + "\n", encoding="utf-8")
            except OSError as error:
                raise OutputError(f"Cannot write transcript {output}: {error.strerror or error}") from error
            if unaligned:
                reporter.unaligned_overlaps(source, unaligned, retry=False)
            result = FileResult(source, output, total / SAMPLE_RATE, elapsed, unaligned + retry_unaligned)
            reporter.file_finished(result)
            return result

    def _run(self, wavs: Sequence[Path], batch_size: int, workdir: Path, *,
             advance: bool) -> list[SegmentResult]:
        by_path: dict[Path, SegmentResult] = {}
        for result in self._engine.transcribe(wavs, batch_size=batch_size, language=self._language,
                                              workdir=workdir, on_idle=self._reporter.tick):
            by_path[result.path] = result
            if advance:
                self._reporter.chunk_finished()
        return [by_path[wav] for wav in wavs]

    def _texts(self, results: Sequence[SegmentResult], workdir: Path) -> tuple[list[str], int]:
        texts: list[str] = []
        unaligned = 0
        for result in results:
            if result.truncated:
                text, nested = self._resolve_truncated(result.path, workdir)
                unaligned += nested
            elif result.error is not None:
                raise EngineError(f"transcribe.cpp failed for {result.path}: {result.error}")
            else:
                text = result.text or ""
            texts.append(text)
        return texts, unaligned

    def _resolve_truncated(self, wav: Path, workdir: Path) -> tuple[str, int]:
        """Re-run one truncated segment as two overlapping halves, recursing on truncated halves."""
        try:
            halves = plan_halves(media.read_frame_count(wav), OVERLAP_FRAMES, MIN_SPLIT_FRAMES)
        except TranscriptionError as error:
            raise TranscriptionError(f"{error}: {wav}") from error
        parts = [write_span(wav, span, workdir / f"{wav.stem}-part-{index}.wav")
                 for index, span in enumerate(halves)]
        self._reporter.truncation_retry(wav)
        texts, nested = self._texts(self._run(parts, 1, workdir, advance=False), workdir)
        text, unaligned = stitch_chunks(texts)
        if unaligned:
            self._reporter.unaligned_overlaps(wav, unaligned, retry=True)
        return text, nested + unaligned
