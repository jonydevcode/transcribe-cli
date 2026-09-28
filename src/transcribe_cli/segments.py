# SPDX-License-Identifier: Apache-2.0
"""Pure span planning over frame counts, plus WAV slicing.

Invariant: adjacent spans share at least the overlap `stitch` needs to find a
phrase on both sides of the join.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from transcribe_cli.errors import TranscriptionError

SAMPLE_RATE = 16000
OVERLAP_SECONDS = 3
MIN_SPLIT_FRAMES = SAMPLE_RATE  # each half must be at least a second, i.e. two seconds in total
OVERLAP_FRAMES = OVERLAP_SECONDS * SAMPLE_RATE


@dataclass(frozen=True)
class Span:
    start: int  # frames
    end: int

    @property
    def frames(self) -> int:
        return self.end - self.start


def plan_chunks(total: int, chunk: int, overlap: int) -> list[Span]:
    """Split `total` frames into spans of at most `chunk`, adjacent ones sharing `overlap`.

    The overlap is capped at half a chunk so that every step makes progress.
    """
    if total <= chunk:
        return [Span(0, total)]
    overlap = min(overlap, chunk // 2)
    spans: list[Span] = []
    start = 0
    while True:
        end = min(start + chunk, total)
        spans.append(Span(start, end))
        if end == total:
            return spans
        start = end - overlap


def plan_halves(total: int, overlap: int, min_frames: int) -> tuple[Span, Span]:
    midpoint = total // 2
    if midpoint < min_frames:
        raise TranscriptionError("transcribe.cpp still truncated audio shorter than two seconds")
    overlap = min(overlap, total // 4)
    return Span(0, midpoint + overlap // 2), Span(midpoint - overlap // 2, total)


def write_span(source: Path, span: Span, dest: Path) -> Path:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        reader.setpos(span.start)
        data = reader.readframes(span.frames)
    with wave.open(str(dest), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(data)
    return dest
