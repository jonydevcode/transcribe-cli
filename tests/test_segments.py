# SPDX-License-Identifier: Apache-2.0
import wave
from pathlib import Path

import pytest

from conftest import write_wav
from transcribe_cli.errors import TranscriptionError
from transcribe_cli.segments import OVERLAP_FRAMES, SAMPLE_RATE, Span, plan_chunks, plan_halves, write_span


def test_short_total_is_one_span() -> None:
    assert plan_chunks(10, 30, 3) == [Span(0, 10)]
    assert plan_chunks(30, 30, 3) == [Span(0, 30)]


def test_chunks_overlap_and_cover_everything() -> None:
    assert plan_chunks(65, 30, 3) == [Span(0, 30), Span(27, 57), Span(54, 65)]


def test_last_chunk_shorter_than_overlap_is_still_emitted() -> None:
    spans = plan_chunks(58, 30, 3)
    assert spans == [Span(0, 30), Span(27, 57), Span(54, 58)]


def test_overlap_is_capped_at_half_a_chunk() -> None:
    assert plan_chunks(10, 4, 100) == [Span(0, 4), Span(2, 6), Span(4, 8), Span(6, 10)]


@pytest.mark.parametrize(("total", "chunk", "overlap"), [(1000, 100, 10), (999, 7, 3), (101, 100, 50)])
def test_adjacent_spans_share_the_overlap(total: int, chunk: int, overlap: int) -> None:
    spans = plan_chunks(total, chunk, overlap)
    assert spans[0].start == 0 and spans[-1].end == total
    for before, after in zip(spans, spans[1:], strict=False):
        assert before.end - after.start == min(overlap, chunk // 2)


def test_halves_overlap() -> None:
    first, second = plan_halves(20 * SAMPLE_RATE, OVERLAP_FRAMES, SAMPLE_RATE)
    total = 20 * SAMPLE_RATE
    assert first.start == 0 and second.end == total
    assert first.end - second.start == OVERLAP_FRAMES - OVERLAP_FRAMES % 2


def test_halves_overlap_is_capped_at_quarter() -> None:
    first, second = plan_halves(4 * SAMPLE_RATE, OVERLAP_FRAMES, SAMPLE_RATE)
    assert first.end - second.start == SAMPLE_RATE


def test_halves_floor_is_two_seconds() -> None:
    plan_halves(2 * SAMPLE_RATE, OVERLAP_FRAMES, SAMPLE_RATE)
    with pytest.raises(TranscriptionError, match="shorter than two seconds"):
        plan_halves(2 * SAMPLE_RATE - 1, OVERLAP_FRAMES, SAMPLE_RATE)


def test_write_span_round_trip(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "source.wav", 2)
    dest = write_span(source, Span(100, 1100), tmp_path / "part.wav")
    with wave.open(str(dest), "rb") as reader:
        assert reader.getnframes() == 1000
        assert (reader.getnchannels(), reader.getframerate(), reader.getsampwidth()) == (1, SAMPLE_RATE, 2)
