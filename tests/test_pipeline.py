# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

from conftest import RecordingReporter
from transcribe_cli.engine import SegmentResult
from transcribe_cli.errors import EngineError, TranscriptionError
from transcribe_cli.models import MODELS
from transcribe_cli.pipeline import FileResult, Transcriber

TRUNCATED = "output truncated: decode hit the cap"


class FakeEngine:
    def __init__(self, respond: Callable[[Path], SegmentResult]) -> None:
        self.respond = respond
        self.batches: list[tuple[list[str], int, str | None]] = []

    def transcribe(self, wavs: Sequence[Path], *, batch_size: int, language: str | None,
                   workdir: Path, on_idle: Callable[[], None] | None = None) -> Iterator[SegmentResult]:
        self.batches.append(([wav.name for wav in wavs], batch_size, language))
        for wav in wavs:
            yield self.respond(wav)


def text(value: str) -> Callable[[Path], SegmentResult]:
    return lambda wav: SegmentResult(wav, value, None)


class Clock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def make(engine: FakeEngine, reporter: RecordingReporter, model: str = "cohere", batch_size: int = 1,
         clock: Callable[[], float] | None = None) -> Transcriber:
    return Transcriber(engine, MODELS[model], language="en", batch_size=batch_size,
                       reporter=reporter, clock=clock or Clock(0.0, 2.0))


def test_single_chunk_file(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    source = wav_factory("a.wav", 4)
    reporter, engine = RecordingReporter(), FakeEngine(text("Hello."))
    result = make(engine, reporter).transcribe_file(source, tmp_path / "out.txt")
    assert (tmp_path / "out.txt").read_text() == "Hello.\n"
    assert result == FileResult(source, tmp_path / "out.txt", 4.0, 2.0, 0)
    assert result.speedup == 2.0
    assert engine.batches == [(["a.wav"], 1, "en")]
    assert reporter.names() == ["file_started", "inference_started", "chunk_finished",
                                "inference_finished", "file_finished"]
    assert reporter.calls[0] == ("file_started", source, "16-bit PCM WAV, 16 kHz, mono")
    assert reporter.calls[1] == ("inference_started", "cohere", 1)


def test_conversion_is_reported(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    source = wav_factory("a.wav", 1, rate=44100)
    reporter = RecordingReporter()
    make(FakeEngine(text("x")), reporter, clock=Clock(0.0, 1.5, 3.0, 4.0)).transcribe_file(
        source, tmp_path / "out.txt")
    assert reporter.names()[:3] == ["file_started", "conversion_started", "conversion_finished"]
    assert reporter.calls[2] == ("conversion_finished", 1.5)


def test_long_file_is_chunked_and_stitched(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    source = wav_factory("long.wav", 65)
    texts = {"chunk-0000.wav": "Please review the guest room door.",
             "chunk-0001.wav": "the guest room door. Please find the exits.",
             "chunk-0002.wav": "Please find the exits. Then leave quickly."}
    reporter = RecordingReporter()
    engine = FakeEngine(lambda wav: SegmentResult(wav, texts[wav.name], None))
    make(engine, reporter).transcribe_file(source, tmp_path / "out.txt")
    assert (tmp_path / "out.txt").read_text() == (
        "Please review the guest room door. Please find the exits. Then leave quickly.\n")
    assert reporter.names().count("chunk_finished") == 3


def test_unaligned_overlap_is_reported_and_counted(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    source = wav_factory("long.wav", 40)
    reporter = RecordingReporter()
    result = make(FakeEngine(text("completely different")), reporter).transcribe_file(
        source, tmp_path / "out.txt")
    assert result.unaligned_overlaps == 1
    assert ("unaligned_overlaps", source, 1, False) in reporter.calls


@pytest.mark.parametrize(("model", "batch_size"), [("cohere", 4), ("qwen", 4), ("parakeet", 1), ("nemotron", 1)])
def test_batching_follows_the_model(tmp_path: Path, wav_factory: Callable[..., Path], model: str,
                                    batch_size: int) -> None:
    engine = FakeEngine(text("x"))
    make(engine, RecordingReporter(), model=model, batch_size=4).transcribe_file(
        wav_factory("a.wav", 1), tmp_path / "out.txt")
    assert engine.batches[0][1] == batch_size


def test_truncated_chunk_is_retried_alone(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    source = wav_factory("long.wav", 40)  # chunk-0000 is 30 s, chunk-0001 is 13 s

    def respond(wav: Path) -> SegmentResult:
        if wav.name == "chunk-0000.wav":
            return SegmentResult(wav, None, TRUNCATED)
        return SegmentResult(wav, {"chunk-0000-part-0.wav": "one two three four five six",
                                   "chunk-0000-part-1.wav": "four five six seven eight nine",
                                   "chunk-0001.wav": "seven eight nine ten eleven"}[wav.name], None)

    engine, reporter = FakeEngine(respond), RecordingReporter()
    make(engine, reporter, batch_size=3).transcribe_file(source, tmp_path / "out.txt")
    assert engine.batches == [(["chunk-0000.wav", "chunk-0001.wav"], 3, "en"),
                              (["chunk-0000-part-0.wav", "chunk-0000-part-1.wav"], 1, "en")]
    assert (tmp_path / "out.txt").read_text() == "one two three four five six seven eight nine ten eleven\n"
    assert reporter.names().count("truncation_retry") == 1
    # retries do not advance the chunk progress
    assert reporter.names().count("chunk_finished") == 2
    assert reporter.names().index("inference_finished") < reporter.names().index("truncation_retry")


def test_truncated_halves_recurse(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    source = wav_factory("a.wav", 20)
    sizes: dict[str, float] = {}

    def respond(wav: Path) -> SegmentResult:
        import wave
        with wave.open(str(wav), "rb") as reader:
            sizes[wav.name] = reader.getnframes() / 16000
        if sizes[wav.name] > 6:
            return SegmentResult(wav, None, TRUNCATED)
        return SegmentResult(wav, "ok", None)

    make(FakeEngine(respond), RecordingReporter(), model="parakeet").transcribe_file(
        source, tmp_path / "out.txt")
    assert "a-part-0-part-0.wav" in sizes and "a-part-1-part-1.wav" in sizes


def test_truncation_floor_is_an_error(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    source = wav_factory("a.wav", 3)
    with pytest.raises(TranscriptionError, match=r"shorter than two seconds: .*part-"):
        make(FakeEngine(lambda wav: SegmentResult(wav, None, TRUNCATED)), RecordingReporter()
             ).transcribe_file(source, tmp_path / "out.txt")
    assert not (tmp_path / "out.txt").exists()


def test_engine_error_row_is_raised(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    reporter = RecordingReporter()
    with pytest.raises(EngineError, match="unsupported language"):
        make(FakeEngine(lambda wav: SegmentResult(wav, None, "unsupported language")), reporter
             ).transcribe_file(wav_factory("a.wav", 1), tmp_path / "out.txt")
    assert "inference_finished" in reporter.names()


def test_unaligned_retry_overlap_is_reported(tmp_path: Path, wav_factory: Callable[..., Path]) -> None:
    def respond(wav: Path) -> SegmentResult:
        if wav.name == "a.wav":
            return SegmentResult(wav, None, TRUNCATED)
        return SegmentResult(wav, "alpha beta" if wav.name.endswith("0.wav") else "gamma delta", None)

    reporter = RecordingReporter()
    result = make(FakeEngine(respond), reporter).transcribe_file(wav_factory("a.wav", 10), tmp_path / "o.txt")
    assert result.unaligned_overlaps == 1
    assert [call[1:] for call in reporter.calls if call[0] == "unaligned_overlaps"] == [
        (tmp_path / "a.wav", 1, True)]
