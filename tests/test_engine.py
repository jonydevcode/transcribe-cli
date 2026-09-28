# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import FakeEngineBinary
from transcribe_cli.engine import SegmentResult, TranscribeCpp, require_vulkan_gpu
from transcribe_cli.errors import EngineError, PrerequisiteError


def engine_for(fake: FakeEngineBinary, model: Path) -> TranscribeCpp:
    return TranscribeCpp(fake.path, model)


def run(engine: TranscribeCpp, wavs: list[Path], workdir: Path, batch_size: int = 1,
        language: str | None = None, on_idle: Callable[[], None] | None = None) -> list[SegmentResult]:
    return list(engine.transcribe(wavs, batch_size=batch_size, language=language,
                                  workdir=workdir, on_idle=on_idle))


def test_requires_vulkan_gpu(fake_engine: FakeEngineBinary) -> None:
    require_vulkan_gpu(fake_engine.path)
    fake_engine.configure(devices="Vulkan0: kind=vulkan type=dgpu")
    require_vulkan_gpu(fake_engine.path)
    fake_engine.configure(devices="kind=cpu")
    with pytest.raises(PrerequisiteError, match="CPU fallback is disabled"):
        require_vulkan_gpu(fake_engine.path)
    fake_engine.configure(devices="kind=vulkan type=cpu")
    with pytest.raises(PrerequisiteError):
        TranscribeCpp(fake_engine.path, Path("m.gguf")).require_vulkan_gpu()


def test_unrunnable_binary(tmp_path: Path) -> None:
    with pytest.raises(PrerequisiteError, match="cannot run transcribe.cpp"):
        require_vulkan_gpu(tmp_path / "missing")


def test_command_line(fake_engine: FakeEngineBinary, dummy_gguf: Path, wav_factory: Callable[..., Path],
                      tmp_path: Path) -> None:
    wav = wav_factory("a.wav", 1)
    run(engine_for(fake_engine, dummy_gguf), [wav], tmp_path, batch_size=1, language="fr")
    (call,) = fake_engine.calls()
    batch = tmp_path / "batch.txt"
    assert call["argv"] == ["-m", str(dummy_gguf), "--backend", "vulkan", "--batch", str(batch),
                            "--batch-jsonl", "--batch-size", "1", "--timestamps", "none", "-l", "fr"]
    assert call["paths"] == [str(wav)]


def test_results_stream_with_text_and_errors(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                             wav_factory: Callable[..., Path], tmp_path: Path) -> None:
    fake_engine.configure(texts={"a.wav": "Hello."}, errors={"b.wav": "unsupported language"})
    first, second = wav_factory("a.wav", 1), wav_factory("b.wav", 1)
    results = run(engine_for(fake_engine, dummy_gguf), [first, second], tmp_path)
    assert results == [SegmentResult(first, "Hello.", None),
                       SegmentResult(second, None, "unsupported language")]
    assert not results[1].truncated


def test_truncation_is_flagged(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                               wav_factory: Callable[..., Path], tmp_path: Path) -> None:
    fake_engine.configure(truncate_longer_than=0.5)
    (result,) = run(engine_for(fake_engine, dummy_gguf), [wav_factory("a.wav", 1)], tmp_path)
    assert result.truncated and result.text is None


def test_batch_size_falls_back_to_one_for_mixed_lengths(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                                        wav_factory: Callable[..., Path],
                                                        tmp_path: Path) -> None:
    engine = engine_for(fake_engine, dummy_gguf)
    run(engine, [wav_factory("a.wav", 1), wav_factory("b.wav", 2)], tmp_path, batch_size=4)
    run(engine, [wav_factory("c.wav", 1), wav_factory("d.wav", 1)], tmp_path, batch_size=4)
    sizes = [call["argv"][call["argv"].index("--batch-size") + 1] for call in fake_engine.calls()]
    assert sizes == ["1", "4"]


def test_junk_and_partial_lines_are_ignored(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                            wav_factory: Callable[..., Path], tmp_path: Path) -> None:
    fake_engine.configure(junk_line=True, partial_last_line=True)
    wav = wav_factory("a.wav", 1)
    assert run(engine_for(fake_engine, dummy_gguf), [wav], tmp_path) == [SegmentResult(wav, "ok", None)]


def test_nonzero_exit_raises_with_stderr_tail(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                                              wav_factory: Callable[..., Path], tmp_path: Path) -> None:
    fake_engine.configure(exit=3)
    with pytest.raises(EngineError, match="progress a.wav"):
        run(engine_for(fake_engine, dummy_gguf), [wav_factory("a.wav", 1)], tmp_path)


def test_missing_result_raises(fake_engine: FakeEngineBinary, dummy_gguf: Path,
                               wav_factory: Callable[..., Path], tmp_path: Path) -> None:
    binary = fake_engine.path
    binary.write_text(f"#!{__import__('sys').executable}\nimport sys\n"
                      "if sys.argv[1:] == ['--list-devices']: sys.exit(0)\nprint('{\"type\": \"batch_header\"}')\n")
    with pytest.raises(EngineError, match="one transcript for each"):
        run(TranscribeCpp(binary, dummy_gguf), [wav_factory("a.wav", 1)], tmp_path)


def test_first_row_per_file_wins_and_unknown_files_are_ignored(tmp_path: Path, dummy_gguf: Path,
                                                               wav_factory: Callable[..., Path]) -> None:
    wav = wav_factory("a.wav", 1)
    binary = tmp_path / "fake"
    binary.write_text(
        f"#!{__import__('sys').executable}\nimport json, sys\n"
        f"print(json.dumps({{'file': 'elsewhere.wav', 'text': 'x'}}))\n"
        f"print(json.dumps({{'file': {str(wav)!r}, 'text': 'first'}}))\n"
        f"print(json.dumps({{'file': {str(wav)!r}, 'text': 'second'}}))\n")
    binary.chmod(0o755)
    assert run(TranscribeCpp(binary, dummy_gguf), [wav], tmp_path) == [SegmentResult(wav, "first", None)]


def test_on_idle_is_called_while_engine_is_silent(tmp_path: Path, dummy_gguf: Path,
                                                  wav_factory: Callable[..., Path]) -> None:
    wav = wav_factory("a.wav", 1)
    binary = tmp_path / "slow"
    binary.write_text(
        f"#!{__import__('sys').executable}\nimport json, time\ntime.sleep(0.6)\n"
        f"print(json.dumps({{'file': {str(wav)!r}, 'text': 'late'}}))\n")
    binary.chmod(0o755)
    idle: list[int] = []
    run(TranscribeCpp(binary, dummy_gguf), [wav], tmp_path, on_idle=lambda: idle.append(1))
    assert len(idle) >= 2
