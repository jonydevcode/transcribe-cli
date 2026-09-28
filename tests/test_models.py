# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from transcribe_cli import models
from transcribe_cli.errors import PrerequisiteError, UsageError
from transcribe_cli.models import DEFAULT_MODEL, MODELS, resolve_model_file


def test_registry() -> None:
    assert list(MODELS) == ["cohere", "qwen", "parakeet", "nemotron"]
    assert DEFAULT_MODEL == "cohere"
    assert MODELS["qwen"].repo == "handy-computer/Qwen3-ASR-1.7B-gguf"
    assert {name: spec.chunk_seconds for name, spec in MODELS.items()} == {
        "cohere": 30, "qwen": 60, "parakeet": 300, "nemotron": 60}
    assert {name for name, spec in MODELS.items() if not spec.supports_batching} == {"parakeet", "nemotron"}


@pytest.mark.parametrize(("model", "requested", "expected"), [
    ("cohere", None, "en"), ("cohere", "auto", "auto"), ("cohere", "fr", "fr"),
    ("nemotron", None, None), ("nemotron", "auto", None), ("nemotron", "de", "de"),
    ("parakeet", None, None), ("parakeet", "en", None),
    ("qwen", None, None),
])
def test_resolve_language(model: str, requested: str | None, expected: str | None) -> None:
    assert MODELS[model].resolve_language(requested) == expected


def test_rejected_languages() -> None:
    with pytest.raises(UsageError, match="automatic language"):
        MODELS["qwen"].resolve_language("zh")
    with pytest.raises(UsageError, match="only supports English"):
        MODELS["parakeet"].resolve_language("fr")


def test_language_help_comes_from_specs() -> None:
    assert models.language_help() == "Language hint (default: en for Cohere, auto for Nemotron)."


def test_local_gguf_override(dummy_gguf: Path) -> None:
    assert resolve_model_file(MODELS["cohere"], str(dummy_gguf)) == dummy_gguf.resolve()


def test_missing_local_gguf(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="GGUF file not found"):
        resolve_model_file(MODELS["cohere"], str(tmp_path / "nope.gguf"))


def test_repo_override_and_default_download(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return "/cache/model.gguf"

    monkeypatch.setattr(models, "hf_hub_download", fake_download)
    assert resolve_model_file(MODELS["qwen"], None) == Path("/cache/model.gguf")
    resolve_model_file(MODELS["qwen"], "someone/custom-gguf")
    assert calls == [("handy-computer/Qwen3-ASR-1.7B-gguf", "Qwen3-ASR-1.7B-Q4_K_M.gguf"),
                     ("someone/custom-gguf", "Qwen3-ASR-1.7B-Q4_K_M.gguf")]


def test_download_failure_is_a_prerequisite_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_: str) -> str:
        raise OSError("offline")

    monkeypatch.setattr(models, "hf_hub_download", boom)
    with pytest.raises(PrerequisiteError, match="offline"):
        resolve_model_file(MODELS["cohere"], None)
