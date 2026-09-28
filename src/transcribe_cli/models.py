# SPDX-License-Identifier: Apache-2.0
"""The model registry: every model-specific fact lives here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from huggingface_hub import hf_hub_download

from transcribe_cli.errors import PrerequisiteError, UsageError


class LanguageSupport(Enum):
    HINT = "hint"  # any hint passed through (Cohere, Nemotron)
    ENGLISH_ONLY = "english"  # only "en" accepted, nothing passed (Parakeet v2)
    AUTO_ONLY = "auto"  # any hint rejected (Qwen3-ASR port)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    filename: str
    chunk_seconds: int  # practical GPU-memory bound, not the model's limit
    language: LanguageSupport
    default_language: str | None
    auto_means_unset: bool = False  # Nemotron: "auto" -> pass no -l
    supports_batching: bool = True
    language_error: str = ""

    def resolve_language(self, requested: str | None) -> str | None:
        if self.language is LanguageSupport.HINT:
            language = requested or self.default_language
            return None if self.auto_means_unset and language == "auto" else language
        if requested is not None and (self.language is LanguageSupport.AUTO_ONLY or requested != "en"):
            raise UsageError(self.language_error)
        return None


MODELS: Mapping[str, ModelSpec] = {spec.name: spec for spec in (
    ModelSpec("cohere", "handy-computer/cohere-transcribe-03-2026-gguf",
              "cohere-transcribe-03-2026-Q8_0.gguf", 30, LanguageSupport.HINT, "en"),
    ModelSpec("qwen", "handy-computer/Qwen3-ASR-1.7B-gguf", "Qwen3-ASR-1.7B-Q4_K_M.gguf", 60,
              LanguageSupport.AUTO_ONLY, None,
              language_error="transcribe.cpp Qwen3-ASR currently supports automatic language detection only."),
    # supports_batching=False: the transcribe.cpp 0.2.3 Parakeet/Nemotron batch path asserts
    # inside ggml. Re-test both when the transcribe.cpp `rev` in flake.nix is bumped.
    ModelSpec("parakeet", "handy-computer/parakeet-tdt-0.6b-v2-gguf",
              "parakeet-tdt-0.6b-v2-Q4_K_M.gguf", 300, LanguageSupport.ENGLISH_ONLY, None,
              supports_batching=False,
              language_error="Parakeet v2 only supports English (--language en)."),
    ModelSpec("nemotron", "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf",
              "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf", 60, LanguageSupport.HINT, None,
              auto_means_unset=True, supports_batching=False),
)}
DEFAULT_MODEL = "cohere"


def language_help() -> str:
    defaults = ", ".join(f"{spec.default_language or 'auto'} for {spec.name.capitalize()}"
                         for spec in MODELS.values() if spec.language is LanguageSupport.HINT)
    return f"Language hint (default: {defaults})."


def resolve_model_file(spec: ModelSpec, override: str | None) -> Path:
    repo = spec.repo
    if override:
        local = Path(override).expanduser()
        if local.is_file():
            return local.resolve()
        if override.endswith(".gguf"):
            raise UsageError(f"GGUF file not found: {override}")
        repo = override
    try:
        return Path(hf_hub_download(repo_id=repo, filename=spec.filename))
    except Exception as error:  # huggingface_hub raises a wide family of network/auth errors
        raise PrerequisiteError(f"could not download {spec.filename} from {repo}: {error}") from error
