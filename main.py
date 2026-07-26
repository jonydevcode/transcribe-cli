# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, PreTrainedModel

DEFAULT_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
COMMON_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".ogg", ".wav", ".flac", ".aac", ".webm"}
NO_SPACE_LANGUAGES = frozenset({"ja", "zh"})
DEFAULT_GPU_BATCH_SIZE = 32
DEFAULT_CPU_BATCH_SIZE = 1


class RuntimeConfig(NamedTuple):
    device: torch.device
    dtype: torch.dtype
    default_batch_size: int


def resolve_runtime_config() -> RuntimeConfig:
    if torch.cuda.is_available():
        return RuntimeConfig(
            device=torch.device("cuda"),
            dtype=torch.float16,
            default_batch_size=DEFAULT_GPU_BATCH_SIZE,
        )

    return RuntimeConfig(
        device=torch.device("cpu"),
        dtype=torch.float32,
        default_batch_size=DEFAULT_CPU_BATCH_SIZE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one or more media files and write each result to INPUT.txt."
        )
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="One or more source media paths or glob patterns.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face ASR model ID. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="ISO 639-1 language code required by the model. Default: en",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Inference batch size for chunked long-form transcription. "
            "Defaults to 32 on ROCm GPU and 1 on CPU."
        ),
    )
    return parser.parse_args()


def expand_input_paths(input_paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_path in input_paths:
        matches = [Path(match) for match in glob.glob(raw_path)]
        if not matches:
            matches = [Path(raw_path)]

        for match in matches:
            resolved_path = match.expanduser().resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            resolved.append(resolved_path)

    return resolved


def validate_input_file(input_file: Path) -> None:
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_file.suffix and input_file.suffix.lower() not in COMMON_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(COMMON_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported input format '{input_file.suffix}'. Expected one of: {supported}"
        )


def load_audio_with_ffmpeg(input_file: Path, sampling_rate: int) -> np.ndarray:
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(input_file),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sampling_rate),
            "-",
        ],
        check=True,
        capture_output=True,
    )
    audio = np.frombuffer(process.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Decoded audio was empty: {input_file}")
    return audio


def can_decode_with_soundfile(input_file: Path) -> bool:
    try:
        sf.info(str(input_file))
    except sf.LibsndfileError:
        return False
    return True


def validate_decoder_dependencies(input_files: list[Path]) -> None:
    files_requiring_ffmpeg = [
        input_file for input_file in input_files if not can_decode_with_soundfile(input_file)
    ]
    if not files_requiring_ffmpeg:
        return

    if shutil.which("ffmpeg") is not None:
        return

    affected_files = "\n".join(f"- {input_file}" for input_file in files_requiring_ffmpeg)
    raise RuntimeError(
        "ffmpeg is required to decode one or more input files, but it was not found on PATH:\n"
        f"{affected_files}"
    )


def load_audio_file(input_file: Path, sampling_rate: int = 16000) -> np.ndarray:
    try:
        audio, source_rate = sf.read(str(input_file), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        return load_audio_with_ffmpeg(input_file, sampling_rate)

    if audio.size == 0:
        raise ValueError(f"Decoded audio was empty: {input_file}")

    mono_audio = audio.mean(axis=1)
    if source_rate == sampling_rate:
        return mono_audio

    return load_audio_with_ffmpeg(input_file, sampling_rate)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def trim_generated_tokens(
    outputs: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> list[list[int]]:
    trimmed: list[list[int]] = []
    for row_idx in range(outputs.shape[0]):
        token_ids = outputs[row_idx].tolist()
        prompt_ids = decoder_input_ids[row_idx].tolist()
        if pad_token_id is None:
            prompt_len = len(prompt_ids)
        else:
            prompt_len = sum(token_id != pad_token_id for token_id in prompt_ids)
        prompt_ids = prompt_ids[:prompt_len]

        starts_with_prompt = (
            prompt_len > 0
            and len(token_ids) >= prompt_len
            and token_ids[:prompt_len] == prompt_ids
        )
        if starts_with_prompt:
            token_ids = token_ids[prompt_len:]

        if eos_token_id is not None and eos_token_id in token_ids:
            token_ids = token_ids[:token_ids.index(eos_token_id)]

        trimmed.append(token_ids)
    return trimmed


def reassemble_chunk_texts(
    texts: list[str],
    audio_chunk_index: list[tuple[int, int | None]],
    language: str,
) -> list[str]:
    separator = "" if language in NO_SPACE_LANGUAGES else " "
    max_sample_idx = max(sample_idx for sample_idx, _ in audio_chunk_index)
    outputs = [""] * (max_sample_idx + 1)
    chunked: dict[int, list[tuple[int, str]]] = {}

    for (sample_idx, chunk_idx), text in zip(audio_chunk_index, texts):
        if chunk_idx is None:
            outputs[sample_idx] = text
            continue
        chunked.setdefault(sample_idx, []).append((chunk_idx, text))

    for sample_idx, chunk_items in chunked.items():
        chunk_items.sort(key=lambda item: item[0])
        non_empty = [text for _, text in chunk_items if text and text.strip()]
        if not non_empty:
            outputs[sample_idx] = ""
            continue
        parts = [non_empty[0].rstrip()] + [text.strip() for text in non_empty[1:]]
        outputs[sample_idx] = separator.join(parts)

    return outputs


def batched_transcribe(
    *,
    processor: AutoProcessor,
    model: PreTrainedModel,
    audio,
    language: str,
    batch_size: int,
) -> str:
    inputs = processor(
        audio=audio,
        sampling_rate=16000,
        return_tensors="pt",
        language=language,
    )
    audio_chunk_index = inputs.pop("audio_chunk_index", None)
    pad_token_id = processor.tokenizer.pad_token_id
    eos_token_id = processor.tokenizer.eos_token_id

    texts: list[str] = []
    num_chunks = inputs["input_features"].shape[0]
    for start in range(0, num_chunks, batch_size):
        end = min(start + batch_size, num_chunks)
        batch_inputs = {
            key: value[start:end]
            for key, value in inputs.items()
        }
        batch_inputs = move_batch_to_device(batch_inputs, model.device, model.dtype)
        with torch.inference_mode():
            outputs = model.generate(**batch_inputs, max_new_tokens=256)
        token_ids = trim_generated_tokens(
            outputs=outputs.cpu(),
            decoder_input_ids=batch_inputs["decoder_input_ids"].cpu(),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        texts.extend(processor.batch_decode(token_ids, skip_special_tokens=True))

    if audio_chunk_index is None:
        return texts[0].strip()

    merged = reassemble_chunk_texts(texts, audio_chunk_index, language)
    return merged[0].strip()


def main() -> None:
    args = parse_args()
    input_files = expand_input_paths(args.input_paths)
    if not input_files:
        raise ValueError("No input files were provided")
    for input_file in input_files:
        validate_input_file(input_file)
    validate_decoder_dependencies(input_files)
    runtime = resolve_runtime_config()

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        dtype=runtime.dtype,
    ).to(runtime.device)
    model.eval()

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = runtime.default_batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    for input_file in input_files:
        audio = load_audio_file(input_file, sampling_rate=16000)
        audio_duration_seconds = len(audio) / 16000
        transcription_started = time.perf_counter()
        text = batched_transcribe(
            processor=processor,
            model=model,
            audio=audio,
            language=args.language,
            batch_size=batch_size,
        )
        transcription_seconds = time.perf_counter() - transcription_started
        throughput = audio_duration_seconds / transcription_seconds

        output_file = input_file.with_suffix(".txt")
        output_file.write_text(text.strip() + "\n", encoding="utf-8")
        print(
            f"Wrote transcript to {output_file} with {args.model_id} "
            f"using {runtime.device.type} ({runtime.dtype})"
        )
        print(
            f"Metrics for {input_file}: {audio_duration_seconds:.2f}s audio in "
            f"{transcription_seconds:.2f}s ({throughput:.2f} audio sec/sec)"
        )


if __name__ == "__main__":
    main()
