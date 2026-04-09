from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
COMMON_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".ogg", ".wav", ".flac", ".aac", ".webm"}
NO_SPACE_LANGUAGES = frozenset({"ja", "zh"})
DEFAULT_CUDA_BATCH_SIZE = 32
DEFAULT_CPU_BATCH_SIZE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio file with Cohere Transcribe and write the result to INPUT.txt."
        )
    )
    parser.add_argument("input_file", type=Path, help="Path to the source audio file.")
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
            "Defaults to 32 on CUDA and 1 on CPU."
        ),
    )
    return parser.parse_args()


def validate_input_file(input_file: Path) -> None:
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_file.suffix and input_file.suffix.lower() not in COMMON_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(COMMON_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported input format '{input_file.suffix}'. Expected one of: {supported}"
        )


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
    model: CohereAsrForConditionalGeneration,
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
    input_file = args.input_file.expanduser().resolve()
    validate_input_file(input_file)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
    )
    model.eval()

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = DEFAULT_CUDA_BATCH_SIZE if torch.cuda.is_available() else DEFAULT_CPU_BATCH_SIZE
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    audio = load_audio(str(input_file), sampling_rate=16000)
    text = batched_transcribe(
        processor=processor,
        model=model,
        audio=audio,
        language=args.language,
        batch_size=batch_size,
    )

    output_file = input_file.with_suffix(".txt")
    output_file.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Wrote transcript to {output_file}")


if __name__ == "__main__":
    main()
