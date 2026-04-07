from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
COMMON_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".ogg", ".wav", ".flac", ".aac", ".webm"}


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
    return parser.parse_args()


def validate_input_file(input_file: Path) -> None:
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if input_file.suffix and input_file.suffix.lower() not in COMMON_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(COMMON_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported input format '{input_file.suffix}'. Expected one of: {supported}"
        )


def main() -> None:
    args = parse_args()
    input_file = args.input_file.expanduser().resolve()
    validate_input_file(input_file)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    texts = model.transcribe(
        processor=processor,
        audio_files=[str(input_file)],
        language=args.language,
    )
    text = texts[0]

    output_file = input_file.with_suffix(".txt")
    output_file.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Wrote transcript to {output_file}")


if __name__ == "__main__":
    main()
