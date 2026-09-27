# SPDX-License-Identifier: Apache-2.0
"""GPU-only transcribe.cpp frontend."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from huggingface_hub import hf_hub_download

TRANSCRIBE_CPP_COMMIT = "c83df3f229accd6f769cb91b638cd365f3624c9c"
TRANSCRIBE_CPP_URL = "https://github.com/handy-computer/transcribe.cpp.git"
MODEL_FILES = {
    "cohere": ("handy-computer/cohere-transcribe-03-2026-gguf", "cohere-transcribe-03-2026-Q8_0.gguf"),
    "qwen": ("handy-computer/Qwen3-ASR-1.7B-gguf", "Qwen3-ASR-1.7B-Q4_K_M.gguf"),
    "parakeet": ("handy-computer/parakeet-tdt-0.6b-v2-gguf", "parakeet-tdt-0.6b-v2-Q4_K_M.gguf"),
    "nemotron": ("handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf", "nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"),
}
COMMON_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".ogg", ".wav", ".flac", ".aac", ".webm"}
SAMPLE_RATE = 16000
# Practical chunk sizes for bounded GPU memory use. The upstream per-call
# limits describe what the model accepts, not what fits in available memory.
CHUNK_SECONDS = {"cohere": 30, "qwen": 60, "parakeet": 300, "nemotron": 60}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe media files on a GPU and write INPUT.txt.")
    parser.add_argument("input_paths", nargs="+", help="Source media paths or glob patterns.")
    parser.add_argument("--model", choices=MODEL_FILES, default="cohere", help="Model family (default: cohere).")
    parser.add_argument("--model-id", default=None, help="Override Hugging Face GGUF repository, or use a local .gguf file.")
    parser.add_argument("--language", default=None, help="Language hint (default: en for Cohere, auto for Nemotron).")
    parser.add_argument("--batch-size", type=int, default=None, help="Audio chunks processed together (default: 1).")
    return parser.parse_args()


def expand_input_paths(raw_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        for match in glob.glob(raw) or [raw]:
            path = Path(match).expanduser().resolve()
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def validate_input_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() not in COMMON_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported input format: {path.suffix}")


def resolve_language(model: str, language: str | None) -> str | None:
    if model == "cohere":
        return language or "en"
    if model == "nemotron":
        return None if language in (None, "auto") else language
    if model == "parakeet":
        if language not in (None, "en"):
            raise ValueError("Parakeet v2 only supports English (--language en).")
        return None
    if language is not None:
        raise ValueError("transcribe.cpp Qwen3-ASR currently supports automatic language detection only.")
    return None


def ensure_binary() -> Path:
    override = os.environ.get("TRANSCRIBE_CPP_BIN")
    if override:
        binary = Path(override).expanduser().resolve()
        if not binary.is_file():
            raise FileNotFoundError(f"TRANSCRIBE_CPP_BIN does not exist: {binary}")
        return binary

    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "transcribe-cli" / TRANSCRIBE_CPP_COMMIT
    binary = root / "build" / "bin" / "transcribe-cli"
    if binary.is_file():
        return binary
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    if not source.is_dir():
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", TRANSCRIBE_CPP_URL], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", TRANSCRIBE_CPP_COMMIT], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    subprocess.run(["cmake", "-S", str(source), "-B", str(root / "build"), "-DTRANSCRIBE_VULKAN=ON", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(root / "build"), "--target", "transcribe-cli", "-j", "4"], check=True)
    if not binary.is_file():
        raise RuntimeError("transcribe.cpp build did not produce transcribe-cli")
    return binary


def require_vulkan_gpu(binary: Path) -> None:
    result = subprocess.run([str(binary), "--list-devices"], capture_output=True, text=True, check=True)
    if not re.search(r"kind=vulkan\s+type=(?:igpu|dgpu)", result.stdout):
        raise RuntimeError("transcribe.cpp did not find a Vulkan GPU; CPU fallback is disabled.")


def resolve_model_file(model: str, model_id: str | None) -> Path:
    repo, filename = MODEL_FILES[model]
    if model_id:
        local = Path(model_id).expanduser()
        if local.is_file():
            return local.resolve()
        if model_id.endswith(".gguf"):
            raise FileNotFoundError(f"GGUF file not found: {model_id}")
        repo = model_id
    return Path(hf_hub_download(repo_id=repo, filename=filename))


def describe_input(input_file: Path) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample",
         "-of", "json", str(input_file)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No audio stream found: {input_file}")
    stream = streams[0]
    codec = stream.get("codec_name", "unknown").upper()
    if codec.startswith("PCM_"):
        codec = "PCM"
    bits = stream.get("bits_per_raw_sample")
    if not bits or not str(bits).isdigit() or int(bits) == 0:
        bits = stream.get("bits_per_sample")
    if bits and str(bits).isdigit() and int(bits) > 0:
        codec = f"{bits}-bit {codec}"
    channels = stream.get("channels")
    layout = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels} channels"
    rate = int(stream["sample_rate"]) / 1000
    container = input_file.suffix[1:].upper()
    format_name = f"{codec} audio" if codec == container else f"{codec} {container}"
    return f"{format_name}, {rate:g} kHz, {layout}"


def convert_and_chunk(input_file: Path, directory: Path, chunk_seconds: int | None) -> tuple[list[Path], float]:
    wav = directory / "audio.wav"
    direct_wav = False
    if input_file.suffix.lower() == ".wav":
        try:
            with wave.open(str(input_file), "rb") as reader:
                direct_wav = (reader.getnchannels(), reader.getframerate(), reader.getsampwidth(),
                              reader.getcomptype()) == (1, SAMPLE_RATE, 2, "NONE")
        except (wave.Error, EOFError):
            pass
    if direct_wav:
        wav = input_file
    else:
        print(f"Converting to 16-bit PCM WAV, 16 kHz, mono: {input_file}", flush=True)
        conversion_started = time.perf_counter()
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(input_file),
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
            check=True,
        )
        print(f"Conversion completed in {time.perf_counter() - conversion_started:.2f}s", flush=True)
    with wave.open(str(wav), "rb") as reader:
        if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth()) != (1, SAMPLE_RATE, 2):
            raise RuntimeError(f"ffmpeg produced an invalid WAV: {input_file}")
        total_frames = reader.getnframes()
        if total_frames == 0:
            raise ValueError(f"Decoded audio was empty: {input_file}")
        if chunk_seconds is None or total_frames <= chunk_seconds * SAMPLE_RATE:
            return [wav], total_frames / SAMPLE_RATE
        chunks: list[Path] = []
        index = 0
        while reader.tell() < total_frames:
            data = reader.readframes(chunk_seconds * SAMPLE_RATE)
            chunk = directory / f"chunk-{index:04d}.wav"
            with wave.open(str(chunk), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(SAMPLE_RATE)
                writer.writeframes(data)
            chunks.append(chunk)
            index += 1
        return chunks, total_frames / SAMPLE_RATE


def join_chunks(texts: list[str]) -> str:
    result = ""
    for text in texts:
        text = text.strip()
        if text:
            if result and not ("\u4e00" <= result[-1] <= "\u9fff" and "\u4e00" <= text[0] <= "\u9fff"):
                result += " "
            result += text
    return result


def split_wav(wav: Path) -> list[Path]:
    with wave.open(str(wav), "rb") as reader:
        frames = reader.getnframes()
        midpoint = frames // 2
        if midpoint < SAMPLE_RATE:
            raise RuntimeError(f"transcribe.cpp still truncated audio shorter than two seconds: {wav}")
        params = reader.getparams()
        halves = [reader.readframes(midpoint), reader.readframes(frames - midpoint)]
    paths = [wav.with_name(f"{wav.stem}-part-{index}.wav") for index in range(2)]
    for path, data in zip(paths, halves):
        with wave.open(str(path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(data)
    return paths


def transcribe(binary: Path, model_file: Path, wavs: list[Path], language: str | None, batch_size: int, directory: Path, model: str = "cohere") -> dict[str, str]:
    batch_file = directory / "batch.txt"
    batch_file.write_text("".join(f"{wav}\n" for wav in wavs), encoding="utf-8")
    # The transcribe.cpp 0.2.3 Parakeet/Nemotron batch path asserts inside ggml.
    if model in {"parakeet", "nemotron"}:
        batch_size = 1
    if batch_size > 1 and len(wavs) > 1:
        frame_counts = []
        for wav in wavs:
            with wave.open(str(wav), "rb") as reader:
                frame_counts.append(reader.getnframes())
        if len(set(frame_counts)) > 1:
            batch_size = 1
    command = [str(binary), "-m", str(model_file), "--backend", "vulkan", "--batch",
               str(batch_file), "--batch-jsonl", "--batch-size", str(batch_size), "--timestamps", "none"]
    if language:
        command.extend(["-l", language])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"transcribe.cpp GPU run failed:\n{result.stderr[-3000:]}\n{result.stdout[-1000:]}")
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    transcripts = {}
    truncated = []
    expected = {str(wav) for wav in wavs}
    for row in rows:
        if "file" in row:
            if row.get("error"):
                if row["error"].startswith("output truncated:") and row["file"] in expected:
                    truncated.append(Path(row["file"]))
                    continue
                raise RuntimeError(f"transcribe.cpp failed for {row['file']}: {row['error']}")
            transcripts[row["file"]] = row["text"]
    for wav in truncated:
        parts = split_wav(wav)
        print(f"Retrying truncated audio in shorter pieces: {wav}", file=sys.stderr, flush=True)
        retry = transcribe(binary, model_file, parts, language, 1, directory, model=model)
        transcripts[str(wav)] = join_chunks([retry[str(part)] for part in parts])
    if set(transcripts) != expected:
        raise RuntimeError("transcribe.cpp did not return one transcript for each audio chunk")
    return transcripts


def main() -> None:
    args = parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    language = resolve_language(args.model, args.language)
    inputs = expand_input_paths(args.input_paths)
    for input_file in inputs:
        validate_input_file(input_file)
    binary = ensure_binary()
    require_vulkan_gpu(binary)
    model_file = resolve_model_file(args.model, args.model_id)

    for input_file in inputs:
        with tempfile.TemporaryDirectory(prefix="transcribe-cli-") as temporary:
            root = Path(temporary)
            print(f"Processing file: {input_file}", flush=True)
            print(f"Input format: {describe_input(input_file)}", flush=True)
            chunks, duration = convert_and_chunk(input_file, root, CHUNK_SECONDS[args.model])
            print(f"Running transcribe.cpp with {args.model} on {len(chunks)} audio chunk(s)...", flush=True)
            started = time.perf_counter()
            transcripts = transcribe(binary, model_file, chunks, language, args.batch_size or 1, root, model=args.model)
            elapsed = time.perf_counter() - started
            output_file = input_file.with_suffix(".txt")
            output_file.write_text(join_chunks([transcripts[str(chunk)] for chunk in chunks]) + "\n", encoding="utf-8")
            print(f"Wrote transcript to {output_file} with {model_file} using Vulkan GPU")
            speedup = duration / elapsed if elapsed > 0 else float("inf")
            print(f"Metrics for {input_file}: {duration:.2f}s audio; batch completed in {elapsed:.2f}s ({speedup:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
