# SPDX-License-Identifier: Apache-2.0
"""GPU-only transcribe.cpp frontend."""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
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
CHUNK_OVERLAP_SECONDS = 3


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
        chunk_frames = chunk_seconds * SAMPLE_RATE
        overlap_frames = min(CHUNK_OVERLAP_SECONDS * SAMPLE_RATE, chunk_frames // 2)
        chunks: list[Path] = []
        start = 0
        while start < total_frames:
            end = min(start + chunk_frames, total_frames)
            reader.setpos(start)
            data = reader.readframes(end - start)
            chunk = directory / f"chunk-{len(chunks):04d}.wav"
            with wave.open(str(chunk), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(SAMPLE_RATE)
                writer.writeframes(data)
            chunks.append(chunk)
            if end == total_frames:
                break
            start = end - overlap_frames
        return chunks, total_frames / SAMPLE_RATE


def join_chunks(texts: list[str]) -> str:
    parts: list[str] = []
    last_char = ""
    for text in texts:
        text = text.strip()
        if text:
            if parts and not ("\u4e00" <= last_char <= "\u9fff" and "\u4e00" <= text[0] <= "\u9fff"):
                parts.append(" ")
            parts.append(text)
            last_char = text[-1]
    return "".join(parts)


def text_tokens(text: str) -> list[tuple[str, int]]:
    # Keep Han characters separate even when they touch Latin text.
    pattern = r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+(?:['’][^\W_\u3400-\u9fff]+)*"
    return [(unicodedata.normalize("NFKC", match.group()).casefold(), match.end())
            for match in re.finditer(pattern, text, re.UNICODE)]


def overlap_offset(previous: str, following: str) -> int | None:
    left = text_tokens(previous)[-32:]
    right = text_tokens(following)[:32]
    left_words = [word for word, _ in left]
    right_words = [word for word, _ in right]
    for size in range(min(len(left), len(right)), 2, -1):
        phrase = left_words[-size:]
        if len(set(phrase)) < 2 or (all("\u3400" <= word <= "\u9fff" for word in phrase) and size < 6):
            continue
        if phrase == right_words[:size]:
            return right[size - 1][1]
    # Permit a few differently recognized words before a shared edge phrase.
    for block in difflib.SequenceMatcher(None, left_words, right_words, autojunk=False).get_matching_blocks():
        if (block.size < 3 or block.a + block.size != len(left)
                or not 1 <= block.b <= 6 or block.a < block.b):
            continue
        phrase = left_words[block.a:block.a + block.size]
        if len(set(phrase)) < 2 or (all("\u3400" <= word <= "\u9fff" for word in phrase) and block.size < 6):
            continue
        preceding = zip(left_words[block.a - block.b:block.a], right_words[:block.b])
        if all(difflib.SequenceMatcher(None, a, b).ratio() >= 0.6 for a, b in preceding):
            return right[block.b + block.size - 1][1]
    return None


def stitch_chunks(texts: list[str]) -> tuple[str, int]:
    parts: list[str] = []
    unresolved = 0
    previous = ""
    for text in texts:
        text = text.strip()
        original = text
        if previous and text:
            offset = overlap_offset(previous, text)
            if offset is None:
                unresolved += 1
            else:
                text = text[offset:].lstrip(" \t\r\n.,!?;:。？！、，")
        parts.append(text)
        previous = original
    return join_chunks(parts), unresolved


def split_wav(wav: Path, directory: Path) -> list[Path]:
    with wave.open(str(wav), "rb") as reader:
        frames = reader.getnframes()
        midpoint = frames // 2
        if midpoint < SAMPLE_RATE:
            raise RuntimeError(f"transcribe.cpp still truncated audio shorter than two seconds: {wav}")
        params = reader.getparams()
        overlap = min(CHUNK_OVERLAP_SECONDS * SAMPLE_RATE, frames // 4)
        ranges = [(0, midpoint + overlap // 2), (midpoint - overlap // 2, frames)]
    paths = [directory / f"{wav.stem}-part-{index}.wav" for index in range(2)]
    for path, (start, end) in zip(paths, ranges):
        with wave.open(str(wav), "rb") as reader:
            reader.setpos(start)
            data = reader.readframes(end - start)
        with wave.open(str(path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(data)
    return paths


class ChunkProgress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.tty = sys.stderr.isatty()
        self.width = 0
        self.render()

    def render(self) -> None:
        if not self.tty:
            if self.done:
                print(f"Chunks processed: {self.done}/{self.total}", file=sys.stderr, flush=True)
            return
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        if width == self.width and self.done == getattr(self, "rendered", -1):
            return
        self.width = width
        self.rendered = self.done
        label = f"Chunks {self.done}/{self.total} "
        space = max(0, width - len(label) - 4)
        if space >= 4:
            filled = space * self.done // self.total
            line = f"{label}[{'#' * filled}{'-' * (space - filled)}]"
        else:
            line = label.rstrip()[:max(1, width - 1)]
        sys.stderr.write("\r\033[K" + line)
        sys.stderr.flush()

    def advance(self) -> None:
        self.done += 1
        self.render()

    def close(self) -> None:
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


def run_batch(command: list[str], wavs: list[Path]) -> subprocess.CompletedProcess[str]:
    progress = ChunkProgress(len(wavs))
    expected = {str(wav) for wav in wavs}
    reported: set[str] = set()
    lines: list[str] = []
    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
                assert process.stdout is not None
                pending = b""
                while True:
                    ready, _, _ = select.select([process.stdout], [], [], 0.2)
                    if not ready:
                        progress.render()
                        continue
                    data = os.read(process.stdout.fileno(), 65536)
                    if not data:
                        break
                    pending += data
                    while b"\n" in pending:
                        raw, pending = pending.split(b"\n", 1)
                        line = raw.decode("utf-8", errors="replace") + "\n"
                        lines.append(line)
                        if line.startswith("{"):
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            path = row.get("file")
                            if path in expected and path not in reported:
                                reported.add(path)
                                progress.advance()
                if pending:
                    lines.append(pending.decode("utf-8", errors="replace"))
                returncode = process.wait()
            errors.seek(0)
            stderr = errors.read()
        return subprocess.CompletedProcess(command, returncode, "".join(lines), stderr)
    finally:
        progress.close()


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
    result = run_batch(command, wavs)
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
        parts = split_wav(wav, directory)
        print(f"Retrying truncated audio in shorter pieces: {wav}", file=sys.stderr, flush=True)
        retry = transcribe(binary, model_file, parts, language, 1, directory, model=model)
        transcripts[str(wav)], unresolved = stitch_chunks([retry[str(part)] for part in parts])
        if unresolved:
            print(f"Warning: could not align {unresolved} retry overlap(s) for {wav}", file=sys.stderr, flush=True)
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
            transcript, unresolved = stitch_chunks([transcripts[str(chunk)] for chunk in chunks])
            elapsed = time.perf_counter() - started
            output_file = input_file.with_suffix(".txt")
            output_file.write_text(transcript + "\n", encoding="utf-8")
            if unresolved:
                print(f"Warning: could not align {unresolved} chunk overlap(s) for {input_file}; "
                      "check for repeated words", file=sys.stderr, flush=True)
            print(f"Wrote transcript to {output_file} with {model_file} using Vulkan GPU")
            speedup = duration / elapsed if elapsed > 0 else float("inf")
            print(f"Metrics for {input_file}: {duration:.2f}s audio; batch completed in {elapsed:.2f}s ({speedup:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
