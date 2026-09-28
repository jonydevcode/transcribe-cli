# transcribe-cli

Copyright 2026 jonydevcode

![AI Assisted](https://img.shields.io/badge/AI--assisted-repository-blue)
![AI Use Disclosed](https://img.shields.io/badge/AI%20use-disclosed-orange)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](./LICENSE)

Small Nix-managed CLI for transcribing local audio and video on a **Vulkan GPU** with [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp). The CLI refuses to run if transcribe.cpp cannot find a Vulkan device; it never falls back to CPU.

## Models

| `--model` | GGUF repository | Default file |
| --- | --- | --- |
| `cohere` (default) | [Cohere Transcribe 03-2026](https://huggingface.co/handy-computer/cohere-transcribe-03-2026-gguf) | `cohere-transcribe-03-2026-Q8_0.gguf` |
| `qwen` | [Qwen3-ASR 1.7B](https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf) | `Qwen3-ASR-1.7B-Q4_K_M.gguf` |
| `parakeet` | [Parakeet TDT 0.6B v2](https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf) | `parakeet-tdt-0.6b-v2-Q4_K_M.gguf` |
| `nemotron` (existing option) | [Nemotron 3.5 ASR](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf) | `nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf` |

## Setup

On x86_64 NixOS:

```bash
nix run . -- recording.wav     # builds transcribe.cpp (Vulkan) and the app, then runs it
nix develop                    # development shell: Python, pytest, ruff, mypy, FFmpeg, vulkan-tools
vulkaninfo --summary
python -m transcribe_cli recording.wav   # inside the shell, from the repository root
```

The flake pins and builds transcribe.cpp with its Vulkan backend and wraps the CLI so `TRANSCRIBE_CPP_BIN` and FFmpeg are on hand. The selected GGUF downloads into the Hugging Face cache on first use. Set `TRANSCRIBE_CPP_BIN=/path/to/transcribe-cli` to use a different Vulkan-enabled transcribe.cpp binary.

The Vulkan driver must expose a compute-capable GPU. On the Radeon 860M, the RADV driver identifies the GPU as `Vulkan0`. The project tested the Cohere model with `--backend vulkan` on that device. The Nix ROCm build was unable to run on its `gfx1152` GPU because the available rocBLAS package lacks the architecture's kernel library; Vulkan is the supported path here.

## Usage

The examples use `transcribe`, the installed command. Run `nix run . -- ARGS` from the repository, or `python -m transcribe_cli ARGS` inside `nix develop`, which sets `PYTHONPATH=src` but does not install the command.

```bash
transcribe recording.wav
transcribe recording.wav --model qwen
transcribe recording.wav --model parakeet
transcribe recording.wav --model nemotron
transcribe a.mp3 b.mp4 'inputs/*.WAV' --batch-size 2
transcribe recording.wav --model cohere --language fr
transcribe recording.wav --model qwen --model-id /path/to/Qwen3-ASR-1.7B-Q8_0.gguf
```

Each source gets a same-name `.txt` file beside it. The CLI accepts MP3, M4A, MP4, OGG, WAV, FLAC, AAC, and WEBM. Two inputs that would write the same `.txt` (for example `a.mp3` and `a.wav`) are rejected before any work starts. Existing 16 kHz mono 16-bit PCM WAV files are passed directly to transcribe.cpp; FFmpeg converts other inputs. Files are processed in order. For each file, the CLI prints its name and detected audio format, announces any conversion to 16 kHz mono 16-bit PCM WAV and its elapsed time, then announces the transcribe.cpp run. It shows a chunk progress bar that adjusts to terminal resizing, or plain chunk counts when output is redirected. It prints the transcript path, audio duration, inference time, and speedup when the file finishes. `--batch-size` controls Cohere and Qwen batch size when a file is split into equal-length chunks; mixed-length chunks run serially. Parakeet and Nemotron also run serially because transcribe.cpp 0.2.3's batch path fails for these models. The default is one to limit GPU memory use.

`--model-id` accepts a local GGUF file or a Hugging Face GGUF repository containing the selected model's default filename. Cohere defaults to English and accepts a language hint. Nemotron defaults to automatic language selection and accepts a supported locale. Qwen detects language automatically; its transcribe.cpp port currently rejects explicit language hints. Parakeet v2 is English-only. Files longer than 30 seconds for Cohere, 60 seconds for Qwen or Nemotron, or 300 seconds for Parakeet are split into chunks to bound memory use. Adjacent chunks share three seconds of audio. The CLI removes repeated text when the end of one transcript matches the start of the next; otherwise it keeps both and reports the unmatched joins. This adds about 11% more audio processing for long Cohere recordings. These practical chunk sizes are shorter than the models' documented per-call audio limits.

If transcribe.cpp reaches its output limit on a chunk, the CLI retries that chunk in progressively shorter pieces while keeping the successful chunks. A persistent truncation below two seconds remains an error.

Failures print `error: <message>` and exit with a code: 2 for usage errors, 3 for a missing prerequisite (no Vulkan GPU, FFmpeg, or transcribe.cpp binary), 4 for unreadable media, 5 for an engine failure, 130 after Ctrl-C.

## Development

```bash
nix develop -c pytest
nix develop -c ruff check .
nix develop -c mypy
nix flake check     # all of the above, hermetically
```

The code map and its rules are in [docs/architecture.md](docs/architecture.md). Adding a model is one `ModelSpec` entry in `src/transcribe_cli/models.py` plus a row in the table above. The accuracy comparison is in [docs/accuracy-check.md](docs/accuracy-check.md).

## AI Use Disclosure

LLM/AI tools were used in the creation and development of this repository. Some contents may be AI-generated or AI-assisted, alongside human-authored specifications, edits, review, testing, and maintenance.
