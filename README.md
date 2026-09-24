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

On x86_64 NixOS, enter the shell:

```bash
nix develop
vulkaninfo --summary
```

The shell supplies Python, Hugging Face Hub, FFmpeg, CMake, Git, and Vulkan build tools. On first use, `main.py` fetches pinned transcribe.cpp source, builds its Vulkan CLI in `~/.cache/transcribe-cli/`, and downloads the selected GGUF into the Hugging Face cache. Later runs reuse both. Set `TRANSCRIBE_CPP_BIN=/path/to/transcribe-cli` to use an existing Vulkan-enabled binary. Set `XDG_CACHE_HOME` to move the build cache.

The Vulkan driver must expose a compute-capable GPU. On the Radeon 860M, the RADV driver identifies the GPU as `Vulkan0`. The project tested the Cohere model with `--backend vulkan` on that device. The Nix ROCm build was unable to run on its `gfx1152` GPU because the available rocBLAS package lacks the architecture's kernel library; Vulkan is the supported path here.

## Usage

```bash
python main.py recording.wav
python main.py recording.wav --model qwen
python main.py recording.wav --model parakeet
python main.py recording.wav --model nemotron
python main.py a.mp3 b.mp4 'inputs/*.WAV' --batch-size 2
python main.py recording.wav --model cohere --language fr
python main.py recording.wav --model qwen --model-id /path/to/Qwen3-ASR-1.7B-Q8_0.gguf
```

Each source gets a same-name `.txt` file beside it. The CLI accepts MP3, M4A, MP4, OGG, WAV, FLAC, AAC, and WEBM. FFmpeg converts inputs to 16 kHz mono WAV. A single transcribe.cpp session processes all files or chunks in the invocation. `--batch-size` controls Cohere and Qwen batch size when all chunks have equal length; mixed-length inputs run serially. Parakeet and Nemotron also run serially because transcribe.cpp 0.2.3's batch path fails for these models. The default is one to limit GPU memory use.

`--model-id` accepts a local GGUF file or a Hugging Face GGUF repository containing the selected model's default filename. Cohere defaults to English and accepts a language hint. Nemotron defaults to automatic language selection and accepts a supported locale. Qwen detects language automatically; its transcribe.cpp port currently rejects explicit language hints. Parakeet v2 is English-only. Long recordings are split into chunks below each model's audio limit.

## Accuracy check

The repository includes `inputs/20260726223646.WAV` and the earlier Cohere transcript `inputs/20260726223646_cohere.txt`. A new GGUF transcription can be compared against that source of truth with:

```bash
python main.py inputs/20260726223646.WAV --model cohere
diff -u inputs/20260726223646_cohere.txt inputs/20260726223646.txt
```

The text may differ because this CLI uses a separate runtime and quantized checkpoint. The final validation should inspect the differences rather than require byte-for-byte identity.

## AI Use Disclosure

LLM/AI tools were used in the creation and development of this repository. Some contents may be AI-generated or AI-assisted, alongside human-authored specifications, edits, review, testing, and maintenance.
