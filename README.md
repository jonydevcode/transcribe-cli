# transcribe-cli

Copyright 2026 jonydevcode

![AI Assisted](https://img.shields.io/badge/AI--assisted-repository-blue)
![AI Use Disclosed](https://img.shields.io/badge/AI%20use-disclosed-orange)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](./LICENSE)

Small Nix-managed CLI for transcribing local audio and video files on an AMD
GPU.

The script accepts one or more local media paths, runs offline transcription with a Hugging Face `transformers` model, and writes a `.txt` transcript next to each input. Choose Cohere Transcribe (the default), Qwen3-ASR-0.6B, or Nemotron 3.5 ASR with `--model`.

[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) remains the default because it has produced high quality results. The Qwen option uses the [official Transformers-native 0.6B checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf). The Nemotron option uses [NVIDIA's Transformers-compatible 0.6B checkpoint](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

## AI Use Disclosure

LLM/AI tools were used in the creation and development of this repository. Some contents may be AI-generated or AI-assisted, alongside human-authored specifications, edits, review, testing, and maintenance.

All material is maintained under the repository’s license unless otherwise noted. Contributors are responsible for making sure any changes they submit are correct, secure, properly licensed, and not copied from sources they are not allowed to use.

## Features

- Accepts common input formats such as `mp3`, `m4a`, `mp4`, `ogg`, `wav`, `flac`, `aac`, and `webm`
- Accepts multiple input files or glob patterns in a single invocation
- Uses Cohere's native long-form chunking or 60-second Qwen and Nemotron chunks with automatic language detection on each chunk
- Loads the model once per CLI invocation, then transcribes each input in sequence
- Writes output to `INPUT.txt`
- Selects Cohere, Qwen, or Nemotron with `--model`; `--model-id` can override the checkpoint for the selected model family
- Supports a configurable language code via `--language`; Qwen and Nemotron default to automatic detection
- Supports configurable inference chunk batching via `--batch-size`
- Reproducible Python and ROCm dependencies supplied entirely by Nix

## Requirements

- x86_64 NixOS
- An AMD GPU supported by ROCm and access to `/dev/kfd`
- Enough disk space and memory for the model download and inference
- Internet access on first run so Hugging Face assets can be downloaded

## Installation

Enter the development environment:

```bash
nix develop
```

The shell provides Python 3.13 and the Nixpkgs `torchWithRocm`, Transformers 5.13.1,
Hugging Face Hub, audio dependencies, FFmpeg, and ROCm diagnostic tools. There
is no separate dependency-install or environment-repair step. PyTorch exposes
ROCm devices through its CUDA-compatible Python API, so
`torch.cuda.is_available()` is expected for AMD GPUs.

When `rocminfo` detects the Radeon 860M (`gfx1152`), the shell selects ROCm's
compatible `gfx1150` code path and enables experimental AOTriton attention for
faster inference. These overrides are not set for other GPUs.

Verify ROCm and PyTorch device access from `nix develop` with:

```bash
rocminfo | grep -E 'Name:|gfx'
python -c 'import torch; print(torch.__version__); print("HIP:", torch.version.hip); print("available:", torch.cuda.is_available()); print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
```

The Cohere model card recommends the native `transformers` path for offline inference and notes testing with `torch==2.10.0`, while expecting nearby versions to work.

## Basic Transcription

Run a transcription with:

```bash
python main.py INPUT.EXT
```

Example:

```bash
python main.py meeting.wav
```

This writes:

```text
meeting.txt
```

into the same directory as the input file.

The default is Cohere. You can select it explicitly with `--model cohere`.

For long Cohere files on GPU, you can reduce memory use with:

```bash
python main.py INPUT.EXT --batch-size 4
```

Select Qwen3-ASR-0.6B, with automatic language detection across English and Chinese segments:

```bash
python main.py INPUT.EXT --model qwen
```

`--model qwen` loads `Qwen/Qwen3-ASR-0.6B-hf`. It uses the same ROCm GPU selection as Cohere and defaults to a batch size of `1` to limit memory use. On a 64.58-second WAV recorded with a Radeon 860M, this checkpoint took 13.33–14.16 seconds (4.56–4.84× real-time), compared with about 30–31 seconds for the 1.7B checkpoint.

To use the larger Qwen model instead:

```bash
python main.py INPUT.EXT --model qwen --model-id Qwen/Qwen3-ASR-1.7B-hf
```

Select Nemotron 3.5 ASR with automatic language detection:

```bash
python main.py INPUT.EXT --model nemotron
```

This uses `nvidia/nemotron-3.5-asr-streaming-0.6b` through Transformers' offline RNNT interface. The CLI splits long files into 60-second chunks and defaults to a batch size of `1`. It writes clean transcript text; Nemotron's detected language tags are removed during decoding. This option does not expose live streaming.

Use a different compatible Hugging Face checkpoint for the selected model family with:

```bash
python main.py INPUT.EXT --model-id MODEL_ID
```

The checkpoint must support the selected model's processor and inference path. The original Qwen checkpoints use Qwen's separate `qwen-asr` package; use the `-hf` checkpoints with this CLI.

Batch multiple files into one run to avoid reloading the model:

```bash
python main.py a.mp3 b.mp3 c.mp3
```

Shell-expanded globs work too:

```bash
python main.py 001-meeting/*
```

## Language Selection

The Cohere model defaults to English:

```bash
python main.py INPUT.EXT --language en
```

Use a different ISO 639-1 language code if needed:

```bash
python main.py INPUT.EXT --language fr
```

Qwen detects the language of each audio chunk by default. Leave `--language` unset for audio containing both English and Chinese. `--language en` or `--language zh` forces a single language.

Nemotron also detects the language of each chunk by default. Use a supported locale such as `--language en-US` or `--language de-DE` to condition transcription on a known language. It also accepts bare language codes such as `de`, or `--language auto`. See the [Nemotron model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for supported locales and accuracy tiers.

## Q4 Quantization

A community [Q4_K_M GGUF of Qwen3-ASR-1.7B](https://huggingface.co/slyusarev/Qwen3-ASR-1.7B-GGUF) is available at about 1.28 GB, plus its required `mmproj` file. It runs with llama.cpp. The CLI's Qwen option uses Transformers checkpoints and does not load GGUF files.

## How It Works

- `main.py` expands input glob patterns, validates each path and extension, and deduplicates repeated matches
- The script loads the processor and model class for the selected model
- It decodes and resamples each input with `load_audio(..., sampling_rate=16000)`
- Cohere's processor chunks long audio and returns `audio_chunk_index` for transcript reassembly
- Qwen audio is split into 60-second chunks and passed through `apply_transcription_request` without a language hint by default; its decoded output is reduced to transcription text
- Nemotron audio is split into 60-second chunks, processed with an automatic language prompt by default, and decoded without language tags
- All models call `model.generate(...)` in explicit mini-batches and use the same PyTorch ROCm device selection
- Each transcript is written to a `.txt` file using the same base filename as its source input

## Notes

- First run will be slower because model code and weights must be downloaded
- Long files can take significant time; GPU memory use depends heavily on `--batch-size`
- Cohere defaults to `--batch-size 8` on ROCm GPU and `1` on CPU; Qwen defaults to `1` to limit memory use
- The script does not create an intermediate converted audio file on disk
- Audio samples obtained from the LJ Speech Dataset

## Version Notes

- The Cohere model card documents the native `transformers` path for offline inference
- Qwen uses its official Transformers-native conversion, which requires Transformers 5.13 or newer
- Nemotron 3.5 ASR uses Transformers' native RNNT implementation, available in Transformers 5.13 or newer
- The runtime selects ROCm (through PyTorch's `cuda` device API) or CPU explicitly instead of relying on `device_map="auto"`
- Manual chunk batching is implemented in the CLI to avoid sending every long-form chunk through `generate(...)` in one large batch

## Project Files

- `main.py`: CLI entrypoint
- `flake.nix`: pinned Nix development environment and dependencies
- `test_main.py`: focused CLI and model chunking tests
- `AGENTS.md`: quick start for future agents

Run the focused tests from the development shell with `python -m unittest -v test_main`. Full Qwen and Nemotron inference requires downloading their model weights.

## Model Checkpoints

- Default model: <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026>
- Qwen default: <https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf>
- Nemotron default: <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>
- Larger Qwen option: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf>

## Reference

- Hugging Face `transformers`: <https://huggingface.co/docs/transformers>

## License

This project is licensed under Apache License 2.0.
