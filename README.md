# transcribe-cli

Copyright 2026 jonydevcode

![AI Assisted](https://img.shields.io/badge/AI--assisted-repository-blue)
![AI Use Disclosed](https://img.shields.io/badge/AI%20use-disclosed-orange)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](./LICENSE)

Small `uv`-managed CLI for transcribing local audio and video files.

The script accepts one or more local media paths, runs offline transcription with a Hugging Face `transformers` model, and writes a `.txt` transcript next to each input.

For the moment, [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) is the selected transcription model because it has produced high quality results. The project is intended to be a model-agnostic transcription CLI, not a Cohere-specific wrapper.

## AI Use Disclosure

LLM/AI tools were used in the creation and development of this repository. Some contents may be AI-generated or AI-assisted, alongside human-authored specifications, edits, review, testing, and maintenance.

All material is maintained under the repository’s license unless otherwise noted. Contributors are responsible for making sure any changes they submit are correct, secure, properly licensed, and not copied from sources they are not allowed to use.

## Features

- Accepts common input formats such as `mp3`, `m4a`, `mp4`, `ogg`, `wav`, `flac`, `aac`, and `webm`
- Accepts multiple input files or glob patterns in a single invocation
- Uses the selected model's native long-form chunking and transcript reassembly path
- Loads the model once per CLI invocation, then transcribes each input in sequence
- Writes output to `INPUT.txt`
- Supports selecting a Hugging Face ASR model via `--model-id`
- Supports a configurable language code via `--language`
- Supports configurable inference chunk batching via `--batch-size`
- Managed with `uv`

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Enough disk space and memory for the model download and inference
- Internet access on first run so Hugging Face assets can be downloaded
- On x86_64 Linux, an AMD GPU supported by ROCm and access to `/dev/kfd`
- On Apple Silicon, use an arm64 Python build so PyTorch can access the `mps` backend

## Installation

Enter the development environment and install dependencies with:

```bash
nix develop
uv sync --frozen
./scripts/link-rocm-libraries
```

On x86_64 Linux, `uv` installs the ROCm build of PyTorch. Other supported
platforms use the CPU build, with Apple Silicon inference accelerated through
PyTorch MPS. PyTorch exposes ROCm devices through its CUDA-compatible Python API,
so `torch.cuda.is_available()` is expected for AMD GPUs.

The link helper supplies versioned ELF library aliases omitted by the standalone
ROCm wheel. It only adds missing symbolic links inside `.venv`; rerunning it is
safe.

The Nix shell selects ROCm's compatible `gfx1150` code path for the Radeon 860M
(`gfx1152`). Without this override, the current ROCm runtime detects the GPU but
crashes while compiling its first device kernel.

Verify ROCm and PyTorch device access from `nix develop` with:

```bash
rocminfo | grep -E 'Name:|gfx'
uv run python -c 'import torch; print(torch.__version__); print("HIP:", torch.version.hip); print("available:", torch.cuda.is_available()); print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
```

The model card currently recommends the native `transformers` path for offline inference and notes testing with `torch==2.10.0`, while expecting nearby versions to work.

If you need to add or update packages later, use:

```bash
uv add PACKAGE_NAME
```

## Basic Transcription

Run a transcription with:

```bash
uv run main.py INPUT.EXT
```

Example:

```bash
uv run main.py meeting.wav
```

This writes:

```text
meeting.txt
```

into the same directory as the input file.

For long files on GPU, you can reduce memory use further with:

```bash
uv run main.py INPUT.EXT --batch-size 4
```

Use a different compatible Hugging Face ASR model with:

```bash
uv run main.py INPUT.EXT --model-id MODEL_ID
```

Batch multiple files into one run to avoid reloading the model:

```bash
uv run main.py a.mp3 b.mp3 c.mp3
```

Shell-expanded globs work too:

```bash
uv run main.py 001-meeting/*
```

## Language Selection

The script defaults to English:

```bash
uv run main.py INPUT.EXT --language en
```

Use a different ISO 639-1 language code if needed:

```bash
uv run main.py INPUT.EXT --language fr
```

## How It Works

- `main.py` expands input glob patterns, validates each path and extension, and deduplicates repeated matches
- The script loads the current `transformers` processor and ASR model class
- It decodes and resamples each input with `load_audio(..., sampling_rate=16000)`
- The processor chunks long audio and returns `audio_chunk_index` for transcript reassembly
- The script calls `model.generate(...)` in explicit mini-batches over the generated chunks
- Generated tokens are trimmed to remove the decoder prompt before text decoding
- The processor reassembles the per-chunk text into one transcript
- Each transcript is written to a `.txt` file using the same base filename as its source input

## Notes

- First run will be slower because model code and weights must be downloaded
- Long files can take significant time; GPU memory use depends heavily on `--batch-size`
- This CLI defaults to `--batch-size 32` on ROCm GPU, `4` on Apple Silicon `mps`, and `1` on CPU
- On macOS, the script enables `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops can fall back to CPU instead of failing outright
- The script does not create an intermediate converted audio file on disk
- Audio samples obtained from the LJ Speech Dataset

## Version Notes

- The current model card documents both the native `transformers` path and a `trust_remote_code=True` helper
- This CLI uses the native path because the current model card identifies it as the recommended offline inference path
- The runtime selects ROCm (through PyTorch's `cuda` device API), `mps`, or `cpu` explicitly instead of relying on `device_map="auto"`
- Manual chunk batching is implemented in the CLI to avoid sending every long-form chunk through `generate(...)` in one large batch

## Project Files

- `main.py`: CLI entrypoint
- `pyproject.toml`: project metadata and dependencies
- `uv.lock`: locked dependency versions

## Current Model

- Selected model: <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026>
- Rationale: currently selected because it has produced high quality transcription results
- Direction: keep the CLI model-agnostic where practical so future models can be evaluated or substituted

## Reference

- Hugging Face `transformers`: <https://huggingface.co/docs/transformers>

## License

This project is licensed under Apache License 2.0.
