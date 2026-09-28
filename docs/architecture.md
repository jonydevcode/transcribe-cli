# Architecture

`transcribe-cli` turns local media files into `.txt` transcripts with transcribe.cpp on a Vulkan GPU. It never falls
back to CPU. Every concern has one module.

| Module | Concern |
| --- | --- |
| `cli.py` | argparse, wiring, exit codes |
| `errors.py` | `TranscribeCliError` hierarchy and exit codes |
| `models.py` | `ModelSpec` registry, language policy, GGUF resolution |
| `inputs.py` | glob expansion, validation, output path policy |
| `media.py` | ffprobe description, WAV passthrough check, ffmpeg normalisation |
| `segments.py` | pure span planning over frame counts, WAV slicing |
| `stitch.py` | pure overlap detection and transcript joining |
| `engine.py` | transcribe.cpp adapter: device check, command line, streamed JSONL |
| `provision.py` | locates the transcribe.cpp binary (built and pinned by the flake) |
| `pipeline.py` | per-file orchestration, truncation retry, `FileResult` |
| `reporting.py` | `Reporter` protocol, `ConsoleReporter`, `ChunkProgress` |

## Dependencies point downward

An arrow means "imports". This is the actual import graph:

```
cli.py ──────► inputs · provision · models · engine (TranscribeCpp) · reporting (ConsoleReporter)
  │
  ▼
pipeline.py ─► models (ModelSpec) · engine (Engine protocol, SegmentResult) · reporting (Reporter protocol)
  │
  ├──► media.py ──► segments (SAMPLE_RATE)
  ├──► segments.py
  └──► stitch.py

errors.py is imported by every module except stitch.py.
Type-only imports (under TYPE_CHECKING): reporting → pipeline.FileResult, media → reporting.Reporter.
```

`pipeline.py` never imports `cli.py` or `ConsoleReporter`. `stitch.py` and `segments.py` import nothing from the
package except `errors`.

## Rules

1. Only `cli.py` and `reporting.py` write to stdout/stderr. Everything else reports through the `Reporter` protocol
   or return values.
2. Only `media.py`, `engine.py` start subprocesses. Each wraps one external tool family (FFmpeg, transcribe.cpp).
3. Only `models.py` knows model names.
4. `stitch.py` and `segments.py` are pure. The only I/O is `segments.write_span`.
5. Dependencies are passed in, not patched: `Transcriber` receives its engine, reporter and clock through its
   constructor. Tests do not patch private functions.
6. User-facing failures are `TranscribeCliError`s. `cli.main` prints `error: <message>` and returns the exit code.
   Any other exception is a bug and keeps its traceback.

## Deliberate deviations from a strict reading of the rules

- `Reporter` has a `tick()` method and `Engine.transcribe` an `on_idle` callback, so the terminal progress bar can
  follow terminal resizes while the engine is silent.
- `ConsoleReporter` takes the resolved model file, so `file_finished` can print it.
- `engine.require_vulkan_gpu(binary)` exists as a function (and as a method) so the GPU check can run before a model
  is downloaded.

## Invariants

- Adjacent audio spans share at least the overlap `stitch.py` needs to find a phrase on both sides of a join.
- `models.ModelSpec.supports_batching=False` records that transcribe.cpp 0.2.3's Parakeet/Nemotron batch path
  asserts inside ggml. Re-test it when the transcribe.cpp pin in `flake.nix` changes.
