# MIGRATION.md — Re-architecting transcribe-cli

## 1. Desired end state

`transcribe-cli` stays a small, single-purpose, Nix-managed tool. It turns local media files into `.txt` transcripts using transcribe.cpp on a Vulkan GPU and never falls back to CPU. It gets no new features. The goal is a codebase where every concern has one home. Adding a model, changing chunking, or swapping the progress display should touch one module, and every behaviour should be testable without a GPU, FFmpeg, the network, or `patch.object` on private functions.

### 1.1 Repository layout

```
transcribe-cli/
├── flake.nix                     # builds transcribe.cpp (Vulkan) + the Python app; devShell; checks
├── flake.lock
├── pyproject.toml                # package metadata, console script, ruff/mypy/pytest config
├── README.md                     # user docs only
├── AGENTS.md                     # agent quick start (updated commands)
├── docs/
│   ├── architecture.md           # the module map + rules from §1.3 (short)
│   ├── accuracy-check.md         # moved from README
│   └── history/AMD_PERFORMANCE.md  # marked historical (PyTorch era, not transcribe.cpp)
├── src/transcribe_cli/
│   ├── __init__.py
│   ├── __main__.py               # `python -m transcribe_cli`
│   ├── cli.py                    # argparse, wiring, exit codes           (~90 lines)
│   ├── errors.py                 # exception hierarchy                     (~25)
│   ├── models.py                 # ModelSpec registry, language policy, GGUF resolution (~80)
│   ├── inputs.py                 # glob expansion, validation, output path policy (~50)
│   ├── media.py                  # ffprobe description, WAV passthrough check, ffmpeg normalisation (~80)
│   ├── segments.py               # pure span planning + WAV slicing        (~70)
│   ├── stitch.py                 # pure overlap detection + transcript joining (~90)
│   ├── engine.py                 # transcribe.cpp adapter: device check, command, streaming JSONL (~110)
│   ├── provision.py              # locate the transcribe.cpp binary        (~25 after Nix phase)
│   ├── pipeline.py               # per-file orchestration, truncation retry, FileResult (~110)
│   └── reporting.py              # Reporter protocol, ConsoleReporter, ChunkProgress (~90)
└── tests/
    ├── conftest.py               # fake engine binary, WAV factory, dummy .gguf
    ├── fixtures/                 # sample_15s.aac, sample_1m6s.aac (moved from repo root)
    ├── test_cli_e2e.py           # black-box: real CLI process, fake transcribe.cpp
    ├── test_models.py
    ├── test_inputs.py
    ├── test_media.py
    ├── test_segments.py
    ├── test_stitch.py
    ├── test_engine.py
    ├── test_pipeline.py
    └── test_reporting.py
```

The total size grows modestly, to roughly 800 lines. None of the files mixes concerns, and the largest file is roughly a quarter of today's `main.py`.

### 1.2 Dependency graph

```
                cli.py
      ┌────────┬──┴────────┬──────────────┐
      ▼        ▼           ▼              ▼
  inputs.py  models.py  provision.py  reporting.py (ConsoleReporter)
                 │                        ▲ implements
                 ▼                        │
             pipeline.py ──── Reporter protocol
      ┌─────────┼──────────┬──────────┐
      ▼         ▼          ▼          ▼
  media.py  segments.py  stitch.py  engine.py
                                      │
                                      ▼
                                  errors.py (imported by all)
```

Dependencies only point downward. `pipeline.py` never imports `cli.py` or `ConsoleReporter`. `stitch.py` and `segments.py` import nothing from the package except `errors`.

### 1.3 Architectural rules (enforced in review, documented in `docs/architecture.md`)

1. **Only `cli.py` and `reporting.py` write to stdout/stderr.** Every other module reports through the `Reporter` protocol or its return values.
2. **Only `media.py`, `engine.py` and `provision.py` start subprocesses.** Each one wraps exactly one external tool family (FFmpeg, transcribe.cpp, git/cmake).
3. **Only `models.py` knows model names.** No other module contains `"cohere"`, `"parakeet"`, etc.
4. **`stitch.py` and `segments.py` are pure.** Given the same inputs they return the same outputs, with no I/O except `segments.write_span`.
5. **Dependencies are passed in, not patched.** `Transcriber` receives its engine, reporter and clock through its constructor. Tests never need `patch.object(module, "private_fn")`.
6. **User-facing failures are `TranscribeCliError`s.** `cli.main` turns them into a one-line message and an exit code. Any other exception is a bug, so it keeps its traceback.

### 1.4 Key interfaces

```python
# models.py
class LanguageSupport(Enum):
    HINT = "hint"              # any hint passed through (Cohere, Nemotron)
    ENGLISH_ONLY = "english"   # only "en" accepted, nothing passed (Parakeet v2)
    AUTO_ONLY = "auto"         # any hint rejected (Qwen3-ASR port)

@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    filename: str
    chunk_seconds: int                 # practical GPU-memory bound, not the model's limit
    language: LanguageSupport
    default_language: str | None       # "en" for Cohere, None for Nemotron
    auto_means_unset: bool = False     # Nemotron: "auto" -> pass no -l
    supports_batching: bool = True     # False for Parakeet/Nemotron on transcribe.cpp 0.2.3

    def resolve_language(self, requested: str | None) -> str | None: ...

MODELS: Mapping[str, ModelSpec]      # the single registry; argparse choices come from here
DEFAULT_MODEL = "cohere"
def resolve_model_file(spec: ModelSpec, override: str | None) -> Path: ...
```

```python
# segments.py  (all frame arithmetic in one place)
@dataclass(frozen=True)
class Span:
    start: int  # frames
    end: int

def plan_chunks(total: int, chunk: int, overlap: int) -> list[Span]: ...
def plan_halves(total: int, overlap: int, min_frames: int) -> tuple[Span, Span]: ...
def write_span(source: Path, span: Span, dest: Path) -> Path: ...
```

```python
# engine.py
@dataclass(frozen=True)
class SegmentResult:
    path: Path
    text: str | None
    error: str | None

    @property
    def truncated(self) -> bool: return bool(self.error and self.error.startswith("output truncated:"))

class Engine(Protocol):
    def transcribe(self, wavs: Sequence[Path], *, batch_size: int,
                   language: str | None, workdir: Path) -> Iterator[SegmentResult]: ...

class TranscribeCpp:                    # the only Engine implementation
    def __init__(self, binary: Path, model_file: Path) -> None: ...
    def require_vulkan_gpu(self) -> None: ...
    def transcribe(...) -> Iterator[SegmentResult]: ...   # streams; raises EngineError on non-zero exit
```

```python
# reporting.py
class Reporter(Protocol):
    def file_started(self, path: Path, audio_format: str) -> None: ...
    def conversion_started(self, path: Path) -> None: ...
    def conversion_finished(self, elapsed: float) -> None: ...
    def inference_started(self, model: str, chunks: int) -> None: ...
    def chunk_finished(self) -> None: ...
    def inference_finished(self) -> None: ...
    def truncation_retry(self, wav: Path) -> None: ...
    def unaligned_overlaps(self, path: Path, count: int, *, retry: bool) -> None: ...
    def file_finished(self, result: FileResult) -> None: ...
```

```python
# pipeline.py
@dataclass(frozen=True)
class FileResult:
    source: Path
    output: Path
    audio_seconds: float
    inference_seconds: float
    unaligned_overlaps: int

    @property
    def speedup(self) -> float: ...

class Transcriber:
    def __init__(self, engine: Engine, model: ModelSpec, *, language: str | None,
                 batch_size: int, reporter: Reporter,
                 clock: Callable[[], float] = time.perf_counter) -> None: ...
    def transcribe_file(self, source: Path, output: Path) -> FileResult: ...
```

### 1.5 Non-goals

These are deliberately left out:

- **No plugin/backend framework.** The `Engine` protocol exists as a test seam, not to prepare for speculative backends. There will be one implementation.
- **No async, threads or parallel files.** One GPU and serial files is the correct model for this hardware.
- **No config file.** CLI flags plus two environment variables are enough.
- **No behaviour changes during the refactor.** Transcript bytes, output paths, and stdout/stderr wording stay identical until Phase 8, where the few deliberate changes are listed and made in separate commits.

---

## 2. Diagnosis: why `main.py` has to change

Everything cited here is in `main.py` at commit `7568477`.

| # | Problem | Evidence | Consequence |
|---|---|---|---|
| D1 | **Model knowledge is spread across 4 places.** | `MODEL_FILES` (L26), `CHUNK_SECONDS` (L36), the `if` chain in `resolve_language` (L69–80), and `model in {"parakeet", "nemotron"}` inside `transcribe` (L371). | Adding Nemotron (`c3d8cd3`) meant edits in four unrelated places. It is easy to miss one, and nothing fails when you do. |
| D2 | **`transcribe()` has at least 7 responsibilities.** | L367–408 writes the batch file, applies model and equal-length batch policies, builds the command, runs it, parses JSONL, classifies errors, retries truncations through recursion, and stitches. | You can't test or change one policy without the other six. The recursive retry depends on `directory` and `model` being threaded through. |
| D3 | **JSONL is parsed twice.** | `run_batch` parses each line for progress (L348–356), then `transcribe` parses the full buffered stdout again (L387). | The two parsers can drift apart (for example, one skips rows without `"file"` and the other raises). The streaming is also wasted, because all stdout is buffered anyway. |
| D4 | **I/O is mixed into logic.** | `print` calls in `convert_and_chunk` (L167, L174), `transcribe` (L401, L405), `main`, and `ChunkProgress`. | Tests have to capture stdout to assert on logic, and the conversion code decides how progress looks. |
| D5 | **The same frame arithmetic appears twice with different formulas.** | Chunk overlap is `min(3s, chunk//2)` (L184). Split overlap is `min(3s, frames//4)`, halved on each side (L274–275). WAV writing is also duplicated (L192–196 vs L281–283). | The overlap invariant that `stitch_chunks` relies on is implicit and defined twice. |
| D6 | **Tests are coupled to private structure.** | `test_main.py` patches `cli.run_batch`, `cli.convert_and_chunk`, `cli.transcribe`, `cli.subprocess.run`, and `cli.time.perf_counter` with an exact call count (`side_effect=[0.0, 1.0, 2.0, 3.0]`). | Every move or rename in the refactor breaks the safety net that is meant to protect it. |
| D7 | **Build provisioning happens at runtime, in a Nix project.** | `ensure_binary` (L83–106) runs git and cmake with a hard-coded `-j 4` against whatever toolchain is on `PATH`. | It is not hermetic, and the pin is duplicated outside Nix. There is also a latent bug: if `git fetch` fails after `git init`, `source/` still exists, so every later run skips the fetch and cmake fails forever until the cache is deleted by hand. |
| D8 | **There is no error boundary.** | `main()` catches nothing, and validation raises bare `ValueError`/`FileNotFoundError`/`RuntimeError`. `--batch-size` is validated after parsing (L413). | Users see Python tracebacks for ordinary mistakes (a typo in a path, ffmpeg not installed, no GPU), and every failure exits with code 1. |
| D9 | **It isn't a package.** | There is no `pyproject.toml`, it runs as `python main.py`, and the flake only offers a devShell. | It can't be installed, `nix run` doesn't work, and there's nowhere to configure ruff, mypy or pytest. |
| D10 | **Output path policy is implicit.** | `input_file.with_suffix(".txt")` appears inline in `main` (L434). | `a.mp3` and `a.wav` both write `a.txt`, and the second silently overwrites the first. That's fine to keep for now, but it should be a named, tested policy. |
| D11 | **Some repo content is stale.** | The flake description says "ROCm", but only Vulkan is used. `AMD_PERFORMANCE.md` describes PyTorch and recommends `--batch-size 8`, which contradicts the current default of 1. The README says `inputs/` is included, but it's gitignored. There's an empty `.codex`, an empty `.agents/` and an empty `scripts/`, and the sample `.aac` fixtures sit in the root. | New contributors (human or agent) get contradictory instructions. |

The pure algorithmic parts (`text_tokens`, `overlap_offset`, `stitch_chunks`) are good code. They are already pure, and the migration should move them as they are.

---

## 3. Migration plan

**Ground rules for every phase**

- Each phase is one PR, or a small series of commits, that leaves `nix develop -c pytest` and `git diff --check` green.
- The behaviour parity checklist (§4) must pass after every phase.
- Code moves in one commit and changes in a later one. Never move and modify in the same commit, so `git log -M` and review stay clear.

### Phase 0 — Build a safety net at the process boundary

**Change**

1. Add `tests/test_cli_e2e.py`. It runs the CLI as a subprocess with:
   - `TRANSCRIBE_CPP_BIN` pointing to a fake engine: a small Python script in `tests/conftest.py`. It answers `--list-devices` with `kind=vulkan type=igpu`, reads the `--batch` file, and emits `{"file": ..., "text": ...}` JSONL. You can script its text per file, including overlapping text across chunks and `output truncated:` errors.
   - `--model-id <tmp>/dummy.gguf`, a local empty file, so there is no Hugging Face access.
   - WAV inputs generated in `tmp_path`: 16 kHz mono for the passthrough case, 44.1 kHz stereo for the conversion case (needs ffmpeg, which the devShell has), and a 95 s file so that chunking and overlap happen.
2. Assert on transcript bytes, output paths, exit codes, and the stdout/stderr lines listed in §4.
3. Keep `test_main.py` running unchanged for now.

**Why**

D6: the existing tests can't survive the refactor. A black-box test that only knows the CLI contract (argv, environment, files, streams) stays valid across every later phase. It is the only test that proves behaviour parity. The fake binary approach is already proven by `test_run_batch_streams_chunk_results`.

### Phase 1 — Packaging and tooling

**Change**

1. Add `pyproject.toml`: `name = "transcribe-cli"`, `requires-python = ">=3.13"`, dependency `huggingface-hub`, `[project.scripts] transcribe-cli = "transcribe_cli.cli:main"`, and `[tool.ruff]`, `[tool.mypy] strict = true`, `[tool.pytest.ini_options]`.
2. Create `src/transcribe_cli/` and move `main.py` there as `cli.py` unchanged. Leave a 3-line `main.py` shim at the root (`from transcribe_cli.cli import main; main()`) so `python main.py` keeps working until Phase 7.
3. Add `pytest`, `ruff` and `mypy` to the devShell Python.
4. Move `sample_*.aac` into `tests/fixtures/`.
5. Point `test_main.py` at `transcribe_cli.cli`.

**Why**

D9. The `src/` layout means tests run against the package and not whatever happens to be in the current directory. The console script is also what the Nix package in Phase 7 installs. pytest's `tmp_path` and `monkeypatch` fixtures replace the `tempfile.TemporaryDirectory()` and `patch.object` boilerplate that makes up much of `test_main.py`. Strict mypy is nearly free because the code is already fully annotated, and it protects the interfaces introduced next.

**Name collision:** the upstream binary is also called `transcribe-cli` (L92). With both on `PATH`, `which transcribe-cli` becomes ambiguous. I recommend naming this project's console script `transcribe` and keeping the repository and package name. This is a user-facing decision, so make it in this phase or explicitly defer it.

### Phase 2 — Extract the pure core: `stitch.py`, `segments.py`

**Change**

1. Move `join_chunks`, `text_tokens`, `overlap_offset` and `stitch_chunks` to `stitch.py` verbatim. Then, in a follow-up commit, name the magic numbers as module constants: `EDGE_TOKENS = 32`, `MIN_PHRASE = 3`, `MIN_HAN_PHRASE = 6`, `MAX_LEADING_MISMATCHES = 6`, `FUZZY_WORD_RATIO = 0.6`. Also pull the Han range check that appears twice (L231, L241) into `_is_han_phrase()`.
2. Create `segments.py` with `Span`, `plan_chunks`, `plan_halves` and `write_span`. Rewrite the loop in `convert_and_chunk` (L181–201) as `plan_chunks` + `write_span`, and `split_wav` as `plan_halves` + `write_span`. Keep both overlap formulas exactly as they are. Document the invariant they serve: *adjacent spans share ≥ the overlap `stitch.py` needs to find a phrase*.
3. Unit test both modules with plain lists and integers, without files, except for a single `write_span` round trip. Move the existing stitch tests here.

**Why**

D5. These are the parts most likely to be tuned (the recent commits were overlap and truncation work), and they are the easiest to test exhaustively. Once span planning is a pure function over frame counts, edge cases become one-line tests: the last chunk shorter than the overlap, a total exactly equal to the chunk length, the two-second retry floor. Today each of those needs a WAV file on disk.

### Phase 3 — `models.py`: one registry

**Change**

1. Introduce `ModelSpec` and `MODELS` as in §1.4. Populate them from `MODEL_FILES`, `CHUNK_SECONDS` and the rules in `resolve_language` and L371.
2. `resolve_language` becomes `ModelSpec.resolve_language`. Keep today's exact edge cases: Cohere passes `"auto"` through literally, Nemotron maps `"auto"` to unset, Parakeet accepts only `en` and passes nothing, and Qwen rejects any value.
3. Move `resolve_model_file` here. This is the only module that imports `huggingface_hub`.
4. Build argparse `choices` from `MODELS`, and generate the `--language` help text from the specs.
5. Delete `MODEL_FILES`, `CHUNK_SECONDS` and the literal model set in `transcribe`.

**Why**

D1. After this change, adding a model is one `ModelSpec(...)` entry plus a README row. The registry records *why* each quirk exists: `supports_batching=False` carries a comment tying it to the transcribe.cpp pin, so when the pin is bumped there is an obvious place to re-test it.

### Phase 4 — `media.py` and `inputs.py`

**Change**

1. `media.py` gets `describe(path) -> str` (from `describe_input`), `is_engine_ready_wav(path) -> bool` (from L156–163), and `normalize(source, workdir, reporter) -> Path`, which returns either the source or a converted `audio.wav`. It also gets `read_frame_count(wav)`. Chunking moves out of this function: `convert_and_chunk` becomes `media.normalize` followed by `segments.plan_chunks`.
2. Missing `ffmpeg`/`ffprobe` (`FileNotFoundError` from `subprocess`) becomes `PrerequisiteError("ffmpeg not found on PATH")`. A non-zero exit becomes `MediaError` containing the tool's stderr.
3. `inputs.py` gets `expand(patterns) -> list[Path]`, `validate(path)`, `SUPPORTED_EXTENSIONS`, and `transcript_path(source) -> Path`, which keeps `with_suffix(".txt")` for now (see D10).

**Why**

`convert_and_chunk` currently answers two unrelated questions: "is this audio in the engine's format?" and "how should it be split for GPU memory?". The first depends on the engine and the second on the model, so they change for different reasons and belong in different modules. Wrapping the tool errors in the module that calls the tool is what makes rule 6 possible.

### Phase 5 — `engine.py`: the transcribe.cpp adapter

**Change**

1. `TranscribeCpp.require_vulkan_gpu()` comes from `require_vulkan_gpu`.
2. `TranscribeCpp.transcribe(...)` writes `batch.txt`, applies the engine's equal-length batching rule (L373–379), builds the command (L380–383), and **yields `SegmentResult`s as JSONL lines arrive**, using the existing `select`/`os.read` loop. It collects stderr into a temporary file as it does now. After the loop, a non-zero exit raises `EngineError` with the stderr tail. Any expected path missing from the output also raises `EngineError`.
3. Remove `ChunkProgress` from the engine. The caller advances progress for each yielded result.
4. Keep `supports_batching` out of the engine. The pipeline passes `batch_size=1` when the model disallows batching, so the engine doesn't need to know model names (rule 3).
5. Move `ensure_binary` to `provision.py` unchanged, and fix the D7 partial-checkout bug there: clone into `source.tmp/` and rename it to `source/` only after a successful checkout.

**Why**

D2 and D3. The engine becomes the only place that knows transcribe.cpp's flags, JSONL schema and quirks, so bumping `TRANSCRIBE_CPP_COMMIT` means re-reading one ~110-line file. A single streaming parser removes the double parse and makes progress a side effect of iteration instead of a separate channel. Engine tests keep using a fake binary, the same approach as today's `test_run_batch_streams_chunk_results`, so they check real process handling without a GPU.

### Phase 6 — `reporting.py` and `pipeline.py`

**Change**

1. `reporting.py` gets the `Reporter` protocol, a `ConsoleReporter` that reproduces today's messages exactly (stdout for status, stderr for progress and warnings), and `ChunkProgress`, with `rendered` initialised in `__init__` instead of read through `getattr(self, "rendered", -1)`.
2. `pipeline.Transcriber.transcribe_file` holds the body of the `main()` loop:
   `describe → normalize → plan_chunks → write_span → engine.transcribe (advancing progress) → resolve truncations → stitch → write output → FileResult`.
   It owns the temporary directory for each file.
3. Truncation retry is a private method, `_resolve_truncated(result) -> str`. It uses `segments.plan_halves` (which raises `TranscriptionError` below the two-second floor), runs the engine with `batch_size=1`, and recurses on its own halves. The recursion now covers only the truncated segment instead of the whole `transcribe()` call.
4. Pass the clock in. `FileResult` carries `audio_seconds` and `inference_seconds`, and `ConsoleReporter.file_finished` formats `"Metrics for …: …s audio; batch completed in …s (…x)"`.
5. `test_pipeline.py` uses a `FakeEngine` (a list of scripted `SegmentResult`s), a `RecordingReporter` (a list of calls), and a fake clock. Remove the `patch.object` calls.

**Why**

D2, D4 and D6. The pipeline expresses the product's workflow in about 60 readable lines with no subprocess or formatting details. Constructor injection turns the eight-deep `patch.object` stacks in `test_main.py` (L264–273, L290–296, L314–322) into plain arguments. That makes the tests shorter and lets them survive renames. Keeping the reporter as a protocol means a future `--quiet` or JSON-lines output is a new class, with no edits to the pipeline.

### Phase 7 — `cli.py`, errors, and retiring `main.py`

**Change**

1. `errors.py`: `TranscribeCliError` is the base class, with `exit_code`. The subclasses are `UsageError` (2), `PrerequisiteError` (3: no Vulkan GPU, missing ffmpeg, missing binary), `MediaError` (4), and `EngineError`/`TranscriptionError` (5).
2. `cli.main(argv: Sequence[str] | None = None) -> int` parses the arguments with a `positive_int` argparse type for `--batch-size`, resolves the model and language, expands and validates inputs, provisions the binary, checks the GPU, builds `Transcriber(TranscribeCpp(...), ConsoleReporter())`, and loops over the files. It catches `TranscribeCliError`, prints `error: <message>` to stderr, and returns the exit code. It also handles `KeyboardInterrupt` by printing a newline and returning 130.
3. Delete `main.py` and `test_main.py`, since every test has been migrated. Update README usage (`transcribe …`/`nix run . -- …`) and `AGENTS.md` (`pytest`, `ruff check`, `mypy`).

**Why**

D8. `main(argv)` returning an int makes the CLI callable in-process from tests, and it gives users a clean message instead of a traceback. Different exit codes let scripts tell "fix your arguments" apart from "your GPU disappeared". Keeping `main.py` until this phase meant nobody's workflow broke partway through the migration.

### Phase 8 — Move provisioning into Nix

**Change**

1. In `flake.nix`, add `packages.transcribe-cpp`: a `stdenv.mkDerivation` using `fetchFromGitHub { owner = "handy-computer"; repo = "transcribe.cpp"; rev = "c83df3f…"; hash = …; fetchSubmodules = <as required>; }`, `cmakeFlags = [ "-DTRANSCRIBE_VULKAN=ON" ]`, and `nativeBuildInputs = [ cmake shaderc ]` / `buildInputs = [ vulkan-headers vulkan-loader ]`.
2. Add `packages.default`: `python313Packages.buildPythonApplication` built from `pyproject.toml`. Use `makeWrapperArgs` to set `TRANSCRIBE_CPP_BIN` and prefix `PATH` with `ffmpeg`.
3. Add `apps.default` so that `nix run . -- recording.wav` works.
4. Add `checks.default`, which runs pytest, ruff and mypy, so `nix flake check` is the single CI entry point.
5. In the devShell, export `TRANSCRIBE_CPP_BIN` from the package, drop `cmake` and `git` from the runtime needs, and fix the "ROCm" description.
6. Reduce `provision.py` to `TRANSCRIBE_CPP_BIN`, then `shutil.which("transcribe-cli")`, then `PrerequisiteError`. Delete the runtime git/cmake build.

**Why**

D7. The README describes the project as a "Nix-managed CLI" for x86_64 NixOS. Building C++ at runtime from Python repeats what Nix does better: a hermetic toolchain, a content-addressed pin, parallel builds, binary caching, and no broken caches. The pinned commit then lives in exactly one place, the flake. The first run no longer spends minutes compiling in the middle of a transcription.

*If non-Nix users are a real audience, keep the runtime build in `provision.py` behind an explicit `transcribe --build-engine` subcommand instead of a silent first-run side effect. Otherwise delete it.*

### Phase 9 — Repository hygiene and deliberate behaviour changes

Make each item below as its own commit, after parity has been proven:

1. **Docs:** move the accuracy check to `docs/accuracy-check.md`, and fix its claim that `inputs/` is in the repo, or ship a small committed fixture instead. Move `AMD_PERFORMANCE.md` to `docs/history/` with a header saying it measured the PyTorch implementation and its `--batch-size 8` advice doesn't apply. Add `docs/architecture.md` with the rules from §1.2–1.3.
2. **Remove clutter:** delete the empty `.codex`, `.agents/` and `scripts/`, and the stray `__pycache__`.
3. **Output collisions (D10):** make `inputs.transcript_path` detect two inputs that map to the same `.txt` and raise a `UsageError` before any GPU work. This is a behaviour change: it turns silent data loss into an error. Optionally add `--output-dir`.
4. **TODO.md:** the open item ("skip chunking when the model accepts the whole file") now reduces to a `max_single_call_seconds` field on `ModelSpec` and one branch in `pipeline`. That shows the new structure is paying for itself.

---

## 4. Behaviour parity checklist

`test_cli_e2e.py` (Phase 0) must assert all of these, and they must hold after every phase up to 8:

- [ ] `--model` choices are `cohere|qwen|parakeet|nemotron`, with `cohere` as the default.
- [ ] Language rules: Cohere defaults to `-l en` and passes any hint, including `auto`. Nemotron passes no `-l` for unset or `auto`, and passes other values through. Parakeet accepts only `en` and passes nothing. Qwen rejects any `--language`.
- [ ] Engine command: `-m <gguf> --backend vulkan --batch <file> --batch-jsonl --batch-size N --timestamps none [-l L]`.
- [ ] Batch size is forced to 1 for Parakeet and Nemotron, and for chunks of mixed length.
- [ ] A 16 kHz mono 16-bit PCM WAV is passed through with no "Converting" line. Other inputs print the "Converting…" and "Conversion completed in" lines.
- [ ] Chunk lengths are 30/60/300/60 s. Adjacent chunks overlap by 3 s (capped at chunk/2).
- [ ] A truncated chunk is split in half with overlap and only that chunk is retried. A segment of 2 s or less that is still truncated is an error.
- [ ] Overlap stitching output is byte-identical on the existing stitch test corpus. The unaligned-overlap warning text is unchanged.
- [ ] Output is `<input>.with_suffix(".txt")` with a trailing newline.
- [ ] stdout lines: `Processing file:`, `Input format:`, `Running transcribe.cpp with … on N audio chunk(s)...`, `Wrote transcript to …`, `Metrics for …: …s audio; batch completed in …s (…x)`, printed per file as each one completes.
- [ ] Progress goes to stderr: a resizable bar on a TTY, and `Chunks processed: i/N` lines otherwise.
- [ ] If no Vulkan igpu/dgpu is listed, the run fails before any file is processed.

---

## 5. Risk and sequencing notes

- **Highest risk:** Phase 5, because it changes the streaming parser that feeds both progress and results. The Phase 0 fake binary must cover interleaved stderr, a partial last line with no trailing newline, a non-JSON stdout line, and a non-zero exit.
- **Parallelisable:** Phases 2, 3 and 4 touch disjoint code, so they can merge in any order after Phase 1.
- **Stop points:** after Phase 7 the codebase is fully re-architected. Phases 8 and 9 are independent improvements, and each can be dropped without leaving things half done.
- **Effort:** about 2–3 focused days in total. Phase 0 and Phase 6 take most of it.
