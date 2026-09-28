# Agent quick start

- Run project commands in `nix develop` (or `nix develop -c ...`). The shell supplies Python, pytest, ruff, mypy, Hugging Face Hub, FFmpeg, and Vulkan tools, and exports `TRANSCRIBE_CPP_BIN` for the flake-built transcribe.cpp.
- The app is the `transcribe_cli` package in `src/`. Run it with `python -m transcribe_cli` in the shell or `nix run . -- FILE`. Read `docs/architecture.md` for the module map and rules. Cohere is the default model; `--model qwen`, `--model parakeet`, and the retained `--model nemotron` use handy-computer GGUF checkpoints through transcribe.cpp. Model facts live only in `models.py`.
- The CLI requires a Vulkan GPU and refuses CPU fallback. The flake pins and builds transcribe.cpp (Vulkan). `TRANSCRIBE_CPP_BIN` can point to a different prebuilt Vulkan CLI.
- Keep Qwen's default language unset so English and Chinese can be detected in the same audio. Nemotron's auto mode also requires passing no language hint to transcribe.cpp.
- Run `pytest`, `ruff check .`, `mypy`, and `git diff --check` in the Nix shell after changes (`nix flake check` runs the first three hermetically). Tests use a fake transcribe.cpp binary from `tests/conftest.py`; no GPU or network is needed.
- Read `README.md` for usage and model behavior, and `docs/accuracy-check.md` for the Cohere accuracy comparison.
