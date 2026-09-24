# Agent quick start

- Run project commands in `nix develop` (or `nix develop -c ...`). The shell supplies Python, ROCm PyTorch, Transformers, and FFmpeg.
- `main.py` is the CLI. Cohere is the default (`--model cohere`); `--model qwen` uses the official Transformers-native `Qwen/Qwen3-ASR-0.6B-hf` checkpoint. The 1.7B checkpoint remains available through `--model-id`.
- Keep Qwen's default language unset so English and Chinese can be detected in the same audio. Both models use `resolve_runtime_config()` for GPU or CPU selection.
- Run `python -m unittest -v test_main` and `git diff --check` in the Nix shell after changes.
- Read `README.md` for usage, model-specific behavior, and the separate Q4 GGUF option.
