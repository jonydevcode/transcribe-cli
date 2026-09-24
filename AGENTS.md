# Agent quick start

- Run project commands in `nix develop` (or `nix develop -c ...`). The shell supplies Python, Hugging Face Hub, FFmpeg, CMake, and Vulkan build tools.
- `main.py` is the CLI. Cohere is the default; `--model qwen`, `--model parakeet`, and the retained `--model nemotron` use handy-computer GGUF checkpoints through transcribe.cpp.
- The CLI requires a Vulkan GPU and refuses CPU fallback. On first run it builds a pinned transcribe.cpp checkout in the user cache. `TRANSCRIBE_CPP_BIN` can point to a prebuilt Vulkan CLI.
- Keep Qwen's default language unset so English and Chinese can be detected in the same audio. Nemotron's auto mode also requires passing no language hint to transcribe.cpp.
- Run `python -m unittest -v test_main` and `git diff --check` in the Nix shell after changes.
- Read `README.md` for usage, model behavior, and the Cohere accuracy comparison.
