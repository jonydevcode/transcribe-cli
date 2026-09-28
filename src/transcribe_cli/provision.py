# SPDX-License-Identifier: Apache-2.0
"""Locate the transcribe.cpp binary. The Nix flake builds and pins it."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from transcribe_cli.errors import PrerequisiteError

ENV_VAR = "TRANSCRIBE_CPP_BIN"
BINARY_NAME = "transcribe-cli"


def locate_binary() -> Path:
    override = os.environ.get(ENV_VAR)
    if override:
        binary = Path(override).expanduser().resolve()
        if not binary.is_file():
            raise PrerequisiteError(f"{ENV_VAR} does not exist: {binary}")
        return binary
    found = shutil.which(BINARY_NAME)
    if found:
        return Path(found)
    raise PrerequisiteError(
        f"transcribe.cpp binary not found: set {ENV_VAR} or run through `nix run` / `nix develop`")
