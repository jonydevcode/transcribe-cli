# SPDX-License-Identifier: Apache-2.0
"""Input expansion, validation and output path policy."""

from __future__ import annotations

import glob
from collections.abc import Iterable, Sequence
from pathlib import Path

from transcribe_cli.errors import UsageError

SUPPORTED_EXTENSIONS = frozenset({".mp3", ".m4a", ".mp4", ".ogg", ".wav", ".flac", ".aac", ".webm"})


def expand(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in patterns:
        for match in glob.glob(raw) or [raw]:
            path = Path(match).expanduser().resolve()
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def validate(path: Path) -> None:
    if not path.is_file():
        raise UsageError(f"Input file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UsageError(f"Unsupported input format: {path.suffix}")


def transcript_path(source: Path) -> Path:
    return source.with_suffix(".txt")


def transcript_paths(sources: Sequence[Path]) -> list[Path]:
    """Output path for each source; two sources mapping to one transcript is a usage error."""
    outputs = [transcript_path(source) for source in sources]
    owners: dict[Path, Path] = {}
    for source, output in zip(sources, outputs, strict=True):
        if output in owners:
            raise UsageError(f"{owners[output]} and {source} would both write {output}")
        owners[output] = source
    return outputs
