# SPDX-License-Identifier: Apache-2.0
"""transcribe.cpp adapter: device check, command line and streamed JSONL results."""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import tempfile
import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from transcribe_cli.errors import EngineError, PrerequisiteError

TRUNCATION_PREFIX = "output truncated:"
STDERR_TAIL = 3000
STDOUT_TAIL = 1000
POLL_SECONDS = 0.2


@dataclass(frozen=True)
class SegmentResult:
    path: Path
    text: str | None
    error: str | None

    @property
    def truncated(self) -> bool:
        return bool(self.error and self.error.startswith(TRUNCATION_PREFIX))


class Engine(Protocol):
    def transcribe(self, wavs: Sequence[Path], *, batch_size: int, language: str | None,
                   workdir: Path, on_idle: Callable[[], None] | None = None) -> Iterator[SegmentResult]:
        """Yield one result per WAV as soon as the engine reports it.

        `on_idle` is called while the engine is silent so callers can refresh a display.
        """
        ...


def require_vulkan_gpu(binary: Path) -> None:
    """Fail unless transcribe.cpp lists a Vulkan GPU. Callable before a model is resolved."""
    try:
        result = subprocess.run([str(binary), "--list-devices"], capture_output=True, text=True)
    except OSError as error:
        raise PrerequisiteError(f"cannot run transcribe.cpp at {binary}: {error}") from error
    if result.returncode:
        raise EngineError(f"transcribe.cpp --list-devices failed:\n{result.stderr[-STDERR_TAIL:]}")
    if not re.search(r"kind=vulkan\s+type=(?:igpu|dgpu)", result.stdout):
        raise PrerequisiteError("transcribe.cpp did not find a Vulkan GPU; CPU fallback is disabled.")


class TranscribeCpp:
    def __init__(self, binary: Path, model_file: Path) -> None:
        self.binary = binary
        self.model_file = model_file

    def require_vulkan_gpu(self) -> None:
        require_vulkan_gpu(self.binary)

    def command(self, batch_file: Path, batch_size: int, language: str | None) -> list[str]:
        command = [str(self.binary), "-m", str(self.model_file), "--backend", "vulkan", "--batch",
                   str(batch_file), "--batch-jsonl", "--batch-size", str(batch_size), "--timestamps", "none"]
        if language:
            command.extend(["-l", language])
        return command

    def transcribe(self, wavs: Sequence[Path], *, batch_size: int, language: str | None,
                   workdir: Path, on_idle: Callable[[], None] | None = None) -> Iterator[SegmentResult]:
        batch_file = workdir / "batch.txt"
        batch_file.write_text("".join(f"{wav}\n" for wav in wavs), encoding="utf-8")
        if batch_size > 1 and len(wavs) > 1 and len({_frame_count(wav) for wav in wavs}) > 1:
            batch_size = 1  # batching needs equal-length inputs
        expected = {str(wav) for wav in wavs}
        reported: set[str] = set()
        stdout_tail = ""
        command = self.command(batch_file, batch_size, language)
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
                assert process.stdout is not None
                pending = b""
                while True:
                    ready, _, _ = select.select([process.stdout], [], [], POLL_SECONDS)
                    if not ready:
                        if on_idle:
                            on_idle()
                        continue
                    data = os.read(process.stdout.fileno(), 65536)
                    if not data:
                        break
                    pending += data
                    *complete, pending = pending.split(b"\n")
                    for raw in complete:
                        stdout_tail = (stdout_tail + raw.decode("utf-8", errors="replace") + "\n")[-STDOUT_TAIL:]
                        result = _parse_row(raw, expected, reported)
                        if result:
                            yield result
                if pending:
                    stdout_tail = (stdout_tail + pending.decode("utf-8", errors="replace"))[-STDOUT_TAIL:]
                    result = _parse_row(pending, expected, reported)
                    if result:
                        yield result
                returncode = process.wait()
            errors.seek(0)
            stderr = errors.read()
        if returncode:
            raise EngineError(f"transcribe.cpp GPU run failed:\n{stderr[-STDERR_TAIL:]}\n{stdout_tail}")
        if reported != expected:
            raise EngineError("transcribe.cpp did not return one transcript for each audio chunk")


def _frame_count(wav: Path) -> int:
    with wave.open(str(wav), "rb") as reader:
        return reader.getnframes()


def _parse_row(raw: bytes, expected: set[str], reported: set[str]) -> SegmentResult | None:
    line = raw.decode("utf-8", errors="replace")
    if not line.startswith("{"):
        return None
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    path = row.get("file")
    if path not in expected or path in reported:
        return None
    reported.add(path)
    error = row.get("error")
    return SegmentResult(Path(path), None if error else row.get("text", ""), error or None)
