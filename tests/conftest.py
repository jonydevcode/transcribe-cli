# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers: a fake transcribe.cpp binary, WAV factory, dummy GGUF."""

from __future__ import annotations

import json
import stat
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

FAKE_ENGINE = '''\
#!/usr/bin/env python3
import json, os, sys, wave

config = json.load(open(os.environ["FAKE_ENGINE_CONFIG"]))
argv = sys.argv[1:]
if argv == ["--list-devices"]:
    print(config.get("devices", "Vulkan0: kind=vulkan type=igpu name=fake"))
    sys.exit(0)
batch = argv[argv.index("--batch") + 1]
paths = open(batch, encoding="utf-8").read().splitlines()
with open(config["log"], "a", encoding="utf-8") as log:
    log.write(json.dumps({"argv": argv, "paths": paths}) + "\\n")
print(json.dumps({"type": "batch_header"}), flush=True)
for path in paths:
    name = os.path.basename(path)
    with wave.open(path, "rb") as reader:
        seconds = reader.getnframes() / reader.getframerate()
    limit = config.get("truncate_longer_than")
    if limit is not None and seconds > limit:
        row = {"file": path, "error": "output truncated: decode hit the context/generation cap"}
    elif name in config.get("errors", {}):
        row = {"file": path, "text": "", "error": config["errors"][name]}
    else:
        row = {"file": path, "text": config.get("texts", {}).get(name, config.get("default", "ok"))}
    print(json.dumps(row), flush=True)
    print("progress " + name, file=sys.stderr, flush=True)
if config.get("junk_line"):
    print("not json at all", flush=True)
if config.get("partial_last_line"):
    sys.stdout.write(json.dumps({"type": "footer"}))
    sys.stdout.flush()
sys.exit(config.get("exit", 0))
'''


def write_wav(path: Path, seconds: float, rate: int = 16000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\0\0" * channels * int(seconds * rate))
    return path


@dataclass
class FakeEngineBinary:
    path: Path
    config_path: Path
    log_path: Path
    config: dict[str, object] = field(default_factory=dict)

    def configure(self, **values: object) -> None:
        self.config.update(values)
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def calls(self) -> list[dict[str, list[str]]]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def env(self) -> dict[str, str]:
        return {"TRANSCRIBE_CPP_BIN": str(self.path), "FAKE_ENGINE_CONFIG": str(self.config_path)}


@pytest.fixture
def fake_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeEngineBinary:
    directory = tmp_path / "engine"
    directory.mkdir()
    binary = directory / "transcribe-cli"
    binary.write_text(f"#!{__import__('sys').executable}\n" + FAKE_ENGINE.split("\n", 1)[1], encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    engine = FakeEngineBinary(binary, directory / "config.json", directory / "calls.jsonl")
    engine.configure(log=str(engine.log_path))
    monkeypatch.setenv("FAKE_ENGINE_CONFIG", str(engine.config_path))  # for in-process use
    return engine


@pytest.fixture
def dummy_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "dummy.gguf"
    path.touch()
    return path


@pytest.fixture
def wav_factory(tmp_path: Path) -> Callable[..., Path]:
    def make(name: str, seconds: float, rate: int = 16000, channels: int = 1) -> Path:
        return write_wav(tmp_path / name, seconds, rate, channels)

    return make


class RecordingReporter:
    """A Reporter that records `(method, args...)` tuples."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def names(self) -> list[str]:
        return [str(call[0]) for call in self.calls]

    def file_started(self, path: Path, audio_format: str) -> None:
        self.calls.append(("file_started", path, audio_format))

    def conversion_started(self, path: Path) -> None:
        self.calls.append(("conversion_started", path))

    def conversion_finished(self, elapsed: float) -> None:
        self.calls.append(("conversion_finished", elapsed))

    def inference_started(self, model: str, chunks: int) -> None:
        self.calls.append(("inference_started", model, chunks))

    def tick(self) -> None:
        pass

    def chunk_finished(self) -> None:
        self.calls.append(("chunk_finished",))

    def inference_finished(self) -> None:
        self.calls.append(("inference_finished",))

    def truncation_retry(self, wav: Path) -> None:
        self.calls.append(("truncation_retry", wav))

    def unaligned_overlaps(self, path: Path, count: int, *, retry: bool) -> None:
        self.calls.append(("unaligned_overlaps", path, count, retry))

    def file_finished(self, result: object) -> None:
        self.calls.append(("file_finished", result))
