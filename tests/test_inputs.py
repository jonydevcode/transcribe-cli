# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest

from transcribe_cli import inputs
from transcribe_cli.errors import UsageError


def test_expand_globs_dedupes_and_keeps_order(tmp_path: Path) -> None:
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    first.touch()
    second.touch()
    assert inputs.expand([str(tmp_path / "*.wav"), str(first)]) == [first, second]


def test_expand_keeps_unmatched_pattern(tmp_path: Path) -> None:
    assert inputs.expand([str(tmp_path / "missing.wav")]) == [tmp_path / "missing.wav"]


def test_validate(tmp_path: Path) -> None:
    good = tmp_path / "a.WAV"
    good.touch()
    inputs.validate(good)
    with pytest.raises(UsageError, match="not found"):
        inputs.validate(tmp_path / "nope.wav")
    bad = tmp_path / "a.txt"
    bad.touch()
    with pytest.raises(UsageError, match="Unsupported input format: .txt"):
        inputs.validate(bad)


def test_transcript_path() -> None:
    assert inputs.transcript_path(Path("/x/talk.mp3")) == Path("/x/talk.txt")


def test_transcript_paths_reject_collisions() -> None:
    assert inputs.transcript_paths([Path("a.mp3"), Path("b.wav")]) == [Path("a.txt"), Path("b.txt")]
    with pytest.raises(UsageError, match=r"a\.mp3 and a\.wav would both write a\.txt"):
        inputs.transcript_paths([Path("a.mp3"), Path("a.wav")])


def test_expand_expands_home_before_globbing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "a.wav").touch()
    assert inputs.expand(["~/*.wav"]) == [tmp_path / "a.wav"]


def test_check_writable(tmp_path: Path) -> None:
    inputs.check_writable([tmp_path / "a.txt"])
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running with privileges that ignore directory permissions")
        with pytest.raises(UsageError, match="Cannot write transcript"):
            inputs.check_writable([locked / "a.txt"])
    finally:
        locked.chmod(0o700)
