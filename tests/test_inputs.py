# SPDX-License-Identifier: Apache-2.0
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
