# SPDX-License-Identifier: Apache-2.0
"""User-facing failures. `cli.main` turns these into one line and an exit code."""

from __future__ import annotations


class TranscribeCliError(Exception):
    exit_code = 1


class UsageError(TranscribeCliError):
    exit_code = 2


class PrerequisiteError(TranscribeCliError):
    """No Vulkan GPU, missing ffmpeg, or missing transcribe.cpp binary."""

    exit_code = 3


class MediaError(TranscribeCliError):
    exit_code = 4


class EngineError(TranscribeCliError):
    exit_code = 5


class TranscriptionError(TranscribeCliError):
    exit_code = 5


class OutputError(TranscribeCliError):
    exit_code = 6
