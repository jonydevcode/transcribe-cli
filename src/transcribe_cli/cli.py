# SPDX-License-Identifier: Apache-2.0
"""Argument parsing, wiring and exit codes."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from transcribe_cli import inputs
from transcribe_cli.engine import TranscribeCpp, require_vulkan_gpu
from transcribe_cli.errors import TranscribeCliError
from transcribe_cli.models import DEFAULT_MODEL, MODELS, language_help, resolve_model_file
from transcribe_cli.pipeline import Transcriber
from transcribe_cli.provision import locate_binary
from transcribe_cli.reporting import ConsoleReporter


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transcribe",
                                     description="Transcribe media files on a GPU and write INPUT.txt.")
    parser.add_argument("input_paths", nargs="+", help="Source media paths or glob patterns.")
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL,
                        help=f"Model family (default: {DEFAULT_MODEL}).")
    parser.add_argument("--model-id", default=None,
                        help="Override Hugging Face GGUF repository, or use a local .gguf file.")
    parser.add_argument("--language", default=None, help=language_help())
    parser.add_argument("--batch-size", type=positive_int, default=1,
                        help="Audio chunks processed together (default: 1).")
    return parser


def run(argv: Sequence[str] | None) -> int:
    args = build_parser().parse_args(argv)
    spec = MODELS[args.model]
    language = spec.resolve_language(args.language)
    sources = inputs.expand(args.input_paths)
    for source in sources:
        inputs.validate(source)
    outputs = inputs.transcript_paths(sources)
    binary = locate_binary()
    require_vulkan_gpu(binary)
    model_file = resolve_model_file(spec, args.model_id)
    transcriber = Transcriber(TranscribeCpp(binary, model_file), spec, language=language,
                              batch_size=args.batch_size, reporter=ConsoleReporter(model_file))
    for source, output in zip(sources, outputs, strict=True):
        transcriber.transcribe_file(source, output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except TranscribeCliError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
