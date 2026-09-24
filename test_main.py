# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import main as cli


class CliTests(unittest.TestCase):
    def test_existing_options_and_new_parakeet(self):
        with patch.object(sys, "argv", ["main.py", "audio.wav"]):
            args = cli.parse_args()
        self.assertEqual(args.model, "cohere")
        self.assertIsNone(args.language)
        with patch.object(sys, "argv", ["main.py", "audio.wav", "--model", "parakeet",
                                        "--model-id", "custom.gguf", "--batch-size", "2"]):
            args = cli.parse_args()
        self.assertEqual((args.model, args.model_id, args.batch_size), ("parakeet", "custom.gguf", 2))
        self.assertIn("nemotron", cli.MODEL_FILES)

    def test_model_repositories_and_languages(self):
        self.assertEqual(cli.MODEL_FILES["qwen"][0], "handy-computer/Qwen3-ASR-1.7B-gguf")
        self.assertEqual(cli.resolve_language("cohere", None), "en")
        self.assertIsNone(cli.resolve_language("qwen", None))
        self.assertIsNone(cli.resolve_language("nemotron", None))
        self.assertIsNone(cli.resolve_language("nemotron", "auto"))
        with self.assertRaisesRegex(ValueError, "automatic language"):
            cli.resolve_language("qwen", "zh")
        with self.assertRaisesRegex(ValueError, "only supports English"):
            cli.resolve_language("parakeet", "fr")

    def test_requires_vulkan_gpu(self):
        with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "kind=cpu", "")):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                cli.require_vulkan_gpu(Path("transcribe-cli"))
        with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "kind=vulkan  type=igpu", "")):
            cli.require_vulkan_gpu(Path("transcribe-cli"))

    def test_batch_invocation_and_error_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wav = root / "audio.wav"
            output = "\n".join([json.dumps({"type": "batch_header"}), json.dumps({"file": str(wav), "text": "Hello."})])
            with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output, "")) as run:
                result = cli.transcribe(Path("transcribe-cli"), Path("model.gguf"), [wav], None, 2, root)
            self.assertEqual(result[str(wav)], "Hello.")
            self.assertIn("vulkan", run.call_args.args[0])
            self.assertIn("--batch-size", run.call_args.args[0])
            error = json.dumps({"file": str(wav), "text": "", "error": "unsupported language"})
            with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, error, "GPU error")):
                with self.assertRaisesRegex(RuntimeError, "GPU error"):
                    cli.transcribe(Path("transcribe-cli"), Path("model.gguf"), [wav], None, 1, root)

    def test_mixed_length_chunks_use_serial_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wavs = [root / "a.wav", root / "b.wav"]
            for path, frames in zip(wavs, [10, 20]):
                with wave.open(str(path), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(16000)
                    writer.writeframes(b"\0\0" * frames)
            output = "\n".join(json.dumps({"file": str(path), "text": "ok"}) for path in wavs)
            with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output, "")) as run:
                cli.transcribe(Path("transcribe-cli"), Path("model.gguf"), wavs, None, 2, root)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--batch-size") + 1], "1")

    def test_parakeet_uses_serial_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wav = root / "a.wav"
            output = json.dumps({"file": str(wav), "text": "ok"})
            with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output, "")) as run:
                cli.transcribe(Path("transcribe-cli"), Path("model.gguf"), [wav], None, 2, root, model="parakeet")
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--batch-size") + 1], "1")

    def test_globs_and_chunk_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.wav"
            second = root / "b.wav"
            first.touch()
            second.touch()
            self.assertEqual(cli.expand_input_paths([str(root / "*.wav"), str(first)]), [first, second])
        self.assertEqual(cli.join_chunks(["你好", "世界", "Hello"]), "你好世界 Hello")

    def test_ffmpeg_conversion_and_chunking(self):
        source = Path(__file__).with_name("sample_15s.aac")
        with tempfile.TemporaryDirectory() as temporary:
            chunks, duration = cli.convert_and_chunk(source, Path(temporary), 5)
            self.assertGreater(duration, 15)
            self.assertEqual(len(chunks), 4)
            for chunk in chunks:
                self.assertTrue(chunk.is_file())

    def test_writes_transcript_next_to_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_file = Path(temporary) / "audio.wav"
            input_file.touch()
            with (
                patch.object(sys, "argv", ["main.py", str(input_file)]),
                patch.object(cli, "ensure_binary", return_value=Path("transcribe-cli")),
                patch.object(cli, "require_vulkan_gpu"),
                patch.object(cli, "resolve_model_file", return_value=Path("model.gguf")),
                patch.object(cli, "convert_and_chunk", side_effect=lambda path, directory, seconds: ([directory / "audio.wav"], 1.0)),
                patch.object(cli, "transcribe", side_effect=lambda binary, model_file, wavs, language, batch, directory, model: {str(wavs[0]): "Transcript"}),
            ):
                cli.main()
            self.assertEqual(input_file.with_suffix(".txt").read_text(), "Transcript\n")


if __name__ == "__main__":
    unittest.main()
