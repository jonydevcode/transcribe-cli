# SPDX-License-Identifier: Apache-2.0

import json
import io
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from contextlib import redirect_stdout
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

    def test_truncated_chunk_is_split_without_rerunning_successful_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wavs = [root / "good.wav", root / "truncated.wav"]
            for path in wavs:
                with wave.open(str(path), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(cli.SAMPLE_RATE)
                    writer.writeframes(b"\0\0" * (4 * cli.SAMPLE_RATE))
            calls = []

            def run(command, **kwargs):
                batch = Path(command[command.index("--batch") + 1]).read_text().splitlines()
                calls.append(batch)
                rows = []
                for path in batch:
                    if path == str(wavs[1]):
                        rows.append({"file": path, "error": "output truncated: decode hit the context/generation cap before end-of-stream"})
                    else:
                        rows.append({"file": path, "text": "first" if path == str(wavs[0]) else Path(path).stem[-1]})
                return subprocess.CompletedProcess(command, 0, "\n".join(json.dumps(row) for row in rows), "")

            with patch.object(cli.subprocess, "run", side_effect=run):
                result = cli.transcribe(Path("transcribe-cli"), Path("model.gguf"), wavs, None, 1, root)
            self.assertEqual(result, {str(wavs[0]): "first", str(wavs[1]): "0 1"})
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], [str(wav) for wav in wavs])
            self.assertTrue(all(Path(path).is_file() for path in calls[1]))

    def test_truncation_retry_stops_at_one_second(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wav = root / "short.wav"
            with wave.open(str(wav), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(cli.SAMPLE_RATE)
                writer.writeframes(b"\0\0" * cli.SAMPLE_RATE)
            error = json.dumps({"file": str(wav), "error": "output truncated: decode hit the context/generation cap before end-of-stream"})
            with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, error, "")):
                with self.assertRaisesRegex(RuntimeError, "still truncated audio shorter than two seconds"):
                    cli.transcribe(Path("transcribe-cli"), Path("model.gguf"), [wav], None, 1, root)

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

    def test_wav_passthrough_and_model_chunk_limits(self):
        self.assertEqual(cli.CHUNK_SECONDS, {"cohere": 30, "qwen": 60, "parakeet": 300, "nemotron": 60})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b"\0\0" * 32000)
            with patch.object(cli.subprocess, "run") as run:
                for model in cli.MODEL_FILES:
                    chunks, duration = cli.convert_and_chunk(source, root, cli.CHUNK_SECONDS[model])
                    self.assertEqual((chunks, duration), ([source], 2.0))
                run.assert_not_called()
            chunks, duration = cli.convert_and_chunk(source, root, 1)
            self.assertEqual((len(chunks), duration), (2, 2.0))
            self.assertTrue(all(chunk != source for chunk in chunks))

    def test_multiple_files_report_each_completion_and_speed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [root / "first.wav", root / "second.wav"]
            for path in inputs:
                path.touch()
            output = io.StringIO()
            def transcribe_file(binary, model_file, wavs, language, batch, directory, model):
                if wavs[0] == inputs[1]:
                    self.assertIn("first.wav: 2.00s audio; batch completed in 1.00s (2.00x)", output.getvalue())
                return {str(wavs[0]): "Transcript"}
            with (
                patch.object(sys, "argv", ["main.py", *(str(path) for path in inputs)]),
                patch.object(cli, "ensure_binary", return_value=Path("transcribe-cli")),
                patch.object(cli, "require_vulkan_gpu"),
                patch.object(cli, "resolve_model_file", return_value=Path("model.gguf")),
                patch.object(cli, "describe_input", return_value="16-bit PCM WAV, 16 kHz, mono"),
                patch.object(cli, "convert_and_chunk", side_effect=lambda path, directory, seconds: ([path], 2.0)),
                patch.object(cli, "transcribe", side_effect=transcribe_file),
                patch.object(cli.time, "perf_counter", side_effect=[0.0, 1.0, 2.0, 3.0]),
                redirect_stdout(output),
            ):
                cli.main()
            self.assertEqual(output.getvalue().count("(2.00x)"), 2)
            self.assertTrue(all(path.with_suffix(".txt").read_text() == "Transcript\n" for path in inputs))

    def test_progress_reports_format_conversion_and_transcription(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [root / "convert.wav", root / "ready.wav"]
            for path, rate in zip(inputs, [48000, 16000]):
                with wave.open(str(path), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(rate)
                    writer.writeframes(b"\0\0" * rate)
            output = io.StringIO()
            with (
                patch.object(sys, "argv", ["main.py", *(str(path) for path in inputs)]),
                patch.object(cli, "ensure_binary", return_value=Path("transcribe-cli")),
                patch.object(cli, "require_vulkan_gpu"),
                patch.object(cli, "resolve_model_file", return_value=Path("model.gguf")),
                patch.object(cli, "transcribe", side_effect=lambda binary, model_file, wavs, language, batch, directory, model: {str(wav): "Transcript" for wav in wavs}),
                redirect_stdout(output),
            ):
                cli.main()
            lines = output.getvalue().splitlines()
            self.assertEqual(lines[0], f"Processing file: {inputs[0]}")
            self.assertEqual(lines[1], "Input format: 16-bit PCM WAV, 48 kHz, mono")
            self.assertEqual(lines[2], f"Converting to 16-bit PCM WAV, 16 kHz, mono: {inputs[0]}")
            self.assertRegex(lines[3], r"^Conversion completed in \d+\.\d{2}s$")
            self.assertEqual(lines[4], "Running transcribe.cpp with cohere on 1 audio chunk(s)...")
            self.assertIn(f"Processing file: {inputs[1]}", lines)
            self.assertIn("Input format: 16-bit PCM WAV, 16 kHz, mono", lines)
            self.assertEqual(sum(line.startswith("Converting to") for line in lines), 1)
            self.assertEqual(sum(line.startswith("Running transcribe.cpp") for line in lines), 2)

    def test_writes_transcript_next_to_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_file = Path(temporary) / "audio.wav"
            input_file.touch()
            with (
                patch.object(sys, "argv", ["main.py", str(input_file)]),
                patch.object(cli, "ensure_binary", return_value=Path("transcribe-cli")),
                patch.object(cli, "require_vulkan_gpu"),
                patch.object(cli, "resolve_model_file", return_value=Path("model.gguf")),
                patch.object(cli, "describe_input", return_value="16-bit PCM WAV, 16 kHz, mono"),
                patch.object(cli, "convert_and_chunk", side_effect=lambda path, directory, seconds: ([directory / "audio.wav"], 1.0)),
                patch.object(cli, "transcribe", side_effect=lambda binary, model_file, wavs, language, batch, directory, model: {str(wavs[0]): "Transcript"}),
            ):
                cli.main()
            self.assertEqual(input_file.with_suffix(".txt").read_text(), "Transcript\n")


if __name__ == "__main__":
    unittest.main()
