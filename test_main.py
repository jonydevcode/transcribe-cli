# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import main as cli

from main import (
    MODEL_IDS,
    NEMOTRON_CHUNK_SECONDS,
    QWEN_CHUNK_SECONDS,
    SAMPLE_RATE,
    nemotron_transcribe,
    parse_args,
    qwen_transcribe,
)


class Batch(dict):
    def to(self, device, dtype):
        return self


class QwenTests(unittest.TestCase):
    def test_model_defaults_to_cohere(self):
        with patch.object(sys, "argv", ["main.py", "audio.wav"]):
            args = parse_args()
        self.assertEqual(args.model, "cohere")
        self.assertIsNone(args.language)

    def test_qwen_defaults_to_small_checkpoint(self):
        self.assertEqual(MODEL_IDS["qwen"], "Qwen/Qwen3-ASR-0.6B-hf")

    def test_qwen_detects_language_for_each_chunk(self):
        processor = Mock()
        processor.apply_transcription_request.side_effect = [
            Batch(input_ids=torch.tensor([[1], [1]])),
            Batch(input_ids=torch.tensor([[1]])),
        ]
        processor.decode.side_effect = [["你好", "世界"], ["Hello"]]
        model = Mock(device=torch.device("cpu"), dtype=torch.float32)
        model.generate.side_effect = [
            torch.tensor([[1, 2], [1, 3]]),
            torch.tensor([[1, 4]]),
        ]
        audio = np.zeros(QWEN_CHUNK_SECONDS * SAMPLE_RATE * 2 + 1, dtype=np.float32)

        text = qwen_transcribe(
            processor=processor,
            model=model,
            audio=audio,
            language=None,
            batch_size=2,
        )

        self.assertEqual(text, "你好世界 Hello")
        self.assertEqual(
            [len(call.kwargs["audio"]) for call in processor.apply_transcription_request.call_args_list],
            [2, 1],
        )
        for call in processor.apply_transcription_request.call_args_list:
            self.assertIsNone(call.kwargs["language"])
        for call in processor.decode.call_args_list:
            self.assertEqual(call.kwargs["return_format"], "transcription_only")


class NemotronTests(unittest.TestCase):
    def test_model_option_uses_official_checkpoint(self):
        with patch.object(sys, "argv", ["main.py", "audio.wav", "--model", "nemotron"]):
            args = parse_args()
        self.assertEqual(args.model, "nemotron")
        self.assertEqual(MODEL_IDS["nemotron"], "nvidia/nemotron-3.5-asr-streaming-0.6b")

    def test_cli_loads_rnnt_model_with_auto_language(self):
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "audio.wav"
            input_file.touch()
            model = Mock(device=torch.device("cpu"), dtype=torch.float32)
            model.to.return_value = model
            with (
                patch.object(sys, "argv", ["main.py", str(input_file), "--model", "nemotron"]),
                patch.object(cli, "validate_decoder_dependencies"),
                patch.object(cli, "load_audio_file", return_value=np.zeros(SAMPLE_RATE, dtype=np.float32)),
                patch.object(cli, "resolve_runtime_config", return_value=cli.RuntimeConfig(torch.device("cpu"), torch.float32, 1)),
                patch.object(cli.AutoProcessor, "from_pretrained", return_value=Mock()) as load_processor,
                patch("transformers.AutoModelForRNNT.from_pretrained", return_value=model) as load_model,
                patch.object(cli, "nemotron_transcribe", return_value="Transcript") as transcribe,
            ):
                cli.main()

            load_processor.assert_called_once_with(MODEL_IDS["nemotron"])
            load_model.assert_called_once_with(MODEL_IDS["nemotron"], dtype=torch.float32)
            self.assertEqual(transcribe.call_args.kwargs["language"], "auto")
            self.assertEqual(transcribe.call_args.kwargs["batch_size"], 1)
            self.assertEqual(input_file.with_suffix(".txt").read_text(), "Transcript\n")

    def test_chunks_audio_and_decodes_clean_text(self):
        processor = Mock()
        processor.side_effect = [
            {"input_features": torch.zeros(2, 3, 4), "prompt_ids": torch.zeros(2, dtype=torch.long), "num_lookahead_tokens": 13},
            {"input_features": torch.zeros(1, 3, 4), "prompt_ids": torch.zeros(1, dtype=torch.long), "num_lookahead_tokens": 13},
        ]
        processor.batch_decode.side_effect = [["你好", "世界"], ["Hello."]]
        model = Mock(device=torch.device("cpu"), dtype=torch.float32)
        model.generate.side_effect = [
            Mock(sequences=torch.tensor([[1], [2]])),
            Mock(sequences=torch.tensor([[3]])),
        ]
        audio = np.zeros(NEMOTRON_CHUNK_SECONDS * SAMPLE_RATE * 2 + 1, dtype=np.float32)

        text = nemotron_transcribe(
            processor=processor,
            model=model,
            audio=audio,
            language="auto",
            batch_size=2,
        )

        self.assertEqual(text, "你好世界 Hello.")
        self.assertEqual([len(call.args[0]) for call in processor.call_args_list], [2, 1])
        for call in processor.call_args_list:
            self.assertEqual(call.kwargs["language"], "auto")
            self.assertEqual(call.kwargs["sampling_rate"], SAMPLE_RATE)
        for call in processor.batch_decode.call_args_list:
            self.assertTrue(call.kwargs["skip_special_tokens"])
        for call in model.generate.call_args_list:
            self.assertTrue(call.kwargs["return_dict_in_generate"])
            self.assertEqual(call.kwargs["num_lookahead_tokens"], 13)


if __name__ == "__main__":
    unittest.main()
