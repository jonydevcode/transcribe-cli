# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from main import MODEL_IDS, QWEN_CHUNK_SECONDS, SAMPLE_RATE, parse_args, qwen_transcribe


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


if __name__ == "__main__":
    unittest.main()
