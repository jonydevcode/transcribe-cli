# Accuracy check

Compare a new GGUF transcription with an earlier transcript of the same recording. The audio and earlier
transcripts live in the local, git-ignored `inputs/` directory, so this check only works on a machine that has them.
For a committed smoke test use `tests/fixtures/sample_15s.aac`.

```bash
python -m transcribe_cli inputs/20260726223646.WAV --model cohere
diff -u inputs/20260726223646_cohere.txt inputs/20260726223646.txt
```

The text may differ because this CLI uses a separate runtime and quantized checkpoint. Validate by inspecting the
differences rather than requiring byte-for-byte identity.
