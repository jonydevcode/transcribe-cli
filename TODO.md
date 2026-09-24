# TODO

- [ ] Check the transcribe.cpp documentation for each model's full-file input support and audio limits. Skip chunking when the selected model can accept the entire file, including Parakeet when the file is within its supported limit. Keep chunking for files that exceed a model's limit.
- [ ] Detect 16 kHz mono WAV input and pass it directly to transcribe.cpp without FFmpeg conversion. Continue converting inputs that do not meet that format requirement.
