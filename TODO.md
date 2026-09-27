# TODO

- [ ] Check the transcribe.cpp documentation for each model's full-file input support and audio limits. Skip chunking when the selected model can accept the entire file, including Parakeet when the file is within its supported limit. Keep chunking for files that exceed a model's limit.
- [x] Detect 16 kHz mono WAV input and pass it directly to transcribe.cpp without FFmpeg conversion. Continue converting inputs that do not meet that format requirement.
- [x] Output "xxx.mp4: 1345.11s audio; batch completed in 39.54s" should also show the multiplier speed-up (e.g. 34.02x)
- [x] When processing multiple files (e.g. python main.py input/\*) immediately after each input audio file is processed, print the status to the screen and the speed up
