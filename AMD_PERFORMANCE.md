# AMD Radeon 860M Performance Notes

This document summarizes performance testing of
`CohereLabs/cohere-transcribe-03-2026` on an AMD Radeon 860M integrated GPU.
The GPU shares system memory and its power budget with the CPU. PyTorch exposes
ROCm through its `cuda` device API, and inference uses FP16.

## Recommendation

Use the Performance power profile and a GPU batch size of 8:

```console
python main.py INPUT --batch-size 8
```

Batch size 8 is the best demonstrated default for this hardware. Larger batches
did not improve short-input performance and substantially reduced throughput on
a 30-minute input. Batch size 4 remains worth one comparison on a long input,
but it has not yet been shown to beat 8.

## Measurements

### 63-second input

The processor produced two audio chunks, so every batch size of 2 or greater
was effectively identical. Raising the configured batch size could not create
additional parallel work.

| Batch size | Time | Throughput |
| ---: | ---: | ---: |
| 1 | 8.34 s | 7.62 audio sec/sec |
| 2 or greater | approximately 5.4 s | approximately 11.6 audio sec/sec |

### 4-minute 22-second input

With the Performance power profile enabled:

| Batch size | Time | Throughput |
| ---: | ---: | ---: |
| 2 | 15.22 s | 17.23 audio sec/sec |
| 4 | 12.64 s | 20.74 audio sec/sec |
| 8 | 10.80-10.83 s | 24.20-24.26 audio sec/sec |
| 16 | 10.94 s | 23.95 audio sec/sec |
| 32 | 11.30 s | 23.19 audio sec/sec |

This input appears to fit in approximately eight chunks. Batch sizes 16 and 32
therefore did not expose more parallelism than batch size 8. Their small timing
differences are consistent with ordinary run-to-run variation.

Power management had a large effect. Before selecting the Performance profile,
batch-8 runs took 11.87-13.27 seconds and a batch-2 run took 29.38 seconds.
Earlier much slower results were likely influenced by power state, clock ramping,
thermal state, or other system activity.

### 30-minute input

With the Performance power profile enabled:

| Batch size | Time | Throughput | Time relative to batch 8 |
| ---: | ---: | ---: | ---: |
| 8 | 115.30 s | 15.61 audio sec/sec | baseline |
| 16 | 153.15 s | 11.75 audio sec/sec | 33% longer |
| 32 | 186.56 s | 9.65 audio sec/sec | 62% longer |

This recording contains enough chunks to fill the larger batches. On this iGPU,
the additional shared-memory bandwidth, cache, temporary-memory, and sustained
power pressure outweighed the reduction in the number of `generate()` calls.

## Why Throughput Is Not Constant

`--batch-size` controls the number of processor-created audio chunks passed to
one `model.generate()` call. It is a ceiling, not a guarantee that every batch
contains that many chunks. Conceptually, the current code performs:

```text
audio -> processor-created chunks -> fixed-size groups -> model.generate()
```

The encoder's work is fairly predictable for fixed-duration chunks, but the
decoder generates text autoregressively, one token at a time. Chunks with the
same audio duration can produce different numbers of tokens. A batch continues
until its longest-generating row finishes, while rows that finished earlier
remain padded in the batch. One unusually long, repetitive, or hallucinated
result can therefore make the entire batch expensive.

The 4-minute input likely required one batch of approximately eight chunks. The
30-minute input required several sequential batches, increasing exposure to:

- variation in generated token counts and speech density;
- wasted decoder work caused by unequal output lengths within a batch;
- chunks that approach the hardcoded `max_new_tokens=256` limit;
- sustained thermal and shared CPU/GPU power limits;
- CPU feature extraction, device transfers, and text decoding.

For these reasons, increasing audio duration does not guarantee linear runtime
or constant audio-seconds-per-second throughput.

## What the Reported Metric Includes

The timer starts immediately before `batched_transcribe()` and includes processor
feature extraction, transfers to and from the GPU, model generation, token
trimming, text decoding, and chunk reassembly. It excludes model loading, media
decoding, and writing the transcript. Processing several files in one invocation
still improves total wall-clock efficiency by loading the model only once.

## Remaining Opportunities

There is no other simple constant currently known to provide a major improvement
comparable to selecting the correct batch size and power profile. Useful next
steps, in priority order, are:

1. Run one batch-size-4 comparison on the 30-minute input. If it is within about
   10% of batch size 8, prefer 8 unless lower memory pressure is important.
2. Add per-batch timing, effective chunk counts, and generated-token lengths.
   This would distinguish variable decoder work from sustained throttling.
3. Compare transcript outputs or hashes across batch sizes. Batching should not
   silently change transcription quality under deterministic greedy decoding.
4. Reduce `max_new_tokens` only if instrumentation shows chunks repeatedly
   reaching 256 tokens; reducing it blindly can truncate legitimate dense speech.
5. Consider a smaller ASR model if throughput is more important than preserving
   the current model's accuracy.
6. Consider `torch.compile` only for a persistent process handling enough inputs
   to amortize compilation. Its ROCm benefit should be measured rather than
   assumed.
7. Consider voice-activity detection or silence segmentation if long silent or
   problematic regions cause excessive generation.

Quantization is not automatically faster on this GPU without an efficient ROCm
kernel path. Increasing reserved UMA memory may avoid out-of-memory failures but
normally does not improve bandwidth or inference speed. FP16 remains the sensible
default datatype.

## Benchmarking Guidance

Keep the laptop plugged in, select the Performance power profile, and minimize
other CPU/GPU activity. For a quick comparison, alternate batch sizes to reduce
ordering and thermal bias and compare representative or median results rather
than the single fastest run:

```fish
for batch in 8 4 8
    echo "batch size: $batch"
    python main.py INPUT --batch-size $batch
end
```

Always pass `--batch-size`; changing only the text printed by `echo` does not
change the program's batch size.
