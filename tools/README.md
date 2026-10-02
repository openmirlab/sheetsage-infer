# CPU baseline evidence

`tests/fixtures/cpu_boundary_baseline/` records the unchanged runtime at commit
`b37c4579a788123e408b80d011f876631daa101f`, before dataset-boundary changes.
It contains derived outputs and hashes, never audio or checkpoint bytes.

Use an independently populated SheetSage/madmom cache and the recorded environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/capture_cpu_baseline.py \
  --reference tests/fixtures/cpu_boundary_baseline /tmp/sheetsage-replay
```

`--record NEW_DIRECTORY` is allowed only at the original Git commit with unchanged runtime
files. The historical regression fixtures remain untouched. The capture uses explicit CPU,
one Torch thread, seed zero, and the default handcrafted pipeline with
`return_intermediaries=True`. It verifies all 13 cached SheetSage artifacts and eight madmom
downbeat weights before inference. SheetSage downloads are disabled; missing cached inputs
fail. Nothing downloads, packages, or redistributes audio/model bytes.

The cached `TEST_WAV` contains 202,311 stereo samples at 44,100 Hz (RMS 0.06765538).
An equal-length, exactly-zero input is encoded only in memory. Both inputs have complete
standalone handcrafted feature arrays. Two default pipeline calls per input record every
returned field: full lead-sheet tuples and LilyPond text, beat indices/times, chunk slices,
and melody/harmony logits. Silence instead raises `ValueError: Audio too short to detect
time signature` during beat detection in both calls; no nonexistent logits are invented.

The real result has nine melody notes, one chord, and 32 tertiaries. All ten stored arrays
are finite; melody/harmony logits have RMS 2.17348/2.48747. Repeated outputs and a separate
process replay are exactly equal. Current LilyPond text also exactly matches the historical
fixture despite its environment guard skipping this machine. That observation does not yet
prove equality on every supported Python/dependency version.

The initial cached real-audio probe took about 11.4 seconds including feature warm-up.
Recorded pipeline calls after separate feature extraction took about 2.5–3.0 seconds;
silence stopped after about 0.5–0.9 seconds. This supports a practical real-model CI lane
with independently acquired cached assets, but no environment-skipped test is counted as
numerical verification. Package/model hashes, package versions, Torch build settings,
platform, input hashes, array shapes/dtypes/RMS/finite checks, and timings live in the JSON.
