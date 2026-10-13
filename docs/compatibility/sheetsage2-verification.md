# SheetSage integration verification — 2026-10-13

Verification target: code commit `decc543` (SheetSage2 backend, shared transcription
API and merge-review fixes). Later documentation-only changes do not change the
runtime tested here. This is integration and delivery evidence, not a benchmark
of transcription or verse/chorus recognition accuracy.

## Contract and delivery checks

| Profile | Contract suite | Delivery checks |
|---|---|---|
| Python 3.10.18, Torch 2.14.1+cpu, torchaudio 2.11.0+cpu, Transformers 4.57.6 | 164 passed, 3 skipped | Dependency check, native imports, wheel from sdist, archive boundary, installed-wheel verification passed |
| Python 3.11.13, existing Torch 2.9.1+cu128 environment | 164 passed, 3 skipped | Original numerical replay below; this environment is not the current full dependency-install profile |
| Python 3.12.11, Torch 2.14.1+cpu, torchaudio 2.11.0+cpu, Transformers 4.57.6 | 164 passed, 3 skipped | Dependency check, native imports, wheel from sdist, archive boundary, installed-wheel verification passed |

The two fresh CPU environments installed the full `.[dev,sheetsage2]` extras.
Installed-wheel verification ran outside the checkout: 37 runtime files matched
the source, 27 legacy files were unchanged, and root exports were additive only.
Targeted Ruff and `git diff --check` passed.

The three suite skips are explicit: two historical fixtures have strict
numerical-environment guards, and the Jukebox GPU smoke requires opt-in. Separate
original-profile replay below supplies exact real/silence CPU numerical coverage
against the immutable CPU boundary baseline; the skipped tests remain skipped.
Jukebox's encoder checkpoints are absent; its GPU smoke was not run. This change
does not alter the legacy Jukebox runtime. macOS/Windows and hosted CI were not
executed in this local verification.

## Model and format checks

- V1, explicit CPU and original Torch 2.9.1 profile: immutable real/silence replay
  passed exactly, including every intermediate array and the expected silence
  error. Direct versus shared inference, repeated native results and repeated
  normalized results matched; the real fixture produced nine notes.
- V2, RTX 4090, Python 3.11 / Torch and torchaudio 2.9.1 / Transformers 4.45.2:
  pinned upstream versus wrapper, repeated real/silence inference, and immutable
  golden comparison passed. Optional tensors were included; timing and peak
  memory measurements alone are excluded from equality.
- V2, installed Python 3.12 CPU wheel outside the checkout: direct/shared
  native, repeated native and repeated normalized results matched exactly,
  producing 16 notes. Structure-only inference returned an `intro` interval
  from 0.02 to 4.5875416667 seconds, with other fields not requested. JSON/MIDI
  exports passed. Four calls with one-thread settings took about 17 minutes;
  this is not a performance benchmark.
- Fourteen direct-versus-shared cases passed: local path, bytes, bytearray,
  memoryview, binary stream, 24 kHz mono NumPy, 48 kHz stereo NumPy, 48 kHz stereo
  Torch, `max_seconds=2.5`, paper waveform, fp32, vocal/major-minor prompts,
  structure-only prompts, and native artifact export with `melody_only` plus
  logits/scores/embeddings. This verifies forwarding and result preservation,
  not equality between different precision or decoding presets.
- Native artifact export produced readable MIDI plus ABC, LAB, JSON, TSV and
  safetensors files. Shared JSON/MIDI exported successfully; shared MIDI note
  counts survived readback. Structure-only results exported JSON and explicitly
  reported the other fields as not requested.

The common short WAV input has SHA-256
`ff8bf45609bfd5fd759af718b1e44e5544a5783c316156b6b7450ec48652718c`.
The paper file reader still depends on upstream `torchaudio.info`; with modern
TorchAudio, callers must use decoded waveforms or the default preset, as
explained in the public API error and README.

## Full music and multiple windows

| Input | Decoded seconds | Shared notes | Windows | Observed structure labels |
|---|---:|---:|---:|---|
| `newjeas_supershy.wav` | 200.829375 | 516 | 1 | intro, verse, chorus |
| `nujabes_luvsic_pt2.wav` | 270.199042 | 528 | 1 | intro, verse, chorus, interlude, outro |
| `cut_liszt.mp3` | 40 | 77 | 1 | intro |
| Full Nujabes track plus 45 seconds of zeros | 315.199042 | 538 | 2 | intro, verse, chorus, interlude, outro |

Every requested shared field converted successfully. Note/section times stayed
inside the decoded audio extent; shared structure intervals matched native LAB
intervals exactly. The 315-second case also matched every native field against
an independent second direct inference, exercising the real 300-second window
boundary. Upstream notation warnings were retained in the shared results.

Source SHA-256 values, in table order for the three files:

- `21d69213a9526b36646362e9579aa902686cd0a2074bba165d3d3be019452feb`
- `42fa5ee6c137373f06413cf43195066180eef2311a6ef20c6c5c495e6af54e4d`
- `0a551787adbc61776e3b0691b7727a2276f2a8bf47fa4aad0fb4e8f89ef83603`

No audio or checkpoint bytes are included in this report. The section names are
model predictions without human ground truth. In particular, the music-plus-zero
case checks stitching and parity; it does not establish silence segmentation
quality or full coverage of every structure class.

## Defects caught before merge

Regression tests first reproduced the defects and then passed with the fixes:

- Mutable/relative checkpoint manifests could make provenance wrong or throw
  after successful inference. Construction now captures the exact validated
  manifest bytes' identity; moving, modifying or deleting the file cannot alter
  that snapshot.
- Long MIDI exports exceeded a reader's tick limit. Adaptive resolution bounds
  the event timeline and an exact tempo prevents cumulative float-conversion
  drift. Independent 601-second notes and 1200-second annotation tails roundtrip
  within the declared tick precision.
- NumPy arrays with identical values but different dtypes escaped the parity
  check. Type, dtype, shape and values are now checked.
- Malformed native rhythm/value containers could throw or appear empty. Field
  failures are explicit and independent valid fields remain available.

## Reproduction and local evidence

See [verification commands](../../tools/README.md#shared-transcription-verification)
for cached native/shared comparison and immutable upstream/CPU replay, and
[output semantics](../result-semantics.md) for the conversion and MIDI policies.
Run the current suite with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/ tools/test_dataset_examples.py -ra
python -m build
python tools/check_distribution_boundary.py dist/*.whl dist/*.tar.gz
```

Local logs and generated results for this run are under
`/tmp/sheetsage-merge-matrix`, `/tmp/sheetsage-merge-v1`,
`/tmp/sheetsage-merge-real`, `/tmp/sheetsage-merge-inputs`, and
`/tmp/sheetsage-merge-upstream.json`. The additional real-input harnesses are
`/tmp/sheetsage_merge_real.py` and `/tmp/sheetsage_merge_inputs.py`; these local
artifacts are not part of the distributed package.
