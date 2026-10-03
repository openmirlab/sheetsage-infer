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

# Dataset-boundary checks

`python -m pytest tests/test_dataset_boundary.py tools/test_dataset_examples.py` checks
the installed registry and repository-only dataset examples without network access.
The example catalogs retain their original hashes; their helper verifies downloaded/cached
bytes before use. Both downloading examples use it; the local-JSON transcription example
remains unchanged. Run `python tools/check_distribution_boundary.py dist/*.whl dist/*.tar.gz`
after building to reject dataset catalogs, helpers, or payloads in either archive.

# Delivery and portable CPU verification

The delivery matrix runs every current contract test on Python 3.10–3.12, including the
repository example tests, then builds a wheel from its sdist and verifies its installed
payload outside the checkout. Historical environment-guarded and optional GPU tests retain
their explicit skip reasons; matrix success is not an end-to-end model-parity claim.

For a separately populated cache, run the real model without allowing downloads:

```bash
# Source checkout; fail if any required cached asset is absent or corrupt.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/capture_cpu_baseline.py \
  --portable --cpu-assets-only --reference tests/fixtures/cpu_boundary_baseline \
  /tmp/sheetsage-portable

# Installed wheel, from an unrelated working directory:
python /path/to/checkout/tools/verify_installed.py /path/to/checkout
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /path/to/checkout/tools/capture_cpu_baseline.py \
  --installed --portable --cpu-assets-only \
  --reference /path/to/checkout/tests/fixtures/cpu_boundary_baseline /tmp/sheetsage-installed
```

Portable mode compares exact lead-sheet notes/chords, LilyPond, segments, beat indices/times,
chunk slices, statuses, and silence errors. Every float array must have the recorded shape
and dtype, finite/nonzero content, and identical repeated-call outputs. Maximum absolute and
relative RMS differences against the immutable original arrays are diagnostic fields, not a
relaxed numerical-equality assertion. Default mode (omit `--portable`) still requires exact
arrays and remains the original-profile regression check; fixtures are never rewritten.

Measured Python 3.11 / Torch 2.14.1+cpu outputs preserved every discrete result and feature/
beat array. Logits differed by maximum absolute 2.861023e-6; relative RMS was 1.824877e-7 for
melody and 1.923831e-7 for harmony. Exact comparison correctly failed for that profile.
Original Torch 2.9.1+cu128 on explicit CPU still replays all ten arrays exactly.

`--cpu-assets-only` requires the seven handcrafted artifacts, TEST_WAV and all eight madmom
downbeat weights. All 13 configured SheetSage manifest digests must still match the original
baseline; every present cache file is byte-hashed. Only the six unused Jukebox artifacts may
be absent, and their names are recorded explicitly. This does not validate GPU/Jukebox inference.

On 2026-10-03, unauthenticated probes returned S3 HTTP 403 and existing Hugging Face fallback
HTTP 401 for **all seven** handcrafted files. TEST_WAV and official madmom weight probes
returned 200. A clean hosted real-model lane therefore needs an independently authorized
access or hosting solution for the seven configured artifacts. No workflow uploads caches,
changes mirrors, or treats unavailable weights as a passing model check.
