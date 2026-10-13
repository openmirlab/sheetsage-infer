# sheetsage-infer

Inference-only fork of [SheetSage](https://github.com/chrisdonahue/sheetsage) (audio -> lead
sheet: melody + chords). Install the maintained source from GitHub (previously published as
`openmirlab-sheetsage-infer` through 0.2.0; that name is now deprecated).

## Orientation

- `sheetsage/infer.py` -- the public entry point (`sheetsage()` + the `main()` CLI): beat-track
  -> chunk -> extract features -> transcribe -> assemble `LeadSheet`. Start here.
- `sheetsage/pipeline/` -- the enums/constants (`types.py`) and the private `_*` step helpers
  (`steps.py`) that `sheetsage()` calls, one per pipeline stage; `infer.py` imports and
  re-exports all of them, so every name that used to live in `infer.py` (enums, constants,
  `_init_extractor`, `_beat_tracking_with_hints`, etc.) is still importable from there.
- `sheetsage/representations/` -- feature extractors: `handcrafted.py` (CPU, always available),
  `jukebox.py` (GPU, optional, imports the `jukebox_infer` package).
- `sheetsage/theory/` -- music-theory classes (`Note`, `Chord`, `LeadSheet`, LilyPond/MIDI export).
- `sheetsage/assets.py` -- resolves named model weights / test fixtures to checksum-verified
  local files, downloaded on first use into `~/.sheetsage` (see `sheetsage/assets/*.json`).
- `sheetsage/beat_track.py` -- madmom-infer's DBN downbeat tracker with a librosa fallback.
- `examples/dataset_assets.py` and `examples/dataset_catalogs/` -- repository-only dataset
  acquisition; the HookTheory examples use this helper. Neither helper nor
  catalogs ship in wheel or sdist. Dataset files and generated transcriptions are not
  tracked. Runtime never imports examples/tools.

## Known constraints (read before touching install/dependency config)

- **madmom-infer**: as of 0.2.1, `madmom` (the git-HEAD-only PyPI-broken dependency) was
  replaced by [`madmom-infer`](https://pypi.org/project/madmom-infer/), a plain PyPI package --
  `pip install` works with no extra steps. `beat_track.py` calls madmom-infer's Python API
  directly (`RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`), not a CLI binary (the
  original `madmom` package shipped a `DBNDownBeatTracker` console script; madmom-infer is a
  pure-Python library with no console script, so the old subprocess-based call was rewritten).
  One gap this bridges: madmom-infer has no ffmpeg-backed resampler, so non-44.1kHz audio is
  resampled with librosa before being handed to it (see `beat_track.py`'s module docstring).
  `beat_track.py` also opts into madmom-infer's exact `fast_viterbi` decoder when the installed
  `DBNDownBeatTrackingProcessor` signature explicitly exposes it; 0.2.0-compatible processors
  keep the legacy constructor call.
- **jukebox_infer**: `sheetsage/representations/jukebox.py` imports the real, published
  `jukebox_infer` package (not vendored). It carries a small documented monkeypatch working
  around a real upstream bug (`RangeEmbedding` not casting `n_time` to `int`, which surfaces as
  a `TypeError` from `torch.Tensor.view`) -- see the comment block at the top of that file. If
  `jukebox_infer` ships a fix, remove the patch.
- **Licensing is two-tier**: code is MIT; weights and HookTheory-derived data downloaded via
  `sheetsage.assets` are CC BY-NC-SA (as is madmom's bundled DBN downbeat model). See `NOTICE`.

## Documentation conformance

README reviewed 2026-07-12 against the org's documentation-conformance shape
(Why-this-exists -> Acknowledgments -> Citation -> Features -> Scope ->
Install -> Quick Start -> ... -> "will NEVER bundle" -> Development ->
License -> Support). One citation error was found and fixed: the README's
Citation section previously read "SheetSage: A Hierarchical Transformer for
Audio to Lead Sheet Transcription" (Donahue, ISMIR 2024) -- a nonexistent
title/venue/year. Checked against the arXiv abstract page
(https://arxiv.org/abs/2212.01884), the real paper is "Melody Transcription
via Generative Pre-Training" by Chris Donahue, John Thickstun, and Percy
Liang, ISMIR 2022. The `NOTICE`/`LICENSE` copyright year (2022) was already
correct -- it matches `chrisdonahue/sheetsage`'s actual repo-creation date
(2022-06-10, verified via `gh api repos/chrisdonahue/sheetsage`) -- so only
the Citation bibtex needed correcting, not the copyright year.

## Testing

`tests/` has import smoke tests, bit-exact regression fixtures (env-guarded: they SKIP rather
than fail on a torch/librosa/numpy/python mismatch, since those genuinely shift floating-point
output -- don't "fix" a skip by loosening a fixture's tolerance without checking whether the env
actually matches first), and an opt-in GPU Jukebox smoke test
(`SHEETSAGE_RUN_JUKEBOX_TESTS=1 uv run pytest tests/test_jukebox_smoke.py`). Run the default
suite with `uv run pytest tests/`.

The dataset boundary is checked by `tests/test_dataset_boundary.py`: all 22 former
HookTheory/RWC tags are rejected while all 24 inference/test tags retain their checksums.
Run repository example tests explicitly with `python -m pytest tools/test_dataset_examples.py`;
they mock downloads and verify catalog-byte preservation, both consumers, cache/checksum
behavior, and manual-only failures. These tests are repository-only like their subjects.
`pyproject.toml` declares explicit Hatchling sdist includes/excludes and a sheetsage-only
wheel. Examples, dataset catalogs/helper, and historical dataset/output directories remain
excluded from both formats. The repository-only example test is excluded from the sdist. Verify both archives with
`python tools/check_distribution_boundary.py dist/*.whl dist/*.tar.gz`.

`tests/fixtures/cpu_boundary_baseline/` is immutable original-runtime evidence at `b37c457`.
See `tools/README.md` for exact CPU replay on cached real/silent inputs, all returned fields,
complete environment and checkpoint hashes. Both repeated real calls and independent-process
replay are exact. Silence currently fails at beat detection; this stage does not change it.
The old guarded tests still skip here; only explicit successful replay counts as parity.
Dataset manifests moved byte-for-byte; no lifecycle, inference-model, numerical-dependency,
Python-support, or weight-hosting changes belong to this boundary stage.

## Device and lifecycle contract

All model-loading public paths (`sheetsage()`, `SheetSageSession`, and the CLI) accept
`device="auto"`, `"cpu"`, `"cuda"`, and `"cuda:N"`. `sheetsage()` and CLI calls with no
device preserve the historic CPU-handcrafted/CUDA-Jukebox selection; `auto` is explicit opt-in.
New sessions default to `auto`. `sheetsage.device.resolve_device()` resolves explicit `auto`
and rejects invalid or unavailable CUDA. Never add a fallback that changes an explicit request,
including after CUDA OOM.

`SheetSageSession.load()` constructs session-owned extractor/transducer components once;
`infer()` only consumes those components. `release()` and `close()` clear those references
without deleting downloaded assets. States are `new`, `ready`, `failed`, `released`, and
`closed`. Failed/interrupted imports, device resolution, or component construction clear
owned components/resolved device, report `failed`, and re-raise the original exception;
retry is allowed. Repeated successful `load()` does not rebuild. `release()` reports
`released` and permits reload, but never revives `closed`. `close()` and context exit are
terminal and idempotent. This does not alter loader-owned partial locals, model construction
order, the per-call beat processor, cache policy, or one-shot inference behavior.
`cache_info()` uses `assets.resolve_asset_path()`, the
same resolver used by `retrieve_asset()`. `sheetsage/config/checkpoints.toml` is packaged and
is the runtime source of truth for all SheetSage assets. It preserves every URL/HuggingFace
resolver leg and records the existing checksum with its explicit algorithm (including SHA-1 CFG
digests and SHA-256 model/STEP digests); it supports generic `SHEETSAGE_ASSET_URL_<TAG>` overrides.
Focused contract coverage is
`uv run pytest tests/test_device_session.py tests/test_session_lifecycle.py`; complete
verification remains `uv run pytest tests/ tools/test_dataset_examples.py` from a checkout.
Lifecycle tests first reproduced seven failures, then passed with failed/released state
reporting and reference cleanup. The committed CPU baseline replay remains exact. Two real
inferences on one loaded session also match every baseline output field, construct one
extractor/two transducers exactly once, and make no download attempts. GPU numerical parity
and the per-call beat-processor lifecycle are outside this change.

When changing `sheetsage/representations/jukebox.py`, `make_models.py`-adjacent code, or
anything in the feature-extraction path: re-run the CPU regression test at minimum, and the GPU
smoke test if you have a free GPU -- a prior campaign found the Jukebox path silently broken
end-to-end (see CHANGELOG's Unreleased section) precisely because nothing exercised it.

## Conventions

- Nav-style headers (what a file is, what it reads/is read by) on load-bearing modules -- keep
  them in sync when you restructure imports.
- CHANGELOG.md: add entries under `[Unreleased]` for behavior changes, not just version bumps.
- Commit messages / PRs: no emojis.

## Delivery packaging

`__about__.py` is the sole version literal, read by Hatchling and re-exported by `__init__.py`.
`python -m build` must build its wheel from the sdist. Preserve all runtime JSON/TOML assets
and test fixtures, including the explicitly included immutable CPU baseline NPZ.
`tests/fixtures/delivery_baseline.json` records the lifecycle-stage wheel payload at `0a471250`.
`tools/verify_installed.py CHECKOUT` checks every installed source/asset byte against final
source and that manifest; only explicit version and SheetSage2 exports in `__init__.py`
are normalized. `tools/check_distribution_boundary.py` lists the exact additive
runtime files: version owner, SheetSage2 session/checkpoint modules and manifest.

The reusable `.github/workflows/verify.yml` runs Python 3.10–3.12 contracts, fresh CPU installs,
`pip check`, scoped verification-tool lint, archive checks, and installed-wheel identity.
The publish workflow depends on it. Do not describe this matrix as model parity: existing
historical environment guards and optional GPU coverage remain explicit skips.
`capture_cpu_baseline.py --portable --cpu-assets-only --installed` is an opt-in cached real
CPU verifier: exact decoded/segment/error/status outputs, shape/dtype/finite/nonzero checks,
repeat equality, and measured float differences. Default replay still requires exact arrays.
All 13 configured SheetSage digests must match the immutable baseline; available cached bytes
are independently verified, while six unused Jukebox files may be absent only in CPU mode.
Seven handcrafted assets are required. Unauthenticated S3 returned 403 and HF fallback 401
for all seven on 2026-10-03; do not add a hosted model job that skips this missing prerequisite,
mirror weights, or upload private caches. Hosting/access requires a separate decision.

## Distribution policy (2026-10-05)

Install the current source from `https://github.com/openmirlab/sheetsage-infer`. GitHub release workflows verify and build distributions but do not upload to PyPI. Existing PyPI versions, where any exist, are historical snapshots. Update installation examples to use Git when changing this package.

## Experimental SheetSage2 backend

The GitHub `master` source includes an independent adapter for M·A·P's SheetSage2 (Jiang et al.,
2026, https://arxiv.org/abs/2610.05336), using its MERT-v2-FullSong encoder.
`sheetsage/sheetsage2.py` owns `SheetSage2Session` and the lazy `sheetsage2()`
one-shot API. It preserves the native rich result dictionary; it does not
convert it to SheetSage1's LeadSheet. Legacy inference modules are unchanged.

`sheetsage/sheetsage2_checkpoints.py` owns manifest validation, resolution,
inspection, SHA-256 checks and atomic downloads. `config/sheetsage2.json` pins
23 adapter/runtime artifacts and 4 MERT2 artifacts, including both revisions,
URLs, hashes and sizes. Local directory overrides are read-only; the same
resolver serves load and cache_info. Inspection reports presence/size, not
verified integrity. load verifies every file before executing upstream code,
then gives both local paths to AutoModel with local_files_only=True. Runtime
code is downloaded into the user cache, not included in wheel/sdist. Source
redistribution terms remain unresolved; see NOTICE before considering a port.

The optional `sheetsage2` extra adds Transformers/torchaudio/safetensors/mir_eval.
mir_eval is used by upstream MIDI chord encoding; no evaluation framework is
added to the core install or our runtime. Transformers 5.19 fails at BART's
removed embed_tokens constructor argument; the justified <5 bound is tracked
in docs/compatibility/sheetsage2-transformers.md. NumPy 2.4.6 works with
Transformers 4.45.2 and Torch/torchaudio 2.9.1. A second 4.57.6 profile also
loads and transcribes the real test clip. These observations do not establish
numerical parity across dependency versions.

Sessions use new/ready/failed/released/closed states, load once, and retain no
shared model manager. infer is ready-only; release permits reload, close is
terminal. Failed/interrupted loading drops session-owned references and allows
retry. Sequential calls are supported; concurrent inference on one session is
not. CPU/CUDA selection uses the existing resolver. Rendering is excluded;
ABC/MIDI/annotations and optional logits/scores/embeddings pass through.

Verification commands (checkout, with the extra installed):

```bash
python -m pytest tests/ tools/test_dataset_examples.py
PYTHONPATH=. python tools/verify_sheetsage2.py ~/.sheetsage/test/test1_44100.wav \
  --cache-dir /path/to/populated/sheetsage2-cache --device cuda \
  --output /tmp/sheetsage2-parity.json
python -m build
python tools/check_distribution_boundary.py dist/*.whl dist/*.tar.gz
```

The verifier requires real cached files, compares direct upstream and wrapper
outputs on real audio plus silence, and checks model reuse. Default unit tests
are offline contracts, not model-quality evidence. The three legacy numerical /
GPU skips remain explicit. Installed-wheel verification preserves every original
runtime hash except the explicitly normalized root exports and explicitly listed
new files; it also compares all added installed bytes to the checkout.

The 2026-10-13 full installed-wheel CPU check used Torch 2.14.1+cpu,
torchaudio 2.11.0+cpu and Transformers 4.57.6. It produced 19 events and 16
notes on the real fixture. Installing PyPI's CUDA torchaudio into CPU Torch
first reproduced a libcudart load failure; the README and CPU CI recipe install
both from the CPU wheel index. Optional import errors now explain this case.
The refreshed uv.lock reflects the existing Jukebox Git pin's Torch >=2.13
requirement; it was already present in that dependency's metadata before this
feature. The original local Torch 2.9.1 environment still replays SheetSage1's
immutable baseline exactly, including its existing silence error.

## Shared transcription and structure

`SheetSageTranscriber` (package-qualified name) and `transcribe` add a shared
interface over the unchanged native sessions. `sheetsage/transcription/api.py`
owns routing/lifecycle, `options.py` owns immutable backend settings,
`adapters.py` owns conversion, and `result.py` owns values, versioned semantics
and JSON/MIDI export. Heavy imports stay lazy. No global runtime or second
model/downloader is introduced. Native APIs and original result objects remain
available; per-call options replace the whole typed object, and session settings
cannot change after construction.

Structure is a first-class optional result. V1 is unsupported; V2 exposes native
labels and half-open intervals, including `(timestamp,structure)`-only prompts.
`require_structure` rejects unsupported/disabled tasks before load.
Capabilities describe support, statuses distinguish ok/empty from unsupported,
not_requested and failed. ABC failure does not invalidate structure. Field
conversion failure retains native evidence and successful other fields.

Before editing converters or exports, read `docs/result-semantics.md`: it records
source traces, model profiles, options and coordinate/value invariants. In
particular V1 raw MIDI pitch is `pc+12*octave`, not `Note.as_midi_pitch()`'s default;
V1 beat maps already include crop offsets; V2 stitched events already include
window offsets. Shared MIDI contains note roles and text labels with generated
velocity, while native MIDI retains upstream voicing/notation conventions.

Verification commands:

```bash
python -m pytest tests/ tools/test_dataset_examples.py
ruff check sheetsage/transcription tools/verify_transcription.py
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/verify_transcription.py \
  ~/.sheetsage/test/test1_44100.wav --model sheetsage1 --device cpu --output /tmp/shared-v1
PYTHONPATH=. python tools/verify_transcription.py \
  ~/.sheetsage/test/test1_44100.wav --model sheetsage2 --device cuda \
  --cache-dir /path/to/populated/sheetsage2-cache --output /tmp/shared-v2
python -m build
python tools/check_distribution_boundary.py dist/*.whl dist/*.tar.gz
```

The installed-byte gate allows five explicit new transcription modules and the
precise additive root exports. Historical runtime hashes remain immutable.
The source archive includes the result-semantics document. The public MIR skills
capability map is updated in the same window. Integration fixtures do not certify
natural whole-song structure quality; the 305-second repeated music/silence smoke
exercises actual 300-second windows, not perceptual segmentation accuracy.

Paper preset file/bytes/stream decoding requires upstream torchaudio.info. Tested
modern torchaudio builds lack it; the shared facade explains this limitation.
Paper waveform inputs bypass that reader and are independently verified.
