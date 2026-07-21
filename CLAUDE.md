# sheetsage-infer

Inference-only fork of [SheetSage](https://github.com/chrisdonahue/sheetsage) (audio -> lead
sheet: melody + chords). Packaged for `pip install sheetsage-infer` (previously published as
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

## Known constraints (read before touching install/dependency config)

- **madmom-infer**: as of 0.2.1, `madmom` (the git-HEAD-only PyPI-broken dependency) was
  replaced by [`madmom-infer`](https://pypi.org/project/madmom-infer/), a plain PyPI package --
  `pip install` works with no extra steps. `beat_track.py` calls madmom-infer's Python API
  directly (`RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`), not a CLI binary (the
  original `madmom` package shipped a `DBNDownBeatTracker` console script; madmom-infer is a
  pure-Python library with no console script, so the old subprocess-based call was rewritten).
  One gap this bridges: madmom-infer has no ffmpeg-backed resampler, so non-44.1kHz audio is
  resampled with librosa before being handed to it (see `beat_track.py`'s module docstring).
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

## Device and lifecycle contract

All model-loading public paths (`sheetsage()`, `SheetSageSession`, and the CLI) accept
`device="auto"`, `"cpu"`, `"cuda"`, and `"cuda:N"`. `sheetsage()` and CLI calls with no
device preserve the historic CPU-handcrafted/CUDA-Jukebox selection; `auto` is explicit opt-in.
New sessions default to `auto`. `sheetsage.device.resolve_device()` resolves explicit `auto`
and rejects invalid or unavailable CUDA. Never add a fallback that changes an explicit request,
including after CUDA OOM.

`SheetSageSession.load()` constructs session-owned extractor/transducer components once;
`infer()` only consumes those components. `release()` and `close()` clear those references
without deleting downloaded assets. `cache_info()` uses `assets.resolve_asset_path()`, the
same resolver used by `retrieve_asset()`. `sheetsage/config/checkpoints.toml` is packaged and
is the runtime source of truth for all SheetSage assets. It preserves every URL/HuggingFace
resolver leg and records the existing checksum with its explicit algorithm (including SHA-1 CFG
digests and SHA-256 model/STEP digests); it supports generic `SHEETSAGE_ASSET_URL_<TAG>` overrides.
Focused contract coverage is
`uv run pytest tests/test_device_session.py`; complete verification remains `uv run pytest tests/`.

When changing `sheetsage/representations/jukebox.py`, `make_models.py`-adjacent code, or
anything in the feature-extraction path: re-run the CPU regression test at minimum, and the GPU
smoke test if you have a free GPU -- a prior campaign found the Jukebox path silently broken
end-to-end (see CHANGELOG's Unreleased section) precisely because nothing exercised it.

## Conventions

- Nav-style headers (what a file is, what it reads/is read by) on load-bearing modules -- keep
  them in sync when you restructure imports.
- CHANGELOG.md: add entries under `[Unreleased]` for behavior changes, not just version bumps.
- Commit messages / PRs: no emojis.
