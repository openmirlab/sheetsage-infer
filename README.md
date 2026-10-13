# SheetSage-Infer

Inference-only music transcription with **SheetSage1** and experimental
**SheetSage2**: a shared Python API for melody, chords, beats, key and meter,
plus timed song sections with V2. Native notation and model-specific outputs
remain available.

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10–3.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Install from GitHub; existing PyPI releases are historical snapshots. The former
package name was `openmirlab-sheetsage-infer`; the import remains `sheetsage`.
This package provides inference, dependency maintenance and adapters, without
training code or bundled model weights/datasets.

## Acknowledgments

The models and research belong to their original authors:

- **[SheetSage](https://github.com/chrisdonahue/sheetsage)** — Chris Donahue,
  John Thickstun and Percy Liang; *Melody Transcription via Generative Pre-Training*
  ([ISMIR 2022 paper](https://arxiv.org/abs/2212.01884)).
- **[M·A·P's SheetSage2](https://huggingface.co/m-a-p/SheetSage2)** — Junyan Jiang,
  Ruibin Yuan, Jiahao Pan, Wei Xue, Yike Guo, Gus Xia and Yann LeCun;
  *SheetSage2: Coherent Lead-Sheet Transcription with Synthetic Supervision*
  ([technical report](https://arxiv.org/abs/2610.05336)), using
  [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong).
- **[OpenAI Jukebox](https://github.com/openai/jukebox)** and
  **[CPJKU madmom](https://github.com/CPJKU/madmom)** underpin V1's features and
  beat tracking. This package uses OpenMIRLab's maintained
  [jukebox-infer](https://github.com/openmirlab/jukebox-infer) and
  [madmom-infer](https://github.com/openmirlab/madmom-infer) dependencies.

This is an independent OpenMIRLab integration, without upstream affiliation or
endorsement. Cite the relevant original work using [Citation](#citation).

## Installation

Requires Python >=3.10 (tested on 3.10–3.12) and FFmpeg on `PATH`. LilyPond is
optional for V1 PDF export. Install the backend you need:

```bash
# SheetSage1
python -m pip install "sheetsage-infer @ git+https://github.com/openmirlab/sheetsage-infer.git@master"

# Both backends
python -m pip install "sheetsage-infer[sheetsage2] @ git+https://github.com/openmirlab/sheetsage-infer.git@master"
```

For a uv project, use `uv add` with the same dependency string. Lock the resolved
Git commit for reproducibility; the tested integration merge is
`7f90ab3433f3934926b6c5c9eab81206aa9faf6e` (replace `@master` with that SHA).

The pinned Jukebox dependency requires **Torch >=2.13**, including for a fresh
V2 install. V2 also requires TorchAudio >=2.8 and Transformers >=4.45.2,<5;
see the [Transformers compatibility note](docs/compatibility/sheetsage2-transformers.md).
For CPU-only V2, install compatible CPU Torch/TorchAudio builds **before** the
package command above:

```bash
python -m pip install 'torch>=2.13' 'torchaudio>=2.8' --index-url https://download.pytorch.org/whl/cpu
```

Before a fresh V1 installation, check the recorded checkpoint-access limitation
in [Weights and offline use](#weights-and-offline-use).

## Shared transcription and song structure

```python
from sheetsage import SheetSageTranscriber

with SheetSageTranscriber("sheetsage2", device="auto",
                         require_structure=True) as model:
    result = model.infer("song.wav")

for section in result.structure or []:
    print(section.start, section.end, section.label)
result.to_json("analysis.json")
result.to_midi("transcription.mid")
```

Select `model="sheetsage1"` for V1 (the default), omitting
`require_structure=True`. Requiring unsupported or disabled structure fails
before model loading. Inspect `model.capabilities` for model support and
`result.statuses` for each call's outcome.

| Shared field | SheetSage1 | SheetSage2 |
|---|---|---|
| `notes` | Lead melody | Vocal and instrumental melody |
| `chords` | Eight chord families | Full or major/minor vocabulary |
| `beats` | Beat-derived rhythm | Decoded rhythm |
| `keys` | Global estimate/fallback | Key events |
| `meters` | Single 3/4 or 4/4 | Meter changes |
| `structure` | Unsupported | Intro, verse, chorus and other native labels |
| `native` | Original LeadSheet/beat tuple | Original dictionary, including ABC/MIDI/LAB/tokens/tensors |

Times are supplied-audio seconds, intervals are `[start, end)`, and note pitches
are MIDI numbers with roles. V1 segment offsets and V2 window offsets are already
included; externally cropped audio starts its own clock. V1's rounded score end
may exceed the audio duration. Labels retain their model-specific vocabulary.

Each field's status is `ok`, `unsupported`, `not_requested`, or `failed` with a
reason. `[]` is a successful empty result; `None` requires checking the status.
Fields can fail independently. Successful conversion does not certify musical
accuracy or complete annotation coverage. Full coordinate/label details live in
[result semantics](docs/result-semantics.md), also inspectable offline through
`result_semantics(model)` or `model.semantics`.

### Structure only

```python
from sheetsage import SheetSage2Options, SheetSageTranscriber

options = SheetSage2Options(prompts=("timestamp", "structure"))
with SheetSageTranscriber("sheetsage2", options=options,
                         require_structure=True) as model:
    result = model.infer("song.wav")
assert result.statuses["structure"].state == "ok"
result.to_json("structure.json")
```

Default V2 prompts request all shared fields. Structure-only calls export JSON;
shared MIDI requires available notes. ABC may be unavailable without rhythm/key
prompts while structure remains usable.

### Options, inputs and exports

Use `SheetSage1Options` for Jukebox, segment/beat hints, detection thresholds and
intermediaries; use `SheetSage2Options` for prompts, precision, windows, presets,
cache paths and tensors. Per-call `infer(audio, options=...)` replaces the whole
option object; use `dataclasses.replace(model.options, ...)` for selected changes.
Session configuration is fixed at construction. V2 `native_output_dir` saves
upstream artifacts; `melody_only=True` controls notation/accompaniment, not tasks.

| Input | V1 | V2 |
|---|---|---|
| Local path or encoded bytes | Yes | Yes |
| URL | Yes | No |
| Binary stream | No | Yes |
| NumPy/Torch waveform | No | Yes, with `sampling_rate` and channels-first layout |

For V2, `sampling_rate` applies only to waveforms. The upstream `paper` file
reader needs `torchaudio.info()`, absent in tested modern TorchAudio builds;
use decoded waveforms for that preset or `default` for encoded input.

`to_json()` returns text and `to_midi()` returns bytes; both accept an output path.
JSON includes configuration, semantics and field statuses, excluding native
binary/tensor payloads. Shared MIDI preserves note roles/pitches with generated
velocity and text annotations; native MIDI retains upstream accompaniment and
rendering conventions. Access V1 LilyPond with `result.native[0].as_lily()` and
V2 ABC with `result.native["abc"]` (check `abc_error`).

### Device and reusable sessions

`SheetSageTranscriber`, `SheetSageSession` and `SheetSage2Session` construct lazily.
Call `load()` before `infer()`, or use a context manager. Sequential calls reuse
the model; concurrent calls on one session are unsupported. `release()` permits
reload, `close()` is terminal/idempotent, and failed loads can retry. Status is
`new`, `ready`, `failed`, `released`, or `closed`; `cache_info()` never downloads.

Sessions default to `device="auto"` (CUDA when available, otherwise CPU). Explicit
`cpu`, `cuda` and `cuda:N` are honored or fail; MPS is unsupported. Legacy V1
`sheetsage()`/CLI defaults remain CPU for handcrafted features and CUDA for
Jukebox. `transcribe(audio, model=..., options=...)` is the shared one-shot API;
`sheetsage()` and `sheetsage2()` retain their native contracts. Each one-shot call
creates and closes a fresh session.

## Native SheetSage1

```python
from pathlib import Path
from sheetsage import SheetSageSession
from sheetsage.align import create_beat_to_time_fn
from sheetsage.utils import engrave

with SheetSageSession(device="cpu") as session:
    lead_sheet, beats, times = session.infer("song.wav")

Path("score.ly").write_text(lead_sheet.as_lily())
Path("score.mid").write_bytes(lead_sheet.as_midi(create_beat_to_time_fn(beats, times)))
# Optional: requires LilyPond.
Path("score.pdf").write_bytes(engrave(lead_sheet.as_lily(), out_format="pdf"))
```

Enable Jukebox with `SheetSageSession(device="cuda", use_jukebox=True)` or
`SheetSage1Options(use_jukebox=True)`. Plan for at least 12 GB VRAM; the 5B prior
requires reported total audio length in `[60, 600)` seconds. Very short input can
raise an assertion. Jukebox comes from the installed `jukebox-infer` dependency.

### SheetSage1 command-line interface

```bash
python -m sheetsage.infer audio.mp3 --segment_start_hint 30 --segment_end_hint 60 --output_dir output
python -m sheetsage.infer --help
```

## Experimental SheetSage2

Install the `sheetsage2` extra as shown in [Installation](#installation).
For native outputs and options, call the independent backend directly:

```python
from sheetsage import SheetSage2Session

with SheetSage2Session(device="auto") as session:
    result = session.infer("song.wav", output_dir="score")
print(result["abc"])
```

The dictionary retains `events`, `midis` (separate MIDI parts), `labs`
(annotations), tokens and requested logits/scores/embeddings. `output_dir` saves
ABC, MIDI and annotation files; omission keeps results in memory. Check
`abc_error` for notation failures. PDF/PNG/SVG and synthesized-audio rendering
are outside this adapter.

Upstream handles mono conversion, 24 kHz resampling and overlapping windows.
`max_seconds`, `preset`, `dtype`, `export_logits`, `export_scores` and
`export_embeddings` pass through. CUDA defaults to BF16 autocast; CPU and adapter
merging use FP32. This backend downloads verified upstream code at runtime;
see [License](#license) for its separate terms.

## Weights and offline use

Weights and datasets are excluded from the package's wheel and source archive.
Checkpoint manifests own URLs, revisions and digests:

| Backend | Manifest | Default cache | Overrides |
|---|---|---|---|
| V1 | [checkpoints.toml](sheetsage/config/checkpoints.toml) | `~/.sheetsage` | `SHEETSAGE_CACHE_DIR`, `SHEETSAGE_ASSET_URL_<TAG>` |
| V2 | [sheetsage2.json](sheetsage/config/sheetsage2.json) | `~/.sheetsage/sheetsage2` | `cache_dir`, `SHEETSAGE2_CACHE_DIR` (otherwise under the V1 cache root) |

**V1 access limitation:** on 2026-10-03, all seven handcrafted checkpoint URLs
returned S3 HTTP 403 and their Hugging Face fallbacks returned HTTP 401 without
credentials. Fresh inference needs separately available, checksum-verified
files. See [cached verification](tools/README.md#delivery-and-portable-cpu-verification)
for the recorded access and baseline evidence.

**V2:** first load fetches about 2.76 GB of weights plus pinned Python/config
files. `load()` verifies every SHA-256 before executing upstream code; model
loaders then resolve only local files. `cache_info()` reports locations,
URLs and sizes, but does not verify integrity. Inspect destinations offline:

```python
from sheetsage import SheetSage2Session

session = SheetSage2Session(cache_dir="/data/sheetsage2")
for component in session.cache_info()["components"].values():
    for artifact in component["artifacts"].values():
        print(artifact["url"], "->", artifact["path"])
session.close()
```

Populate those paths, then use `local_files_only=True`. Alternatively, pass
`model_path` and `base_model_path` for read-only directories containing every
pinned adapter/MERT2 file; merged `save_pretrained` checkpoints are incompatible.
A trusted `checkpoint_config` can override artifact identities/URLs. Corrupt
files fail explicitly and must be removed or replaced before retrying.

## License

[LICENSE](LICENSE) covers this repository's code; [NOTICE](NOTICE) records the
separate upstream terms:

- Repository code, including adapted SheetSage code: **MIT**.
- SheetSage1 weights and HookTheory-derived data: **CC BY-NC-SA 3.0**.
- SheetSage2/MERT2 weights: model cards label them **CC BY-NC 4.0**. Redistribution
  terms for the distinct SheetSage2 Python source remain unresolved; this package
  downloads it at runtime rather than redistributing it. Our MIT license does
  not cover these downloads.
- Jukebox weights retain OpenAI's terms; madmom source is BSD-2-Clause and its
  pretrained models are CC BY-NC-SA 4.0.

## Citation

Cite the original work for the backend you use:

```bibtex
@inproceedings{donahue2022melody,
  title={Melody Transcription via Generative Pre-Training},
  author={Donahue, Chris and Thickstun, John and Liang, Percy},
  booktitle={Proceedings of the 23rd International Society for Music Information Retrieval Conference (ISMIR)},
  year={2022}
}

@misc{jiang2026sheetsage2,
  title={SheetSage2: Coherent Lead-Sheet Transcription with Synthetic Supervision},
  author={Jiang, Junyan and Yuan, Ruibin and Pan, Jiahao and Xue, Wei and Guo, Yike and Xia, Gus and LeCun, Yann},
  year={2026},
  eprint={2610.05336},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```

## Development

```bash
git clone https://github.com/openmirlab/sheetsage-infer.git
cd sheetsage-infer
python -m pip install -e '.[dev,sheetsage2]' build
python -m pytest tests/ tools/test_dataset_examples.py
python -m build
python tools/check_distribution_boundary.py dist/*.whl dist/*.tar.gz
```

CI checks fresh CPU installs on Python 3.10–3.12, contract tests, wheel-from-sdist
builds and installed files outside the checkout. Model tests need separately
populated caches; skipped fixtures are not model-parity evidence. See the
[verification commands](tools/README.md) and [integration report](docs/compatibility/sheetsage2-verification.md)
for real/silence CPU/GPU checks, full-song/multiwindow results and remaining gaps.
These checks establish integration behavior, not transcription/structure accuracy.

[Examples](examples/) are checkout-only. Dataset examples acquire data explicitly
with checksum verification; HookTheory/RWC catalogs are excluded from distributions
and their tags are absent from the installed asset API. See [CLAUDE.md](CLAUDE.md)
for architecture and contribution guidance, [CHANGELOG.md](CHANGELOG.md) for
history, and [GitHub Issues](https://github.com/openmirlab/sheetsage-infer/issues)
for support.
