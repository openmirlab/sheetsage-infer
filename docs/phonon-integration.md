# Phonon integration: SheetSage1, SheetSage2 and song structure

Use the remote `openmirlab/sheetsage-infer` source with the `sheetsage2` extra.
The shared API keeps the original transcription capabilities and adds V2 song
structure. This guide describes integration work for Phonon; it does not claim
that Phonon's implementation is already complete.

## Remote dependency

The tested integration merge is `7f90ab3433f3934926b6c5c9eab81206aa9faf6e`.
For a reproducible application dependency:

```bash
uv add "sheetsage-infer[sheetsage2] @ git+https://github.com/openmirlab/sheetsage-infer.git@7f90ab3433f3934926b6c5c9eab81206aa9faf6e"
```

Or install that same remote revision with pip:

```bash
python -m pip install "sheetsage-infer[sheetsage2] @ git+https://github.com/openmirlab/sheetsage-infer.git@7f90ab3433f3934926b6c5c9eab81206aa9faf6e"
```

For subsequent updates, resolve `@master`, run the application checks below,
and commit the resulting dependency lock. Replace an existing local/editable
SheetSage dependency with the Git source, including any `[tool.uv.sources]` path
or workspace override. Package version `0.2.1` alone does not identify this
integration: verify the locked Git revision and the exported shared API. With
pip, use `--force-reinstall --no-deps` for the command above if replacing a
same-version local build after its dependencies have already been installed.

Follow the [README installation guidance](../README.md#experimental-sheetsage2)
for matching Torch/TorchAudio CPU or CUDA builds and FFmpeg. The backend remains
experimental. Existing PyPI snapshots do not provide this integration.

## Common contract and individual capabilities

| Phonon data or action | Shared API | Model-specific behavior |
|---|---|---|
| Melody | `result.notes` | V1 role `lead`; V2 roles `vocal` and `instrumental` |
| Chords | `result.chords` | Preserve model-native labels; V2 root/interval metadata is optional/null |
| Beat grid and downbeats | `result.beats`, each event's `is_downbeat` | Positions are one-based denominator beats |
| Key and meter | `result.keys`, `result.meters` | Preserve intervals and model-specific estimates |
| Song sections | `result.structure` | V2 intervals have `start`, `end`, `label`; V1 is unsupported |
| Common exports | `result.to_json()`, `result.to_midi()` | JSON retains the full shared contract; shared MIDI needs available notes |
| Native capabilities | Typed options and `result.native` | V1 Jukebox, segment hints and LilyPond; V2 ABC, LAB, separate MIDI parts and tensors |

Keep the existing V1 workflow while adding a model selector. Read
`model.capabilities` for support and `result.statuses[field]` for each call's
outcome: `ok`, `unsupported`, `not_requested`, or `failed`. An `ok` empty list is
a valid empty result; the other states use `None` and may include a reason.
A field conversion or native ABC failure must not erase other successful fields.

Times are seconds relative to the audio passed to the API; pitches are MIDI note
numbers. If Phonon physically crops an audio file, map those seconds back using
that crop's offset. V1 segment-hint offsets and V2 window offsets are already
included by the backend. Preserve original structure labels, including combined
labels, and keep adjacent equal intervals unless a separate display layer merges
them. See [result semantics](result-semantics.md) for score bounds, label
namespaces, timing precision and MIDI policies.

## Reusable analysis and structure-only task

```python
from sheetsage import SheetSage2Options, SheetSageTranscriber

# Default prompts produce all shared fields, including structure.
with SheetSageTranscriber("sheetsage2", device="auto") as model:
    result = model.infer("song.wav")
    result.to_json("analysis.json")
    result.to_midi("analysis.mid")

# A separate structure-only workflow.
options = SheetSage2Options(prompts=("timestamp", "structure"))
with SheetSageTranscriber("sheetsage2", options=options,
                         require_structure=True) as model:
    result = model.infer("song.wav")
    if result.statuses["structure"].state == "ok":
        sections = result.structure
    result.to_json("structure.json")
```

In a long-lived Phonon service, retain one loaded session for sequential calls
and close it when released by the application. Serialize calls to the same
session. Select `model="sheetsage1"` with `SheetSage1Options` for the original
backend; use `SheetSage2Options` for V2. Per-call options replace the complete
option object; use `dataclasses.replace` to change selected inference settings.
Session configuration stays fixed at construction.

Structure-only output has no shared notes, so export it as JSON. Full prompts
provide structure together with melody/chords/beats/keys/meters. Preserve
`result.native` for the richer original exporters rather than recreating their
notation or accompaniment from shared events.

Both backends accept local paths and encoded bytes. V2 also accepts binary
streams and NumPy/Torch waveforms with an explicit sampling rate and
channels-first layout. Use decoded waveforms for the paper preset with modern
TorchAudio; see the README for the upstream file-reader limitation.

## Phonon acceptance checks

- Resolve and install the dependency from its remote Git revision in a clean
  application environment; confirm the lock has no local SheetSage path override.
- Keep the existing V1 melody/chord workflow and native exports working. Display
  its structure capability as unsupported.
- Exercise V2 full analysis and structure-only analysis on actual audio. Render
  section start/end/label on the same audio timeline as the shared notes/beats.
- Check all four field statuses, valid empty results, and independent failures.
  Verify an unavailable ABC score still permits successful structure output.
- Check JSON and available-note MIDI exports, session reuse/cleanup, and a
  Phonon-created crop without applying offsets twice.

The library's [verification report](compatibility/sheetsage2-verification.md)
records Python 3.10–3.12 tests, real CPU/GPU inference, input/export checks and a
315-second two-window case. That evidence establishes library integration
behavior; Phonon's own wiring and annotated structure accuracy need separate
validation.
