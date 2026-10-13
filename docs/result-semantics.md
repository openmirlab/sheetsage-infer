# Shared transcription result interpretation

`SheetSageTranscriber` and `transcribe` expose schema version 1 through
`result_semantics(model)`, `model.semantics` and the `semantics` snapshot in every
`TranscriptionResult`. Discovery is offline and imports no model stack. The
existing native APIs return their original objects; this declaration covers
conversion into the shared result. Native tensor interpretation is not claimed.

## Contract

Times use seconds on the supplied audio's clock. An independently cropped file
starts its own clock; its offset in a parent recording is unknown. Intervals use
[start,end); beat timestamps are estimated point events. No model latency, sample
accuracy or uniform feature grid is implied. `analysis_start/analysis_end` bound
the analyzed score (V1) or decoded audio after `max_seconds` (V2).

Notes use MIDI integer pitches with no octave recentering, no estimated velocity
and no confidence. Roles are V1 `lead`, V2 `vocal` (track 0) and `instrumental`
(track 1). Nonpositive note/label spans are omitted, matching usable event export
semantics. Sparse labels persist until the next event of the same field or the
analysis end. An initial unlabeled gap is retained. The last same-time update
wins; adjacent equal labels are retained rather than merged.

Labels remain scoped to the selected model. V1 chord labels use `root:quality`
with eight decoder families (maj/min/min7/7/maj7/sus4/dim/aug). V1 chord
`root_pitch_class` is 0=C and `interval_steps` are successive pitch gaps, not
root-relative offsets. These optional fields are null for V2 because no additional
chord parsing is performed; the raw label is preserved. V2 key/chord spelling can
differ from LAB's key-aware respelling. Structure preserves the pinned model's
23 native labels, including combined labels, without the lossy MIREX reduction.
No genre-independent label accuracy or calibrated confidence is asserted.

`statuses[field]` distinguishes `ok` (including empty), `unsupported`,
`not_requested` and `failed` (reason included). The latter three have a null field.
Malformed requested fields fail independently, so invalid rhythm does not erase
valid structure. `ok` means conversion succeeded, not that the model annotated
all music correctly or covered every instant. Upstream warnings/windows are
retained in execution evidence; the full original object is `native`.

## Source trace and profiles

V1 native source at the pre-facade commit `f0b9c6b` remains unchanged:

- `sheetsage/pipeline/steps.py:_format_lead_sheet` constructs
  `Note(p % 12,p // 12)` from predicted MIDI pitch. The inverse is
  `pc + 12*octave`; generic `Note.as_midi_pitch()` instead assumes a relative
  octave and adds 60. The adapter only accepts inference-result semantics.
- `sheetsage/align.py:create_beat_to_time_fn` interpolates/extrapolates actual
  `segment_beats/beat_times`. Tertiaries divide by 4. Beat times already contain
  segment offsets; adding `segment_start_hint` again would be incorrect.
- `LeadSheet` rounds score extent to a full measure. Label ends based on that
  extent may exceed original sequence/audio end. Melody offsets retain the
  decoder's next-onset/sequence-end duration. Context beats outside score bounds
  are omitted; original beat arrays remain in `native`.
- Both handcrafted and Jukebox profiles use this decoder. Segment hints, legacy
  behavior, chunk settings, meter/tempo hints, thresholds and enabled tasks are
  captured as effective options. Disabled melody/harmony is `not_requested`.
- Keys can fall back to C major on errors and tempo to 120 BPM. Native output does
  not distinguish fallback from a matching estimate. No stronger inference is
  added. Tempo stays in execution metadata; timing uses actual beat locations.

V2 runtime revision `b83c3d79a6d98bcffaf8f9ed112c90aa887887bc` is identified by
`sheetsage/config/sheetsage2.json`; upstream files are downloaded, not bundled:

- `generation_sheetsage2.py:stitched_window_events` already adds window offsets
  to event `time` and note `end_time` and resolves overlap acceptance. The adapter
  neither adds `window_start` nor treats `global_subbeat` as V1 beat coordinates.
- `exports_sheetsage2.py:interval_rows` and raw event fields define sparse label
  intervals. `labels_sheetsage2.py` defines the native structure vocabulary.
- Rhythm's `eighth_position` is zero-based in eighth notes. A one-based
  denominator-beat position is `(eighth_position * denominator / 8) + 1`.
  Nonintegral or out-of-range positions fail the beats field. Sparse meters carry
  forward; valid meter changes remain available independently of beat-grid errors.
- `default` and `paper` presets share raw event semantics. Native paper exports
  may extend short notes to 40 ms; the shared adapter deliberately retains decoded
  `end_time` for both profiles. Native accompaniment and quantized playback stay
  available in `native`, including when `melody_only=True` changes their rendering.
- All eight prompt tasks are accepted: timestamp, downbeat_meter, structure, key,
  chord_majmin/chord_full, melody_vocal/melody_full. Each slash pair is mutually
  exclusive. Timestamp is required; other missing tasks are `not_requested`.
  `(timestamp,structure)` is a supported request with no speed guarantee.
- The effective preset/prompts/dtype/overlap/lookahead/windows and warnings are
  retained. Checkpoint manifest hash and component revisions accompany results;
  custom manifests preserve their actual identifiers. A manifest override does
  not certify a different upstream decoder's compatibility with this adapter.

No second audio decoder/resampler is added: files/bytes are forwarded unchanged;
V1 URL and V2 stream/waveform inputs retain backend rules. V2 files use its
preset-specific decoder and waveforms require an explicit rate. The upstream paper
file reader uses torchaudio.info, absent in the tested 2.9.1/2.11.0 builds; the facade
reports this before inference and supports the paper waveform path. Input channel
layout is documented in README. Input normalization, model accuracy and downstream
application acceptance are separate from result conversion.

## Exports

JSON includes values, statuses, effective options, execution metadata and semantic
snapshot. Native MIDI bytes/tensors are deliberately excluded; callbacks become
names, not executable objects. JSON rejects nonfinite numeric values.

Shared MIDI uses a 120-BPM encoding clock and 9600 ticks per quarter: note times
round to the nearest tick (about52 microseconds). This tempo is not an estimate.
Notes retain roles/pitches, use generated velocity 100, and have no generated chord
accompaniment. Chord/key/structure labels become text events; meter changes become
MIDI time signatures. The richer interval/status contract is in JSON. Native MIDI
and ABC/LilyPond remain available for upstream rendering conventions.

## Evidence and limits

`tests/test_transcription_results.py` uses independent coordinates/pitches for
crop offsets, nonlinear beat timing, score extrapolation, MIDI pitches 21/60/108,
sparse/duplicate labels, both melody roles, 3/4→6/8 meter conversion, off-grid
rhythm, absent/empty/failed structure, paper-tail behavior and MIDI/JSON exports.
Temporarily adding 60 to pitch, adding 10 seconds to crop timing, or extending every
structure label to song end each makes the relevant regression fail.
`tests/test_transcription_api.py` checks routing, option validation, lifecycle and
load reuse with doubles. `tools/verify_transcription.py` compares real native
returns with the public facade and repeats calls, including optional tensors and
structure-only V2. It excludes elapsed time and peak GPU memory from equality.

A305-second repeated-real-audio/silence fixture exercised two unchanged 300-second
V2 windows on 2026-10-13 and produced 589 notes with valid shared bounds. That checks
stitching and conversion, not natural-song section accuracy: this fixture yielded
one structure interval. Natural-song structure quality, full Jukebox GPU parity,
and the complete supported platform/Python matrix remain unmeasured here.
