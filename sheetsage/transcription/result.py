"""Shared transcription values, interpretation and portable exports.

Owns the seconds/MIDI contract independently of native rendering conventions.
Reads: dataclasses, JSON, pathlib; pretty_midi only during MIDI export.
Read by: transcription adapters/api and public callers.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

_MIDI_MAX_PPQ = 9600
_MIDI_EVENT_TICK_BUDGET = 4_000_000


@dataclass(frozen=True)
class NoteEvent:
    start: float
    end: float
    pitch: int
    role: str


@dataclass(frozen=True)
class LabelInterval:
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class ChordInterval(LabelInterval):
    root_pitch_class: int | None = None
    interval_steps: tuple[int, ...] | None = None


@dataclass(frozen=True)
class BeatEvent:
    time: float
    position: int
    numerator: int
    denominator: int

    @property
    def is_downbeat(self):
        return self.position == 1


@dataclass(frozen=True)
class MeterInterval:
    start: float
    end: float
    numerator: int
    denominator: int


@dataclass(frozen=True)
class OutputStatus:
    state: str  # ok, unsupported, not_requested, failed
    reason: str | None = None


def result_semantics(model: str) -> dict:
    """Inspect the versioned shared contract without importing models or loading weights.

    Declarations describe conversion of native *inference* results, not arbitrary
    user-authored LeadSheet objects. Effective options accompany every result.
    """
    if model not in ("sheetsage1", "sheetsage2"):
        raise ValueError(f"Unknown model: {model!r}")
    return {
        "schema_version": 1,
        "model": model,
        "time": {
            "unit": "seconds", "origin": "supplied_audio_start",
            "parent_recording_offset": "unknown for externally cropped inputs",
            "intervals": "[start, end)",
            "extent": ("beat-derived rounded score extent; may exceed audio duration"
                       if model == "sheetsage1" else "decoded audio duration after max_seconds"),
            "basis": ("linear interpolation/extrapolation of actual beat map; no extra crop offset"
                      if model == "sheetsage1" else "stitched event time/end_time; no extra window offset"),
            "initial_label_gaps": "unannotated; not filled",
        },
        "notes": {
            "pitch_unit": "MIDI note number", "tuning": "model note vocabulary; no tuning estimate",
            "roles": ["lead"] if model == "sheetsage1" else ["vocal", "instrumental"],
            "offsets": ("next detected onset or sequence end" if model == "sheetsage1"
                        else "decoded end_time, retained for both default and paper presets"),
            "octave": "raw model pitch; no playback transposition",
            "confidence": "not produced", "velocity": "not predicted",
        },
        "labels": {
            "namespace": f"{model}.native",
            "chords": ("root:quality, eight decoder families; interval_steps are successive gaps"
                       if model == "sheetsage1" else "raw decoder labels, before key-aware LAB respelling"),
            "keys": ("major/minor estimate; native C-major fallback is indistinguishable"
                     if model == "sheetsage1" else "raw decoder labels"),
            "structure": ("unsupported" if model == "sheetsage1" else
                          "native structure labels preserved without MIREX reduction"),
            "interval_end": "next event of same field or analysis_end; zero spans omitted",
        },
        "beats": {"position": "one-based denominator-beat index; 1 is downbeat",
                  "timestamps": "estimated point events, not uniform frames",
                  "invalid_grid": "field failed; other outputs remain available"},
        "missing": {"null": "see field status: unsupported, not_requested or failed",
                    "empty_list": "requested and decoded with no events",
                    "ok": "conversion succeeded; not a model accuracy or complete-coverage claim"},
        "profiles": ({"features": ["handcrafted", "jukebox"],
                      "meaning_options": ["segment_start_hint", "segment_end_hint", "legacy_behavior",
                                          "beats_per_measure_hint", "beats_per_minute_hint",
                                          "detect_melody", "detect_harmony", "melody_threshold", "harmony_threshold"],
                      "tempo": "native rounded median measure BPM or fallback 120; not used for timing"}
                     if model == "sheetsage1" else
                     {"presets": ["default", "paper"],
                      "meaning_options": ["prompts", "max_seconds", "sampling_rate", "preset",
                                          "overlap_seconds", "lookahead_seconds", "dtype", "melody_only"],
                      "paper": "shared notes retain raw end_time; native MIDI/LAB can extend notes to 40 ms",
                      "melody_only": "native accompaniment/notation switch; does not remove shared chords",
                      "features": "tokens/tensors remain native; no shared tensor interpretation claimed"}),
        "midi": {"policy": "notes by role, chord/structure text, key text and meter events",
                 "velocity": "constant 100, generated", "tempo": "constant 120 encoding clock, not estimated tempo",
                 "resolution": "adaptive integer PPQ, 1 through 9600",
                 "resolution_policy": "9600 for zero extent; otherwise min(9600, floor(4000000 / (2 * latest exported second)))",
                 "event_tick_budget": _MIDI_EVENT_TICK_BUDGET,
                 "precision": "nearest tick; one tick is 1 / (2 * PPQ) seconds",
                 "timestamp_limits": "finite nonnegative seconds, at most 2000000; otherwise ValueError",
                 "accompaniment": "not synthesized; native MIDI remains available"},
    }


def _json_value(value):
    """JSON snapshot for configuration, without serializing native tensors/models."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if callable(value):
        return {"callable": getattr(value, "__qualname__", type(value).__name__)}
    return {"type": type(value).__name__, "serialized": False}


def _midi_resolution(notes, annotations, meters):
    """Bound tick-to-time allocations when readers reopen long event exports."""
    latest = 0.0
    timestamps = (
        [time for note in notes for time in (note.start, note.end)]
        + [interval.start for interval in annotations]
        + [meter.start for meter in meters]
    )
    for timestamp in timestamps:
        if not math.isfinite(timestamp) or not 0 <= timestamp <= _MIDI_EVENT_TICK_BUDGET / 2:
            raise ValueError(f"Invalid MIDI timestamp: {timestamp!r}; expected finite seconds from 0 to 2000000")
        latest = max(latest, timestamp)
    if latest <= _MIDI_EVENT_TICK_BUDGET / (2 * _MIDI_MAX_PPQ):
        return _MIDI_MAX_PPQ
    return math.floor(_MIDI_EVENT_TICK_BUDGET / (2 * latest))


@dataclass
class TranscriptionResult:
    model: str
    analysis_start: float
    analysis_end: float
    notes: list[NoteEvent] | None
    chords: list[ChordInterval] | None
    beats: list[BeatEvent] | None
    keys: list[LabelInterval] | None
    meters: list[MeterInterval] | None
    structure: list[LabelInterval] | None
    statuses: dict[str, OutputStatus]
    native: Any = field(repr=False)
    execution: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    semantics: dict = field(default_factory=dict)
    schema_version: int = field(default=1, init=False)

    def __post_init__(self):
        if not self.semantics:
            self.semantics = result_semantics(self.model)

    def to_dict(self):
        """Portable result including effective semantics; native bytes/tensors are excluded."""
        result = {}
        for item in fields(self):
            if item.name == "native":
                continue
            value = getattr(self, item.name)
            if item.name == "statuses":
                value = {k: asdict(v) for k, v in value.items()}
            elif item.name in ("notes", "chords", "beats", "keys", "meters", "structure"):
                value = None if value is None else [asdict(v) for v in value]
            result[item.name] = _json_value(value)
        return result

    def to_json(self, path=None):
        """Return JSON text and optionally save it. Does not rerun inference."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_midi(self, path=None):
        """Export shared notes/annotations, preserving seconds to MIDI tick precision.

        This is a portable event export; use native results for upstream voicing,
        octave adjustment, click tracks or notation. No synthesis is performed.
        Resolution is at most 9600 PPQ and decreases for long event spans to
        keep the final event within four million ticks at the 120 BPM clock.
        """
        import mido
        import pretty_midi

        if self.notes is None:
            raise ValueError("notes are unavailable; inspect statuses['notes'] or export JSON")
        annotations = [interval for name in ("chords", "keys", "structure")
                       for interval in getattr(self, name) or ()]
        resolution = _midi_resolution(self.notes, annotations, self.meters or ())
        midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=resolution)
        instruments = {}
        for note in self.notes:
            if not (math.isfinite(note.start) and math.isfinite(note.end)
                    and 0 <= note.start < note.end and 0 <= note.pitch <= 127):
                raise ValueError(f"Invalid note for MIDI: {note}")
            if note.role not in instruments:
                instruments[note.role] = pretty_midi.Instrument(program=0, name=note.role)
            instruments[note.role].notes.append(
                pretty_midi.Note(100, note.pitch, note.start, note.end))
        midi.instruments = list(instruments.values())
        for name in ("chords", "keys", "structure"):
            for interval in getattr(self, name) or ():
                midi.text_events.append(pretty_midi.Text(f"{name}:{interval.label}", interval.start))
        for meter in self.meters or ():
            if meter.denominator <= 0 or meter.denominator & (meter.denominator - 1):
                raise ValueError("MIDI requires a power-of-two meter denominator")
            midi.time_signature_changes.append(pretty_midi.TimeSignature(
                meter.numerator, meter.denominator, meter.start))
        stream = io.BytesIO()
        midi.write(stream)
        # Some PPQs make pretty_midi truncate the 120 BPM clock to 499999 us.
        # Preserve the exact encoding clock so drift cannot accumulate in long files.
        stream.seek(0)
        encoded = mido.MidiFile(file=stream)
        for track in encoded.tracks:
            for message in track:
                if message.type == "set_tempo":
                    message.tempo = 500000
        normalized = io.BytesIO()
        encoded.save(file=normalized)
        data = normalized.getvalue()
        if path is not None:
            Path(path).write_bytes(data)
        return data
