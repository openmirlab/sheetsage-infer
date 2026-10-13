"""Convert native inference values to the shared transcription contract.

Owns beat/second, octave, role and sparse-event interpretation. Never calls a
model or renderer and never modifies native results. Reads: result, align (V1
only), package checkpoint manifest for provenance; read by: transcription.api.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..__about__ import __version__
from .result import (
    BeatEvent,
    ChordInterval,
    LabelInterval,
    MeterInterval,
    NoteEvent,
    OutputStatus,
    TranscriptionResult,
)

_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_CHORD_FAMILIES = {(4, 3): "maj", (3, 4): "min", (3, 4, 3): "min7",
                   (4, 3, 3): "7", (4, 3, 4): "maj7", (5, 2): "sus4",
                   (3, 3): "dim", (4, 4): "aug"}
_KEY_SCALES = {(2, 2, 1, 2, 2, 2): "major", (2, 1, 2, 2, 1, 2): "minor"}
_FIELDS = ("notes", "chords", "beats", "keys", "meters", "structure")


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite event time")
    return value


def _integer(value):
    integer = int(value)
    if integer != value:
        raise ValueError(f"Expected an integer: {value!r}")
    return integer


def _note(start, end, pitch, role):
    start, end, pitch = _number(start), _number(end), _integer(pitch)
    if start < 0 or not 0 <= pitch <= 127:
        raise ValueError("Note time/pitch outside supported range")
    return NoteEvent(start, end, pitch, role) if end > start else None


def _v1_chord(start, end, chord):
    root, steps = chord
    if root is None:
        return ChordInterval(start, end, "N")
    steps = tuple(int(x) for x in steps)
    quality = _CHORD_FAMILIES.get(steps)
    if quality is None:
        raise ValueError(f"Unknown SheetSage1 inference chord: {steps}")
    return ChordInterval(start, end, f"{_PITCH_NAMES[root]}:{quality}", int(root), steps)


def _v1_key(start, end, key):
    root, steps = key
    mode = _KEY_SCALES.get(tuple(steps))
    if mode is None:
        raise ValueError(f"Unknown SheetSage1 inference scale: {steps}")
    return LabelInterval(start, end, f"{_PITCH_NAMES[root]}:{mode}")


def _apply(result, field, convert, requested=True):
    if not requested:
        result.statuses[field] = OutputStatus("not_requested", "Disabled by effective task options")
        return
    try:
        value = convert()
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        result.statuses[field] = OutputStatus("failed", str(exc))
        result.warnings.append(f"{field}: {exc}")
    else:
        setattr(result, field, value)
        result.statuses[field] = OutputStatus("ok")


def _empty(model, start, end, native, options):
    return TranscriptionResult(model=model, analysis_start=start, analysis_end=end,
                               **dict.fromkeys(_FIELDS), statuses={}, native=native,
                               execution={"options": dict(options), "package_version": __version__})


def from_sheetsage1(native, options):
    """Normalize inference tuples, including tuples with optional intermediaries."""
    from ..align import create_beat_to_time_fn

    lead_sheet, segment_beats, beat_times = native[:3]
    meters, tempos, keys, harmony, melody, length = lead_sheet
    beat_to_time = create_beat_to_time_fn(segment_beats, beat_times)
    def to_time(tertiary):
        return _number(beat_to_time(tertiary / 4))
    start, end = to_time(0), to_time(length)
    result = _empty("sheetsage1", start, end, native, options)
    manifest = Path(__file__).parents[1] / "config/checkpoints.toml"
    result.execution["provenance"] = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    result.execution.update(profile="jukebox" if options.get("use_jukebox", False) else "handcrafted",
                            native_tempo_changes=[(int(t), int(tempo[0])) for t, tempo in tempos],
                            extent_basis="LeadSheet rounded measure extent, not source duration")

    def intervals(changes, make):
        return [make(to_time(t), to_time(changes[i + 1][0] if i + 1 < len(changes) else length), value)
                for i, (t, value) in enumerate(changes)]

    def notes():
        output = [_note(to_time(t), to_time(t + duration), int(note[0]) + 12 * int(note[1]), "lead")
                  for t, duration, note in melody]
        return [note for note in output if note is not None]

    def beats():
        numerator = int(meters[0][1][0])
        return [BeatEvent(_number(time), int(beat) % numerator + 1, numerator, 4)
                for beat, time in zip(segment_beats, beat_times, strict=True)
                if 0 <= beat < length / 4]

    _apply(result, "notes", notes, options.get("detect_melody", True))
    _apply(result, "chords", lambda: intervals(harmony, _v1_chord), options.get("detect_harmony", True))
    _apply(result, "keys", lambda: intervals(keys, _v1_key))
    _apply(result, "meters", lambda: intervals(meters, lambda a, b, m: MeterInterval(a, b, int(m[0]), 4)))
    _apply(result, "beats", beats)
    result.statuses["structure"] = OutputStatus("unsupported", "SheetSage1 has no structure model")
    return result


def _v2_intervals(events, field, end, make=LabelInterval):
    updates = []
    for event in events:
        if field in event["values"]:
            start = _number(event["time"])
            if not 0 <= start <= end:
                raise ValueError(f"{field} event outside decoded audio extent")
            updates.append((start, event["values"][field]))
    # Stable sorting retains the last state update at identical timestamps.
    updates.sort(key=lambda x: x[0])
    output = []
    for index, (start, value) in enumerate(updates):
        stop = updates[index + 1][0] if index + 1 < len(updates) else end
        if stop > start:
            output.append(make(start, stop, value))
    return output


def _label(start, end, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty native label")
    return LabelInterval(start, end, value)


def _chord(start, end, value):
    value = _label(start, end, value).label
    return ChordInterval(start, end, value)


def _meter(value):
    numerator, denominator = (_integer(x) for x in value)
    if numerator < 1 or denominator < 1:
        raise ValueError("Meter values must be positive")
    return numerator, denominator


def _v2_meters(events, end):
    changes = []
    last = None
    for event in events:
        rhythm = event["values"].get("rhythm", {})
        if "meter" in rhythm:
            meter = _meter(rhythm["meter"])
            if meter != last:
                changes.append({"time": event["time"], "values": {"meter": meter}})
                last = meter
    return _v2_intervals(changes, "meter", end, lambda a, b, m: MeterInterval(a, b, *m))


def _v2_beats(events, end):
    meter, beats = None, []
    for event in events:
        rhythm = event["values"].get("rhythm", {})
        if "meter" in rhythm:
            meter = _meter(rhythm["meter"])
        if "eighth_position" not in rhythm:
            continue
        if meter is None:
            raise ValueError("Rhythm position has no preceding meter")
        numerator, denominator = meter
        scaled = _integer(rhythm["eighth_position"]) * denominator
        if scaled % 8 or not 0 <= scaled // 8 < numerator:
            raise ValueError(f"Off-grid rhythm position in {numerator}/{denominator}")
        time = _number(event["time"])
        if not 0 <= time <= end:
            raise ValueError("Beat outside decoded audio extent")
        beats.append(BeatEvent(time, scaled // 8 + 1, numerator, denominator))
    return beats


def _v2_notes(events, end):
    result = []
    for event in events:
        start = _number(event["time"])
        if not 0 <= start <= end:
            raise ValueError("Note onset outside decoded audio extent")
        for note in event["values"].get("melody", ()):
            track = _integer(note["track"])
            if track not in (0, 1):
                raise ValueError(f"Unknown melody track: {track}")
            # Decoder has already clipped/stamped end_time. Preserve it even when
            # the paper-native renderer would re-extend a short note to40ms.
            value = _note(start, note["end_time"], note["pitch"], ("vocal", "instrumental")[track])
            if value is not None:
                result.append(value)
    return sorted(result, key=lambda n: (n.start, n.end, n.pitch, n.role))


def _v2_provenance(options):
    manifest_path = options.get("checkpoint_config") or Path(__file__).parents[1] / "config/sheetsage2.json"
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "components": {name: {k: config[k] for k in ("repository", "revision") if k in config}
                           for name, config in manifest["components"].items()}}


def from_sheetsage2(native, options):
    """Normalize stitched raw events, keeping native exports/tensors untouched."""
    end = _number(native["duration_seconds"])
    if end <= 0:
        raise ValueError("Invalid decoded audio duration")
    result = _empty("sheetsage2", 0.0, end, native, options)
    result.warnings.extend(native.get("warnings", ()))
    result.warnings.extend(native.get("diagnostics", ()))
    if native.get("abc_error"):
        result.warnings.append(f"native ABC: {native['abc_error']}")
    result.execution.update({key: native[key] for key in (
        "prompts", "preset", "dtype", "melody_only", "window_seconds", "overlap_seconds",
        "lookahead_seconds", "windows", "elapsed_seconds", "peak_gpu_mib") if key in native})
    result.execution["provenance"] = _v2_provenance(options)
    prompts = set(native["prompts"])
    events = native["events"]
    _apply(result, "notes", lambda: _v2_notes(events, end), bool(prompts & {"melody_full", "melody_vocal"}))
    _apply(result, "chords", lambda: _v2_intervals(events, "chord", end, _chord),
           bool(prompts & {"chord_full", "chord_majmin"}))
    _apply(result, "keys", lambda: _v2_intervals(events, "key", end, _label), "key" in prompts)
    _apply(result, "structure", lambda: _v2_intervals(events, "structure", end, _label), "structure" in prompts)
    _apply(result, "meters", lambda: _v2_meters(events, end), "downbeat_meter" in prompts)
    _apply(result, "beats", lambda: _v2_beats(events, end), "downbeat_meter" in prompts)
    return result
