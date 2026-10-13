"""Shared MIDI export stays readable across full-song and annotation-only spans."""

import io

import pretty_midi
import pytest

from sheetsage.transcription.result import (
    LabelInterval,
    MeterInterval,
    NoteEvent,
    TranscriptionResult,
)


def _result(*, notes=None, structure=None, meters=None):
    return TranscriptionResult(
        model="sheetsage2", analysis_start=0, analysis_end=1201,
        notes=notes or [], chords=[], beats=[], keys=[], meters=meters or [],
        structure=structure or [], statuses={}, native={},
    )


def test_six_hundred_second_midi_preserves_pitch_and_time_on_readback():
    note = NoteEvent(start=599.1234, end=601.4567, pitch=73, role="vocal")
    result = _result(notes=[note])
    midi = pretty_midi.PrettyMIDI(io.BytesIO(result.to_midi()))
    assert 1 <= midi.resolution < 9600
    actual = midi.instruments[0].notes[0]
    assert actual.pitch == note.pitch
    assert actual.start == pytest.approx(note.start, abs=1 / (2 * midi.resolution))
    assert actual.end == pytest.approx(note.end, abs=1 / (2 * midi.resolution))


@pytest.mark.parametrize("annotation", ["structure", "meter"])
def test_long_annotation_tail_sets_resolution_even_with_no_notes(annotation):
    result = _result(
        structure=[LabelInterval(1200.1234, 1201, "outro")] if annotation == "structure" else [],
        meters=[MeterInterval(1200.1234, 1201, 3, 4)] if annotation == "meter" else [],
    )
    midi = pretty_midi.PrettyMIDI(io.BytesIO(result.to_midi()))
    assert 1 <= midi.resolution < 9600
    assert midi.get_tempo_changes()[1].tolist() == pytest.approx([120.0], rel=1e-12)
    events = midi.text_events if annotation == "structure" else midi.time_signature_changes
    actual = events[-1]
    assert actual.time == pytest.approx(1200.1234, abs=1 / (2 * midi.resolution))
    if annotation == "structure":
        assert actual.text == "structure:outro"
    else:
        assert (actual.numerator, actual.denominator) == (3, 4)


def test_short_midi_retains_original_precision():
    result = _result(notes=[NoteEvent(1.1234, 2.3456, 60, "lead")])
    midi = pretty_midi.PrettyMIDI(io.BytesIO(result.to_midi()))
    assert midi.resolution == 9600


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -1.0, 2000001.0])
def test_unrepresentable_annotation_time_is_rejected(timestamp):
    result = _result(structure=[LabelInterval(timestamp, timestamp + 1, "outro")])
    with pytest.raises(ValueError, match="MIDI.*timestamp"):
        result.to_midi()
