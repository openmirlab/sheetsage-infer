"""Independent coordinate, missing-value and export checks for shared results."""

import copy
import io
import json

import pytest

from sheetsage.transcription.adapters import from_sheetsage1, from_sheetsage2
from sheetsage.transcription.result import result_semantics

FULL = ["timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full"]


def event(time, **values):
    return {"time": time, "values": values, "window_start": 100.0, "global_subbeat": 99}


def native2(events, prompts=FULL, **kwargs):
    return {"events": events, "duration_seconds": 10.0, "prompts": list(prompts),
            "warnings": [], "preset": "default", **kwargs}


def native1():
    # Predicted pitch60 -> Note(0,5). Times include a10-second source crop and
    # nonuniform beat spacing. Context beats -1 and5 must not leak into output.
    from sheetsage.theory import LeadSheet

    lead = LeadSheet([(0, (4, 2, 2))], [(0, (120,))],
                     [(0, (0, (2, 2, 1, 2, 2, 2)))], [(0, (0, (4, 3))), (8, (7, (4, 3, 3)))],
                     [(0, 4, (0, 5)), (4, 4, (9, 1)), (8, 8, (0, 9))], 16)
    return lead, [-1, 0, 1, 2, 3, 4, 5], [9, 10, 10.5, 11.5, 12, 13, 14]


def test_v1_original_clock_pitch_and_intervals():
    native = native1()
    result = from_sheetsage1(native, {"segment_start_hint": 10})
    assert result.native is native
    assert (result.analysis_start, result.analysis_end) == (10, 13)
    assert [(n.start, n.end, n.pitch, n.role) for n in result.notes] == [
        (10, 10.5, 60, "lead"), (10.5, 11.5, 21, "lead"), (11.5, 13, 108, "lead")]
    assert [(c.start, c.end, c.label) for c in result.chords] == [(10, 11.5, "C:maj"), (11.5, 13, "G:7")]
    assert result.chords[0].interval_steps == (4, 3)
    assert [b.time for b in result.beats] == [10, 10.5, 11.5, 12]
    assert [b.position for b in result.beats] == [1, 2, 3, 4]
    assert result.beats[0].is_downbeat
    assert result.structure is None
    assert result.statuses["structure"].state == "unsupported"
    assert result.keys[0].label == "C:major"


def test_v1_disabled_outputs_and_intermediaries():
    native = (*native1(), {"large": "intermediaries"})
    result = from_sheetsage1(native, {"detect_melody": False, "detect_harmony": False})
    assert result.native is native
    assert result.notes is result.chords is None
    assert result.statuses["notes"].state == result.statuses["chords"].state == "not_requested"


@pytest.mark.parametrize("numerator", [3, 4])
def test_v1_meter_and_extrapolated_rounded_score(numerator):
    from sheetsage.theory import LeadSheet

    lead = LeadSheet([(0, (numerator, 2, 2))], [(0, (120,))],
                     [(0, (9, (2, 1, 2, 2, 1, 2)))], [], [], 5)
    result = from_sheetsage1((lead, [0, 1], [2, 2.5]), {})
    assert result.analysis_end == 2 + numerator * .5
    assert result.keys[0].label == "A:minor"
    assert result.meters[0].numerator == numerator
    assert result.notes == []
    assert result.statuses["notes"].state == "ok"


@pytest.mark.parametrize("preset", ["default", "paper"])
def test_v2_raw_time_roles_and_tail_not_renderer_time(preset):
    native = native2([
        event(1, melody=[{"pitch": 48, "track": 0, "end_time": 2.5},
                         {"pitch": 80, "track": 1, "end_time": 3}]),
        event(9.99, melody=[{"pitch": 60, "track": 0, "end_time": 10}]),
    ], preset=preset)
    before = copy.deepcopy(native)
    result = from_sheetsage2(native, {})
    assert result.native is native and native == before
    assert [(n.start, n.end, n.pitch, n.role) for n in result.notes] == [
        (1, 2.5, 48, "vocal"), (1, 3, 80, "instrumental"), (9.99, 10, 60, "vocal")]
    assert result.statuses["structure"].state == "ok" and result.structure == []


def test_structure_sparse_labels_duplicates_and_independent_fields():
    native = native2([
        event(1, structure="intro", key="Db:major", chord="C#:maj"),
        event(2, melody=[]),
        event(4, structure="verse"),
        event(4, structure="intro and verse"),
        event(8, structure="chorus", chord="N"),
        event(10, structure="outro"),
    ])
    result = from_sheetsage2(native, {})
    assert [(s.start, s.end, s.label) for s in result.structure] == [
        (1, 4, "intro"), (4, 8, "intro and verse"), (8, 10, "chorus")]
    assert [(c.start, c.end, c.label) for c in result.chords] == [(1, 8, "C#:maj"), (8, 10, "N")]
    assert result.keys[0].label == "Db:major"
    assert result.keys[0].end == 10


def test_structure_only_and_abc_failure_are_usable():
    result = from_sheetsage2(native2([event(0, structure="silence")],
                                    prompts=["timestamp", "structure"], abc_error="No beats"), {})
    assert result.structure[0].label == "silence"
    assert result.statuses["structure"].state == "ok"
    assert result.notes is result.chords is result.beats is result.keys is result.meters is None
    assert result.statuses["notes"].state == "not_requested"
    assert "native ABC: No beats" in result.warnings
    with pytest.raises(ValueError, match="notes are unavailable"):
        result.to_midi()


def test_structure_not_requested_empty_failed_are_distinct():
    omitted = from_sheetsage2(native2([], prompts=["timestamp"]), {})
    empty = from_sheetsage2(native2([]), {})
    failed = from_sheetsage2(native2([event(0, structure=None, key="C:major")]), {})
    assert omitted.structure is None and omitted.statuses["structure"].state == "not_requested"
    assert empty.structure == [] and empty.statuses["structure"].state == "ok"
    assert failed.structure is None and failed.statuses["structure"].state == "failed"
    assert failed.keys[0].label == "C:major"
    assert failed.native["events"][0]["values"]["structure"] is None


def test_rhythm_denominator_conversion_and_meter_changes():
    result = from_sheetsage2(native2([
        event(0, rhythm={"meter": [3, 4], "eighth_position": 0}),
        event(1, rhythm={"eighth_position": 2}),
        event(2, rhythm={"eighth_position": 4}),
        event(3, rhythm={"meter": [6, 8], "eighth_position": 0}),
        event(4, rhythm={"eighth_position": 1}),
        event(5, rhythm={"eighth_position": 5}),
    ]), {})
    assert [b.position for b in result.beats] == [1, 2, 3, 1, 2, 6]
    assert [(m.start, m.end, m.numerator, m.denominator) for m in result.meters] == [(0, 3, 3, 4), (3, 10, 6, 8)]


def test_offgrid_rhythm_does_not_erase_structure():
    result = from_sheetsage2(native2([event(0, structure="verse", rhythm={"meter": [4, 4], "eighth_position": 1})]), {})
    assert result.beats is None and result.statuses["beats"].state == "failed"
    assert result.structure[0].label == "verse" and result.meters[0].numerator == 4


@pytest.mark.parametrize("bad", [float("nan"), -1, 11])
def test_malformed_structure_coordinates_are_explicit(bad):
    result = from_sheetsage2(native2([event(bad, structure="verse")]), {})
    assert result.structure is None and result.statuses["structure"].state == "failed"


def test_json_snapshot_and_no_native_serialization(tmp_path):
    native = native2([event(1, structure="verse")], tensor=object(), midi=b"binary")
    result = from_sheetsage2(native, {"native_path": tmp_path, "progress": print})
    path = tmp_path / "result.json"
    payload = json.loads(result.to_json(path))
    assert json.loads(path.read_text()) == payload
    assert "native" not in payload
    assert payload["schema_version"] == 1
    assert payload["semantics"]["time"]["origin"] == "supplied_audio_start"
    assert payload["execution"]["options"]["native_path"] == str(tmp_path)
    assert payload["structure"] == [{"start": 1, "end": 10, "label": "verse"}]
    assert payload["execution"]["provenance"]["components"]["sheetsage2"]["revision"].startswith("b83c3d")


def test_common_midi_preserves_raw_pitch_time_roles_and_markers(tmp_path):
    import pretty_midi

    result = from_sheetsage2(native2([
        event(1.1234, structure="verse", chord="D:min", rhythm={"meter": [4, 4], "eighth_position": 0},
              melody=[{"pitch": 45, "track": 0, "end_time": 2.3456},
                      {"pitch": 88, "track": 1, "end_time": 3}])]), {})
    path = tmp_path / "shared.mid"
    data = result.to_midi(path)
    assert data == path.read_bytes()
    midi = pretty_midi.PrettyMIDI(io.BytesIO(data))
    assert [i.name for i in midi.instruments] == ["vocal", "instrumental"]
    assert [i.notes[0].pitch for i in midi.instruments] == [45, 88]
    assert midi.instruments[0].notes[0].start == pytest.approx(1.1234, abs=1/19200)
    assert midi.instruments[0].notes[0].end == pytest.approx(2.3456, abs=1/19200)
    assert {t.text for t in midi.text_events} == {"structure:verse", "chords:D:min"}
    assert midi.time_signature_changes[0].numerator == 4
    v1 = pretty_midi.PrettyMIDI(io.BytesIO(from_sheetsage1(native1(), {}).to_midi()))
    assert [n.pitch for n in v1.instruments[0].notes] == [60, 21, 108]
    assert v1.instruments[0].notes[0].start == 10


def test_semantics_are_fresh_and_model_specific():
    v1 = result_semantics("sheetsage1")
    v1["notes"]["roles"].append("wrong")
    assert result_semantics("sheetsage1")["notes"]["roles"] == ["lead"]
    assert result_semantics("sheetsage2")["notes"]["roles"] == ["vocal", "instrumental"]
    with pytest.raises(ValueError, match="Unknown model"):
        result_semantics("other")


def test_v1_fractional_beats_interpolate_instead_of_using_bpm():
    from sheetsage.theory import LeadSheet

    lead = LeadSheet([(0, (4, 2, 2))], [(0, (120,))],
                     [(0, (0, (2, 2, 1, 2, 2, 2)))], [], [(1, 5, (0, 5))], 16)
    result = from_sheetsage1((lead, [0, 1, 2, 3, 4], [10, 11, 13, 14, 15]), {})
    assert (result.notes[0].start, result.notes[0].end) == (10.25, 12)
