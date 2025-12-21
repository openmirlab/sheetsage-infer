#!/usr/bin/env pytest
"""Test output format conversions (LilyPond, MIDI, etc.)."""

from io import BytesIO

import pytest


@pytest.fixture
def sample_lead_sheet():
    """Create a minimal lead sheet for testing."""
    from sheetsage.theory import Chord, Harmony, Key, LeadSheet, Melody, Meter, Note, Tempo

    # Create minimal valid lead sheet
    meter = Meter(4, 4)
    tempo = Tempo(120)
    key = Key("C", "major")

    # Simple melody: C, D, E, F for one measure
    melody = Melody(
        notes=[
            Note(60, 4),  # C for quarter note
            Note(62, 4),  # D for quarter note
            Note(64, 4),  # E for quarter note
            Note(65, 4),  # F for quarter note
        ]
    )

    # Simple harmony: C major for whole measure
    harmony = Harmony(
        chords=[
            Chord("C", 16),  # C major for whole note
        ]
    )

    lead_sheet = LeadSheet(meter=meter, tempo=tempo, key=key, harmony=harmony, melody=melody)

    return lead_sheet


@pytest.fixture
def beat_to_time_fn():
    """Create a simple beat-to-time function for testing."""
    import numpy as np

    from sheetsage.align import create_beat_to_time_fn

    # Simple: 4 beats at 120 BPM (0.5 seconds per beat)
    segment_beats = np.array([0, 1, 2, 3, 4])
    segment_beats_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    return create_beat_to_time_fn(segment_beats, segment_beats_times)


class TestLilyPondExport:
    """Test LilyPond format export."""

    def test_as_lily_returns_string(self, sample_lead_sheet):
        """Test as_lily returns a string."""
        lily_code = sample_lead_sheet.as_lily()

        assert isinstance(lily_code, str)
        assert len(lily_code) > 0

    def test_as_lily_contains_version(self, sample_lead_sheet):
        """Test LilyPond code contains version declaration."""
        lily_code = sample_lead_sheet.as_lily()

        assert "\\version" in lily_code

    def test_as_lily_has_structure(self, sample_lead_sheet):
        """Test LilyPond code has basic structure."""
        lily_code = sample_lead_sheet.as_lily()

        # Should contain basic LilyPond elements
        # (exact format depends on implementation)
        assert len(lily_code) > 100  # Should be substantial


class TestMIDIExport:
    """Test MIDI format export."""

    def test_as_midi_returns_bytes(self, sample_lead_sheet, beat_to_time_fn):
        """Test as_midi returns bytes."""
        midi_bytes = sample_lead_sheet.as_midi(beat_to_time_fn)

        assert isinstance(midi_bytes, bytes)
        assert len(midi_bytes) > 0

    def test_as_midi_valid_format(self, sample_lead_sheet, beat_to_time_fn):
        """Test MIDI bytes are in valid MIDI format."""
        midi_bytes = sample_lead_sheet.as_midi(beat_to_time_fn)

        # MIDI files start with "MThd"
        assert midi_bytes[:4] == b"MThd"

    def test_as_midi_can_be_parsed(self, sample_lead_sheet, beat_to_time_fn):
        """Test MIDI can be parsed by pretty_midi."""
        try:
            import pretty_midi
        except ImportError:
            pytest.skip("pretty_midi not installed")

        midi_bytes = sample_lead_sheet.as_midi(beat_to_time_fn)

        # Should parse without error
        midi = pretty_midi.PrettyMIDI(BytesIO(midi_bytes))
        assert midi is not None
        assert len(midi.instruments) > 0


@pytest.mark.skipif(True, reason="Requires LilyPond binary installed")
class TestEngraving:
    """Test engraving (LilyPond to PNG/PDF)."""

    def test_engrave_png(self, sample_lead_sheet):
        """Test PNG engraving."""
        from sheetsage.utils import engrave

        lily_code = sample_lead_sheet.as_lily()

        try:
            png_bytes = engrave(lily_code, out_format="png")

            assert isinstance(png_bytes, bytes)
            assert len(png_bytes) > 0

            # PNG files start with specific magic bytes
            assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

        except Exception as e:
            if "lilypond" in str(e).lower():
                pytest.skip("LilyPond not installed")
            raise

    def test_engrave_pdf(self, sample_lead_sheet):
        """Test PDF engraving."""
        from sheetsage.utils import engrave

        lily_code = sample_lead_sheet.as_lily()

        try:
            pdf_bytes = engrave(
                lily_code, out_format="pdf", transparent=False, trim=False, hide_footer=False
            )

            assert isinstance(pdf_bytes, bytes)
            assert len(pdf_bytes) > 0

            # PDF files start with "%PDF"
            assert pdf_bytes[:4] == b"%PDF"

        except Exception as e:
            if "lilypond" in str(e).lower():
                pytest.skip("LilyPond not installed")
            raise


class TestTheoryTabConversion:
    """Test TheoryTab format conversions."""

    def test_from_theorytab_exists(self):
        """Test from_theorytab method exists."""
        from sheetsage.theory import LeadSheet

        assert hasattr(LeadSheet, "from_theorytab")
        assert callable(LeadSheet.from_theorytab)

    @pytest.mark.skip(reason="Requires TheoryTab format example data")
    def test_from_theorytab_conversion(self):
        """Test conversion from TheoryTab format."""

        # This would require actual TheoryTab format data
        # Skip for now
        pass


class TestOutputToFile:
    """Test saving outputs to files."""

    def test_save_lilypond_to_file(self, sample_lead_sheet, tmp_path):
        """Test saving LilyPond code to file."""
        lily_code = sample_lead_sheet.as_lily()

        lily_file = tmp_path / "test.ly"
        lily_file.write_text(lily_code, encoding="utf-8")

        assert lily_file.exists()
        assert lily_file.stat().st_size > 0

        # Read back and verify
        content = lily_file.read_text(encoding="utf-8")
        assert content == lily_code

    def test_save_midi_to_file(self, sample_lead_sheet, beat_to_time_fn, tmp_path):
        """Test saving MIDI to file."""
        midi_bytes = sample_lead_sheet.as_midi(beat_to_time_fn)

        midi_file = tmp_path / "test.mid"
        midi_file.write_bytes(midi_bytes)

        assert midi_file.exists()
        assert midi_file.stat().st_size > 0

        # Read back and verify
        content = midi_file.read_bytes()
        assert content == midi_bytes
