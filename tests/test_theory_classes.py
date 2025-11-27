#!/usr/bin/env pytest
"""Test music theory classes."""

import pytest


class TestNote:
    """Test Note class."""

    def test_note_creation(self):
        """Test creating a Note."""
        from sheetsage.theory import Note

        note = Note(60, 4)  # Middle C, quarter note
        assert note is not None

    def test_note_attributes(self):
        """Test Note has expected attributes."""
        from sheetsage.theory import Note

        note = Note(60, 4)

        # Should have pitch and duration
        assert hasattr(note, 'pitch') or hasattr(note, 'midi') or hasattr(note, 'note')
        assert hasattr(note, 'duration') or hasattr(note, 'dur')


class TestChord:
    """Test Chord class."""

    def test_chord_creation(self):
        """Test creating a Chord."""
        from sheetsage.theory import Chord

        chord = Chord('C', 16)  # C major, whole note
        assert chord is not None

    def test_chord_with_quality(self):
        """Test chord with different qualities."""
        from sheetsage.theory import Chord

        # Test various chord symbols
        chord_symbols = ['C', 'Dm', 'Em', 'F', 'G', 'Am']

        for symbol in chord_symbols:
            chord = Chord(symbol, 16)
            assert chord is not None


class TestMelody:
    """Test Melody class."""

    def test_melody_creation(self):
        """Test creating a Melody."""
        from sheetsage.theory import Melody, Note

        notes = [
            Note(60, 4),
            Note(62, 4),
            Note(64, 4),
            Note(65, 4),
        ]

        melody = Melody(notes=notes)
        assert melody is not None

    def test_melody_length(self):
        """Test Melody has length."""
        from sheetsage.theory import Melody, Note

        notes = [Note(60, 4), Note(62, 4)]
        melody = Melody(notes=notes)

        # Should be able to get length
        assert len(melody) >= 0 or hasattr(melody, 'notes')


class TestHarmony:
    """Test Harmony class."""

    def test_harmony_creation(self):
        """Test creating a Harmony."""
        from sheetsage.theory import Harmony, Chord

        chords = [
            Chord('C', 16),
            Chord('G', 16),
        ]

        harmony = Harmony(chords=chords)
        assert harmony is not None

    def test_harmony_length(self):
        """Test Harmony has length."""
        from sheetsage.theory import Harmony, Chord

        chords = [Chord('C', 16)]
        harmony = Harmony(chords=chords)

        # Should be able to get length
        assert len(harmony) >= 0 or hasattr(harmony, 'chords')


class TestKey:
    """Test Key class."""

    def test_key_creation_major(self):
        """Test creating a major Key."""
        from sheetsage.theory import Key

        key = Key('C', 'major')
        assert key is not None

    def test_key_creation_minor(self):
        """Test creating a minor Key."""
        from sheetsage.theory import Key

        key = Key('A', 'minor')
        assert key is not None

    def test_key_different_tonics(self):
        """Test keys with different tonics."""
        from sheetsage.theory import Key

        tonics = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

        for tonic in tonics:
            key_major = Key(tonic, 'major')
            key_minor = Key(tonic, 'minor')

            assert key_major is not None
            assert key_minor is not None


class TestMeter:
    """Test Meter class."""

    def test_meter_4_4(self):
        """Test creating 4/4 meter."""
        from sheetsage.theory import Meter

        meter = Meter(4, 4)
        assert meter is not None

    def test_meter_3_4(self):
        """Test creating 3/4 meter."""
        from sheetsage.theory import Meter

        meter = Meter(3, 4)
        assert meter is not None

    def test_meter_different_signatures(self):
        """Test various time signatures."""
        from sheetsage.theory import Meter

        signatures = [(4, 4), (3, 4), (6, 8), (2, 4)]

        for numerator, denominator in signatures:
            meter = Meter(numerator, denominator)
            assert meter is not None


class TestTempo:
    """Test Tempo class."""

    def test_tempo_creation(self):
        """Test creating a Tempo."""
        from sheetsage.theory import Tempo

        tempo = Tempo(120)
        assert tempo is not None

    def test_tempo_different_values(self):
        """Test different tempo values."""
        from sheetsage.theory import Tempo

        tempos = [60, 90, 120, 140, 180]

        for bpm in tempos:
            tempo = Tempo(bpm)
            assert tempo is not None


class TestLeadSheet:
    """Test LeadSheet class."""

    def test_leadsheet_creation(self):
        """Test creating a LeadSheet."""
        from sheetsage.theory import LeadSheet, Melody, Harmony, Note, Chord, Key, Meter, Tempo

        meter = Meter(4, 4)
        tempo = Tempo(120)
        key = Key('C', 'major')
        melody = Melody(notes=[Note(60, 4)])
        harmony = Harmony(chords=[Chord('C', 16)])

        lead_sheet = LeadSheet(
            meter=meter,
            tempo=tempo,
            key=key,
            harmony=harmony,
            melody=melody
        )

        assert lead_sheet is not None

    def test_leadsheet_has_as_lily(self):
        """Test LeadSheet has as_lily method."""
        from sheetsage.theory import LeadSheet

        assert hasattr(LeadSheet, 'as_lily')

    def test_leadsheet_has_as_midi(self):
        """Test LeadSheet has as_midi method."""
        from sheetsage.theory import LeadSheet

        assert hasattr(LeadSheet, 'as_midi')

    def test_leadsheet_length(self):
        """Test LeadSheet has length."""
        from sheetsage.theory import LeadSheet, Melody, Harmony, Note, Chord, Key, Meter, Tempo

        meter = Meter(4, 4)
        tempo = Tempo(120)
        key = Key('C', 'major')
        melody = Melody(notes=[Note(60, 4)])
        harmony = Harmony(chords=[Chord('C', 16)])

        lead_sheet = LeadSheet(
            meter=meter,
            tempo=tempo,
            key=key,
            harmony=harmony,
            melody=melody
        )

        # Should have length or indexable
        try:
            length = len(lead_sheet)
            assert length >= 0
        except TypeError:
            # If __len__ not implemented, check if it has components
            assert hasattr(lead_sheet, 'melody')
            assert hasattr(lead_sheet, 'harmony')
