#!/usr/bin/env pytest
"""Test function signatures and API contracts."""

import pytest
import inspect


class TestSheetsageSignature:
    """Test main sheetsage function signature."""

    def test_sheetsage_required_params(self):
        """Test sheetsage has all required parameters."""
        from sheetsage.infer import sheetsage

        sig = inspect.signature(sheetsage)
        params = list(sig.parameters.keys())

        # Required parameters from notebook usage
        required = [
            'audio',
            'use_jukebox',
            'segment_start_hint',
            'segment_end_hint',
            'beats_per_minute_hint'
        ]

        for param in required:
            assert param in params, f"Parameter '{param}' not found in sheetsage signature"

    def test_sheetsage_optional_params(self):
        """Test sheetsage has expected optional parameters."""
        from sheetsage.infer import sheetsage

        sig = inspect.signature(sheetsage)
        params = list(sig.parameters.keys())

        # Known optional parameters
        optional = [
            'measures_per_chunk',
            'avoid_chunking_if_possible',
            'legacy_behavior'
        ]

        for param in optional:
            assert param in params, f"Optional parameter '{param}' not found"

    def test_sheetsage_return_annotation(self):
        """Test sheetsage return type annotation if present."""
        from sheetsage.infer import sheetsage

        sig = inspect.signature(sheetsage)
        # Just verify it doesn't raise an error
        assert sig is not None


class TestRepresentationSignatures:
    """Test feature extractor signatures."""

    def test_handcrafted_callable(self):
        """Test Handcrafted extractor is callable."""
        from sheetsage.representations import Handcrafted

        extractor = Handcrafted()
        assert callable(extractor)

        # Check __call__ signature
        sig = inspect.signature(extractor.__call__)
        params = list(sig.parameters.keys())

        expected = ['audio_path', 'offset', 'duration']
        for param in expected:
            assert param in params, f"Parameter '{param}' not found in Handcrafted.__call__"

    def test_jukebox_callable(self):
        """Test JukeboxEmbeddings extractor has correct interface."""
        from sheetsage.representations import JukeboxEmbeddings

        # Check class has __call__ method
        assert hasattr(JukeboxEmbeddings, '__call__')

        # Check signature
        sig = inspect.signature(JukeboxEmbeddings.__call__)
        params = list(sig.parameters.keys())

        expected = ['audio_path', 'offset', 'duration']
        for param in expected:
            assert param in params, f"Parameter '{param}' not found in JukeboxEmbeddings.__call__"


class TestTheorySignatures:
    """Test music theory class methods."""

    def test_leadsheet_as_lily(self):
        """Test LeadSheet.as_lily method exists."""
        from sheetsage.theory import LeadSheet

        assert hasattr(LeadSheet, 'as_lily')
        assert callable(LeadSheet.as_lily)

    def test_leadsheet_as_midi(self):
        """Test LeadSheet.as_midi method exists."""
        from sheetsage.theory import LeadSheet

        assert hasattr(LeadSheet, 'as_midi')
        assert callable(LeadSheet.as_midi)

        # Check signature
        sig = inspect.signature(LeadSheet.as_midi)
        params = list(sig.parameters.keys())

        assert 'beat_to_time_fn' in params, "as_midi should accept beat_to_time_fn"

    def test_leadsheet_from_theorytab(self):
        """Test LeadSheet.from_theorytab method exists."""
        from sheetsage.theory import LeadSheet

        assert hasattr(LeadSheet, 'from_theorytab')
        assert callable(LeadSheet.from_theorytab)


class TestUtilitySignatures:
    """Test utility function signatures."""

    def test_engrave_signature(self):
        """Test engrave function signature."""
        from sheetsage.utils import engrave

        sig = inspect.signature(engrave)
        params = list(sig.parameters.keys())

        expected = ['lily_code', 'out_format', 'transparent', 'trim', 'hide_footer']
        for param in expected:
            assert param in params, f"Parameter '{param}' not found in engrave"

    def test_create_beat_to_time_fn_signature(self):
        """Test create_beat_to_time_fn signature."""
        from sheetsage.align import create_beat_to_time_fn

        sig = inspect.signature(create_beat_to_time_fn)
        params = list(sig.parameters.keys())

        expected = ['segment_beats', 'segment_beats_times']
        for param in expected:
            assert param in params, f"Parameter '{param}' not found"

    def test_beat_track_signature(self):
        """Test beat_track function signature."""
        from sheetsage.beat_track import beat_track

        sig = inspect.signature(beat_track)
        params = list(sig.parameters.keys())

        # Should accept audio_path and optional parameters
        assert 'audio_path' in params or 'audio' in params
