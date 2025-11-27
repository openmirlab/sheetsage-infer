#!/usr/bin/env pytest
"""Test that all modules and classes can be imported correctly."""

import pytest


class TestCoreImports:
    """Test core module imports."""

    def test_import_sheetsage_function(self):
        """Test main sheetsage function import."""
        from sheetsage.infer import sheetsage
        assert callable(sheetsage)

    def test_import_status_enum(self):
        """Test Status enum import."""
        from sheetsage.infer import Status
        assert hasattr(Status, 'FETCHING_AUDIO')
        assert hasattr(Status, 'DETECTING_BEATS')
        assert hasattr(Status, 'EXTRACTING_FEATURES')
        assert hasattr(Status, 'TRANSCRIBING')
        assert hasattr(Status, 'FORMATTING')


class TestRepresentationImports:
    """Test feature extractor imports."""

    def test_import_handcrafted(self):
        """Test Handcrafted extractor import."""
        from sheetsage.representations import Handcrafted
        assert Handcrafted is not None

    def test_import_jukebox_embeddings(self):
        """Test JukeboxEmbeddings import."""
        from sheetsage.representations import JukeboxEmbeddings
        assert JukeboxEmbeddings is not None

    def test_import_base_representation(self):
        """Test base Representation class import."""
        from sheetsage.representations.base import Representation
        assert Representation is not None


class TestTheoryImports:
    """Test music theory module imports."""

    def test_import_lead_sheet(self):
        """Test LeadSheet import."""
        from sheetsage.theory import LeadSheet
        assert LeadSheet is not None

    def test_import_melody(self):
        """Test Melody import."""
        from sheetsage.theory import Melody
        assert Melody is not None

    def test_import_harmony(self):
        """Test Harmony import."""
        from sheetsage.theory import Harmony
        assert Harmony is not None

    def test_import_note(self):
        """Test Note import."""
        from sheetsage.theory import Note
        assert Note is not None

    def test_import_chord(self):
        """Test Chord import."""
        from sheetsage.theory import Chord
        assert Chord is not None

    def test_import_key(self):
        """Test Key import."""
        from sheetsage.theory import Key
        assert Key is not None

    def test_import_meter(self):
        """Test Meter import."""
        from sheetsage.theory import Meter
        assert Meter is not None

    def test_import_tempo(self):
        """Test Tempo import."""
        from sheetsage.theory import Tempo
        assert Tempo is not None


class TestUtilityImports:
    """Test utility function imports."""

    def test_import_engrave(self):
        """Test engrave function import."""
        from sheetsage.utils import engrave
        assert callable(engrave)

    def test_import_retrieve_audio_bytes(self):
        """Test retrieve_audio_bytes import."""
        from sheetsage.utils import retrieve_audio_bytes
        assert callable(retrieve_audio_bytes)

    def test_import_beat_track(self):
        """Test beat_track import."""
        from sheetsage.beat_track import beat_track
        assert callable(beat_track)

    def test_import_create_beat_to_time_fn(self):
        """Test create_beat_to_time_fn import."""
        from sheetsage.align import create_beat_to_time_fn
        assert callable(create_beat_to_time_fn)


class TestModuleImports:
    """Test neural module imports."""

    def test_import_enc_only_transducer(self):
        """Test EncOnlyTransducer import."""
        from sheetsage.modules.modules import EncOnlyTransducer
        assert EncOnlyTransducer is not None

    def test_import_transformer_encoder(self):
        """Test TransformerEncoder import."""
        from sheetsage.modules.modules import TransformerEncoder
        assert TransformerEncoder is not None

    def test_import_identity_encoder(self):
        """Test IdentityEncoder import."""
        from sheetsage.modules.modules import IdentityEncoder
        assert IdentityEncoder is not None


class TestAssetImports:
    """Test asset management imports."""

    def test_import_retrieve_asset(self):
        """Test retrieve_asset import."""
        from sheetsage.assets import retrieve_asset
        assert callable(retrieve_asset)
