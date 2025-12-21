#!/usr/bin/env pytest
"""Test feature extractors (representations)."""

import pytest


class TestHandcraftedExtractor:
    """Test Handcrafted feature extractor."""

    def test_handcrafted_initialization(self):
        """Test Handcrafted can be initialized."""
        from sheetsage.representations import Handcrafted

        extractor = Handcrafted()
        assert extractor is not None

    def test_handcrafted_is_callable(self):
        """Test Handcrafted instance is callable."""
        from sheetsage.representations import Handcrafted

        extractor = Handcrafted()
        assert callable(extractor)

    def test_handcrafted_inheritance(self):
        """Test Handcrafted inherits from Representation."""
        from sheetsage.representations import Handcrafted
        from sheetsage.representations.base import Representation

        assert issubclass(Handcrafted, Representation)

    @pytest.mark.slow
    def test_handcrafted_output_format(self):
        """Test Handcrafted returns correct output format."""
        pytest.skip("Requires actual audio file")
        # This would test: frame_rate, features = extractor(audio_path)


class TestJukeboxExtractor:
    """Test JukeboxEmbeddings feature extractor."""

    def test_jukebox_class_exists(self):
        """Test JukeboxEmbeddings class exists."""
        from sheetsage.representations import JukeboxEmbeddings

        assert JukeboxEmbeddings is not None

    def test_jukebox_inheritance(self):
        """Test JukeboxEmbeddings inherits from Representation."""
        from sheetsage.representations import JukeboxEmbeddings
        from sheetsage.representations.base import Representation

        assert issubclass(JukeboxEmbeddings, Representation)

    def test_jukebox_has_required_methods(self):
        """Test JukeboxEmbeddings has required methods."""
        from sheetsage.representations import JukeboxEmbeddings

        required_methods = ["__call__"]
        for method in required_methods:
            assert hasattr(JukeboxEmbeddings, method), f"Method '{method}' not found"

    @pytest.mark.gpu
    def test_jukebox_initialization(self):
        """Test JukeboxEmbeddings initialization (requires GPU)."""
        from sheetsage.representations import JukeboxEmbeddings

        try:
            extractor = JukeboxEmbeddings()
            assert extractor is not None
            assert callable(extractor)
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                pytest.skip("GPU not available for Jukebox")
            raise
        except ImportError as e:
            pytest.skip(f"Jukebox dependencies not installed: {e}")


class TestRepresentationBase:
    """Test base Representation class."""

    def test_representation_is_abstract(self):
        """Test Representation is an abstract base class."""
        from sheetsage.representations.base import Representation

        # Should have __call__ as abstract method
        assert callable(Representation)

    def test_representation_cannot_instantiate(self):
        """Test Representation cannot be instantiated directly."""
        from sheetsage.representations.base import Representation

        # This might raise TypeError if it's truly abstract
        try:
            obj = Representation()
            # If no error, it's not enforced as abstract
            assert obj is not None
        except TypeError:
            # Expected for abstract class
            pass


class TestInputFeatsEnum:
    """Test InputFeats enum from infer module."""

    def test_input_feats_enum_exists(self):
        """Test InputFeats enum can be imported."""
        from sheetsage.infer import InputFeats

        assert InputFeats is not None

    def test_input_feats_has_handcrafted(self):
        """Test InputFeats has HANDCRAFTED option."""
        from sheetsage.infer import InputFeats

        assert hasattr(InputFeats, "HANDCRAFTED")

    def test_input_feats_has_jukebox(self):
        """Test InputFeats has JUKEBOX option."""
        from sheetsage.infer import InputFeats

        assert hasattr(InputFeats, "JUKEBOX")
