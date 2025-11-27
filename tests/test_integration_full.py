#!/usr/bin/env pytest
"""Full integration tests for the inference pipeline."""

import pytest
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def test_audio_url():
    """Provide test audio URL."""
    return 'https://foodgroup.bandcamp.com/track/universe'


@pytest.fixture
def test_params_handcrafted():
    """Parameters for handcrafted feature testing."""
    return {
        'use_jukebox': False,
        'segment_start_hint': 69,
        'segment_end_hint': 88,
        'beats_per_minute_hint': 76
    }


@pytest.fixture
def test_params_jukebox():
    """Parameters for Jukebox feature testing."""
    return {
        'use_jukebox': True,
        'segment_start_hint': 69,
        'segment_end_hint': 88,
        'beats_per_minute_hint': 76
    }


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


class TestInferenceHandcrafted:
    """Test inference with Handcrafted features (CPU-only, fast)."""

    @pytest.mark.slow
    def test_basic_inference(self, test_audio_url, test_params_handcrafted):
        """Test basic inference with handcrafted features."""
        from sheetsage.infer import sheetsage

        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            test_audio_url,
            **test_params_handcrafted
        )

        # Verify outputs
        assert lead_sheet is not None
        assert segment_beats is not None
        assert segment_beats_times is not None

    @pytest.mark.slow
    def test_output_types(self, test_audio_url, test_params_handcrafted):
        """Test that outputs have correct types."""
        from sheetsage.infer import sheetsage
        from sheetsage.theory import LeadSheet
        import numpy as np

        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            test_audio_url,
            **test_params_handcrafted
        )

        # Check types
        assert isinstance(lead_sheet, LeadSheet)
        assert isinstance(segment_beats, np.ndarray)
        assert isinstance(segment_beats_times, np.ndarray)

    @pytest.mark.slow
    def test_output_lengths(self, test_audio_url, test_params_handcrafted):
        """Test that output arrays have consistent lengths."""
        from sheetsage.infer import sheetsage

        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            test_audio_url,
            **test_params_handcrafted
        )

        # Beats and beat times should have same length
        assert len(segment_beats) == len(segment_beats_times)
        assert len(segment_beats) > 0

        # Lead sheet should have content
        assert len(lead_sheet) > 0


@pytest.mark.gpu
class TestInferenceJukebox:
    """Test inference with Jukebox features (requires GPU)."""

    @pytest.mark.slow
    def test_jukebox_inference(self, test_audio_url, test_params_jukebox):
        """Test inference with Jukebox embeddings."""
        from sheetsage.infer import sheetsage

        try:
            lead_sheet, segment_beats, segment_beats_times = sheetsage(
                test_audio_url,
                **test_params_jukebox
            )

            assert lead_sheet is not None
            assert len(segment_beats) > 0

        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip("GPU not available")
            raise
        except ImportError as e:
            pytest.skip(f"Jukebox not installed: {e}")


class TestNotebookPattern:
    """Test the exact pattern used in Inference.ipynb."""

    @pytest.mark.slow
    def test_notebook_cell_0_pattern(self, test_audio_url):
        """Test Cell 0 pattern: main inference call."""
        # Exact pattern from notebook
        AUDIO_URL = test_audio_url
        USE_JUKEBOX = False  # Use False for CI/fast testing
        SEGMENT_START_HINT = 69
        SEGMENT_END_HINT = 88
        BPM_HINT = 76

        from sheetsage.infer import sheetsage

        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            AUDIO_URL,
            use_jukebox=USE_JUKEBOX,
            segment_start_hint=SEGMENT_START_HINT,
            segment_end_hint=SEGMENT_END_HINT,
            beats_per_minute_hint=BPM_HINT
        )

        assert lead_sheet is not None
        assert segment_beats is not None
        assert segment_beats_times is not None

    @pytest.mark.slow
    def test_notebook_cell_1_pattern(self, test_audio_url):
        """Test Cell 1 pattern: lead sheet display."""
        from sheetsage.infer import sheetsage

        # Get lead sheet
        lead_sheet, _, _ = sheetsage(
            test_audio_url,
            use_jukebox=False,
            segment_start_hint=69,
            segment_end_hint=88,
            beats_per_minute_hint=76
        )

        # Pattern from Cell 1
        lily_code = lead_sheet.as_lily()

        assert isinstance(lily_code, str)
        assert len(lily_code) > 0

    @pytest.mark.slow
    def test_notebook_cell_2_pattern(self, test_audio_url):
        """Test Cell 2 pattern: MIDI creation."""
        from sheetsage.infer import sheetsage
        from sheetsage.align import create_beat_to_time_fn

        # Get outputs
        lead_sheet, segment_beats, segment_beats_times = sheetsage(
            test_audio_url,
            use_jukebox=False,
            segment_start_hint=69,
            segment_end_hint=88,
            beats_per_minute_hint=76
        )

        # Pattern from Cell 2
        beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
        midi_bytes = lead_sheet.as_midi(beat_to_time_fn)

        assert isinstance(midi_bytes, bytes)
        assert len(midi_bytes) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_url(self):
        """Test handling of invalid audio URL."""
        from sheetsage.infer import sheetsage

        with pytest.raises(Exception):
            sheetsage(
                'https://invalid-url-that-does-not-exist.com/audio.mp3',
                use_jukebox=False,
                segment_start_hint=0,
                segment_end_hint=10,
                beats_per_minute_hint=120
            )

    @pytest.mark.slow
    def test_different_bpm_hints(self, test_audio_url):
        """Test with different BPM hints."""
        from sheetsage.infer import sheetsage

        bpm_hints = [60, 120, 180]

        for bpm in bpm_hints:
            lead_sheet, _, _ = sheetsage(
                test_audio_url,
                use_jukebox=False,
                segment_start_hint=69,
                segment_end_hint=88,
                beats_per_minute_hint=bpm
            )

            assert lead_sheet is not None
