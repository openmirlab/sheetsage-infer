"""Compatibility tests for the optional exact DBN Viterbi accelerator."""

import numpy as np

from sheetsage import beat_track


DECODED = np.array([[0.01, 1], [0.02, 2], [0.03, 1], [0.04, 2]], dtype=float)


class _FakeRNNProcessor:
    def __call__(self, _path):
        return np.zeros((4, 2), dtype=float)


def _run_with_processor(monkeypatch, processor):
    import madmom_infer.features.downbeats as downbeats

    monkeypatch.setattr(downbeats, "RNNDownBeatProcessor", _FakeRNNProcessor)
    monkeypatch.setattr(downbeats, "DBNDownBeatTrackingProcessor", processor)
    return beat_track._madmom_infer_dbn(44100, np.zeros(32), None, None)


def test_dbn_forwards_fast_viterbi_when_explicitly_supported(monkeypatch):
    captured = {}

    class SupportedProcessor:
        def __init__(self, beats_per_bar, fps=None, fast_viterbi=False, **kwargs):
            captured.update(
                beats_per_bar=beats_per_bar,
                fps=fps,
                fast_viterbi=fast_viterbi,
                kwargs=kwargs,
            )

        def __call__(self, _activations):
            return DECODED

    result = _run_with_processor(monkeypatch, SupportedProcessor)

    assert captured["fast_viterbi"] is True
    assert captured["kwargs"] == {}
    assert result == (0, 2, [0.01, 0.02, 0.03, 0.04])


def test_dbn_keeps_legacy_constructor_and_decoded_output(monkeypatch):
    captured = {}

    class LegacyProcessor:
        def __init__(self, beats_per_bar, fps=None, **kwargs):
            captured.update(beats_per_bar=beats_per_bar, fps=fps, kwargs=kwargs)

        def __call__(self, _activations):
            return DECODED

    result = _run_with_processor(monkeypatch, LegacyProcessor)

    assert captured["kwargs"] == {}
    assert result == (0, 2, [0.01, 0.02, 0.03, 0.04])
