"""Bit-exact regression test for the CPU (handcrafted) feature path.

This is the P0 baseline from the ADOPT campaign's jukebox-dedup work,
turned into a fixture: `Handcrafted()(TEST_MP3)` recorded byte-for-byte
in the environment noted in `fixtures/handcrafted_test_mp3_env.json`.
Skips (not fails) on a different torch/librosa/numpy/python environment,
since those are known to shift float32 mel-spectrogram output at the last
few bits (confirmed here: the *original* upstream SheetSage reference
fixture, `TEST_MP3_OAFMELSPEC_REF` in assets/test.json, no longer matches
under librosa>=0.11 -- max abs diff ~12.8 -- which is exactly why this
project records its own fixture + env fingerprint instead of trusting a
fixture recorded in an unknown environment).

Network note: downloads TEST_MP3 via sheetsage.assets on first run (small,
checksum-verified); skips if that download fails (e.g. offline CI).
"""

import numpy as np
import pytest

from tests._env_fixtures import FIXTURES_DIR, env_matches


def test_handcrafted_matches_recorded_fixture():
    env_path = FIXTURES_DIR / "handcrafted_test_mp3_env.json"
    if not env_matches(env_path):
        pytest.skip("torch/librosa/numpy/python environment differs from the recorded fixture")

    from sheetsage.assets import retrieve_asset
    from sheetsage.representations import Handcrafted

    try:
        mp3_path = retrieve_asset("TEST_MP3")
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"could not download TEST_MP3 asset: {exc}")

    rate, feats = Handcrafted()(mp3_path)

    ref = np.load(FIXTURES_DIR / "handcrafted_test_mp3_ref.npy")
    assert rate == pytest.approx(31.25)
    np.testing.assert_array_equal(feats, ref)
