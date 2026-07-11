"""End-to-end (CPU) lead-sheet regression test -- the ADOPT campaign's P0 baseline.

Runs the full `sheetsage()` pipeline (beat tracking -> handcrafted features
-> transcription -> LilyPond) on the small bundled TEST_WAV asset and checks
the LilyPond output is byte-identical to the fixture recorded before/after
the jukebox_modules -> jukebox_infer swap (see CHANGELOG). This is the
concrete instance of the campaign's accuracy rule: "lead-sheet outputs
verified unchanged (discrete: notes/chords -- exact equality on same env)".

Only skips on an environment mismatch (torch/librosa/numpy/python); unlike
the handcrafted-feature test this compares discrete LilyPond text (notes/
chords), so it is less exposed to floating-point drift, but beat detection
and the underlying model math still depend on the recorded env.
"""

import pytest

from tests._env_fixtures import FIXTURES_DIR, env_matches


def test_lead_sheet_matches_recorded_fixture():
    env_path = FIXTURES_DIR / "lead_sheet_test_wav_env.json"
    if not env_matches(env_path):
        pytest.skip("torch/librosa/numpy/python environment differs from the recorded fixture")

    from sheetsage.assets import retrieve_asset
    from sheetsage.infer import sheetsage

    try:
        wav_path = retrieve_asset("TEST_WAV")
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"could not download TEST_WAV asset: {exc}")

    lead_sheet, _segment_beats, _segment_beats_times = sheetsage(wav_path, use_jukebox=False)
    lily = lead_sheet.as_lily()

    ref = (FIXTURES_DIR / "lead_sheet_test_wav_ref.ly").read_text()
    assert lily == ref
