"""GPU smoke test for the Jukebox representation path (opt-in, not run by default).

This exercises exactly the path that was broken before the ADOPT campaign's
jukebox_modules -> jukebox_infer swap: `sheetsage.representations.jukebox.
JukeboxEmbeddings` previously crashed at construction time with
`ModuleNotFoundError: No module named 'jukebox_modules.data'` (the vendored
fork never included that submodule). It also covers the numpy-float64 ->
`torch.Tensor.view()` TypeError workaround documented at the top of
sheetsage/representations/jukebox.py.

Opt-in because it loads a ~5GB checkpoint onto the GPU (may collide with
other work on a shared GPU) and downloads on first run if not cached under
~/.cache/jukebox. Requires:
  - a CUDA device
  - SHEETSAGE_RUN_JUKEBOX_TESTS=1 in the environment
  - checkpoints already cached (auto_download=False; won't trigger a
    multi-GB download from a test run)

Run explicitly with:
  SHEETSAGE_RUN_JUKEBOX_TESTS=1 uv run pytest tests/test_jukebox_smoke.py -v
"""

import os

import numpy as np
import pytest

_SKIP_REASON = None
try:
    import torch

    if os.environ.get("SHEETSAGE_RUN_JUKEBOX_TESTS") != "1":
        _SKIP_REASON = "set SHEETSAGE_RUN_JUKEBOX_TESTS=1 to run (loads a ~5GB GPU checkpoint)"
    elif not torch.cuda.is_available():
        _SKIP_REASON = "no CUDA device available"
except ImportError:  # pragma: no cover - torch is a core dependency
    _SKIP_REASON = "torch not importable"

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


def test_jukebox_embeddings_produce_finite_activations():
    from sheetsage.representations.jukebox import JukeboxEmbeddings

    # The 5B prior's conditioner requires >=60s of (reported) total audio
    # length (min_duration hparam) -- a pre-existing Jukebox architecture
    # constraint, not something the campaign's swap changed. Loop a short
    # clip to clear that floor without needing a large fixture file.
    from sheetsage.assets import retrieve_asset

    try:
        wav_path = retrieve_asset("TEST_JUKEBOX_LEGACY")
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"could not download TEST_JUKEBOX_LEGACY asset: {exc}")

    import soundfile as sf

    audio, sr = sf.read(wav_path)
    if audio.shape[0] / sr < 60.0:
        reps = int(60.0 * sr / audio.shape[0]) + 1
        audio = np.tile(audio, (reps,) + (1,) * (audio.ndim - 1))
        looped_path = "/tmp/_sheetsage_jukebox_smoke_looped.wav"
        sf.write(looped_path, audio, sr)
        wav_path = looped_path

    extractor = JukeboxEmbeddings(model_name="5b", device="cuda", auto_download=False)
    rate, embeddings = extractor(wav_path, offset=0.0, duration=20.0)

    assert rate > 0
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == extractor.embedding_dim
    assert np.isfinite(embeddings.astype(np.float32)).all()
