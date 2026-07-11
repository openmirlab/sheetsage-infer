"""Environment fingerprinting for bit-exact regression fixtures.

Bit-exact regression fixtures (recorded numpy arrays / LilyPond text) are
only valid in the exact env they were recorded in -- torch/librosa/numpy
versions and CPU vs GPU all affect floating-point results at the last few
bits, and librosa itself has changed mel-spectrogram output across versions
(see test_handcrafted_regression.py for a concrete, confirmed example).
Per project convention, such fixtures carry a recorded `*_env.json` sidecar;
tests compare the current environment's fingerprint against it and SKIP
(not fail) on mismatch, since a mismatch means "not comparable" rather than
"regressed".

Named with a leading underscore (not `test_*`) so pytest does not try to
collect it as a test module.

Reads: (leaf module); read by: tests/test_*_regression.py
"""

import json
import pathlib
import platform

import librosa
import numpy as np
import torch

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def current_env():
    return {
        "torch": torch.__version__,
        "librosa": librosa.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
    }


def env_matches(recorded_env_path):
    """True if the current environment matches a recorded fixture's env.json."""
    if not pathlib.Path(recorded_env_path).exists():
        return False
    with open(recorded_env_path) as f:
        recorded = json.load(f)
    return recorded == current_env()
