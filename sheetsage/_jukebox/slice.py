"""Reversible backend seam for measuring a future Jukebox dependency slice."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import ModuleType

from .hparams import Hyperparams as SliceHyperparams


@dataclass(frozen=True)
class JukeboxBackend:
    """The four symbols consumed by ``representations.jukebox``."""

    Hyperparams: type
    make_model: object
    conditioners: ModuleType
    empty_cache: object
    mode: str


def _external() -> JukeboxBackend:
    hparams = importlib.import_module("jukebox_infer.hparams")
    models = importlib.import_module("jukebox_infer.make_models")
    conditioners = importlib.import_module("jukebox_infer.prior.conditioners")
    torch_utils = importlib.import_module("jukebox_infer.utils.torch_utils")
    return JukeboxBackend(
        Hyperparams=hparams.Hyperparams,
        make_model=models.make_model,
        conditioners=conditioners,
        empty_cache=torch_utils.empty_cache,
        mode="external",
    )


def get_backend(mode: str | None = None) -> JukeboxBackend:
    """Return the selected backend; ``slice`` currently aliases external code.

    The alias is intentional: it proves the private seam and parity harness
    without silently copying 3,918 lines of model implementation. A future
    support package can replace only this branch while preserving the seam.
    """

    selected = (mode or os.getenv("SHEETSAGE_JUKEBOX_BACKEND", "external")).lower()
    if selected not in {"external", "slice"}:
        raise ValueError("SHEETSAGE_JUKEBOX_BACKEND must be 'external' or 'slice'")
    backend = _external()
    if selected == "slice":
        return JukeboxBackend(**{**backend.__dict__, "Hyperparams": SliceHyperparams, "mode": "slice"})
    return backend
