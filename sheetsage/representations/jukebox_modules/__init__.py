"""
Jukebox-Infer: Minimal inference-only implementation of OpenAI Jukebox

This package provides a streamlined version of Jukebox for music generation,
optimized for PyTorch 2.7+ and single-GPU inference.
"""

__version__ = "0.1.0"

from jukebox_modules.api import Jukebox
from jukebox_modules.embeddings import (
    JukeboxEmbeddings,
    get_jukebox_singleton,
    init_jukebox_singleton,
)
from jukebox_modules.make_models import download_checkpoints

__all__ = [
    "Jukebox",
    "download_checkpoints",
    "JukeboxEmbeddings",
    "init_jukebox_singleton",
    "get_jukebox_singleton",
    "__version__",
]
