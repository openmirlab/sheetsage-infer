import numpy as np

from ..assets import retrieve_asset
from .handcrafted import OAFMelSpec as Handcrafted

# Optional Jukebox import
try:
    from .jukebox import JukeboxEmbeddings as _JukeboxEmbeddings

    class JukeboxEmbeddings(_JukeboxEmbeddings):
        def __init__(self):
            super().__init__(model_name="5b", device="cuda", num_layers=53, auto_download=True)
except ImportError:
    # Jukebox not available
    class JukeboxEmbeddings:
        def __init__(self):
            raise ImportError("Jukebox module not installed. Install it separately if needed.")
