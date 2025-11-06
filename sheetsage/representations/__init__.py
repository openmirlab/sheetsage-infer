import numpy as np

from ..assets import retrieve_asset
from .handcrafted import OAFMelSpec as Handcrafted

# Optional Jukebox import
try:
    from .jukebox import Jukebox as _Jukebox
    
    class Jukebox(_Jukebox):
        def __init__(self):
            super().__init__(num_layers=53, fp16=False, log=False)
except ImportError:
    # Jukebox not available
    class Jukebox:
        def __init__(self):
            raise ImportError("Jukebox module not installed. Install it separately if needed.")
