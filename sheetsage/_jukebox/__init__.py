"""Private Jukebox compatibility seam used by the slice experiment.

This namespace deliberately owns only the small contract SheetSage consumes.
It is not a vendored copy of Jukebox: ``external`` remains the default and
``slice`` is an experimental wiring path for parity measurements.
"""

from .slice import JukeboxBackend, get_backend

__all__ = ["JukeboxBackend", "get_backend"]
