"""Explicit lifecycle facade for SheetSage inference."""
class SheetSageSession:
    """Own one configurable SheetSage pipeline without global runtime state."""
    def __init__(self, *, use_jukebox=False, **defaults):
        self.use_jukebox = use_jukebox
        self.defaults = dict(defaults)
        self._ready = False
        self._closed = False

    @property
    def status(self):
        if self._closed:
            return "closed"
        return "ready" if self._ready else "new"

    def load(self):
        if self._closed:
            raise RuntimeError("session is closed")
        self._ready = True
        return self

    def infer(self, audio_path_bytes_or_url, **kwargs):
        if self.status != "ready":
            raise RuntimeError("session is not ready; call load() first")
        options = dict(self.defaults)
        options.update(kwargs)
        options.setdefault("use_jukebox", self.use_jukebox)
        from .infer import sheetsage
        return sheetsage(audio_path_bytes_or_url, **options)

    def release(self):
        # Pipeline model instances are owned by its component caches.
        from .pipeline.steps import _init_extractor, _init_model
        _init_extractor.cache_clear()
        _init_model.cache_clear()
        self._ready = False

    def close(self):
        self.release()
        self._closed = True

    def cache_info(self):
        from . import CACHE_DIR
        return {"directory": str(CACHE_DIR), "exists": CACHE_DIR.exists()}

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()

__all__ = ["SheetSageSession"]
