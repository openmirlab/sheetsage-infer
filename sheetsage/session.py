"""Session-local model lifetime for SheetSage inference.

Reads: device, pipeline, assets; read by: sheetsage.__init__, public callers.
"""


class SheetSageSession:
    """Own one configurable SheetSage pipeline without global runtime state."""
    def __init__(self, *, use_jukebox=False, device="auto", **defaults):
        self.use_jukebox = use_jukebox
        self.device = device
        self.defaults = dict(defaults)
        self._components = None
        self._resolved_device = None
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
        if self._ready:
            return self
        from .device import resolve_device
        from .pipeline import InputFeats, load_components

        self._resolved_device = resolve_device(self.device)
        input_feats = InputFeats.JUKEBOX if self.use_jukebox else InputFeats.HANDCRAFTED
        self._components = load_components(
            input_feats,
            True,
            True,
            self._resolved_device,
        )
        self._ready = True
        return self

    def infer(self, audio_path_bytes_or_url, **kwargs):
        if self.status != "ready":
            raise RuntimeError("session is not ready; call load() first")
        options = dict(self.defaults)
        options.update(kwargs)
        options.setdefault("use_jukebox", self.use_jukebox)
        if options["use_jukebox"] != self.use_jukebox:
            raise ValueError("use_jukebox is fixed when a SheetSageSession is created")
        from .infer import sheetsage
        return sheetsage(
            audio_path_bytes_or_url,
            **options,
            device=self._resolved_device,
            _components=self._components,
        )

    def release(self):
        self._components = None
        self._resolved_device = None
        self._ready = False

    def close(self):
        self.release()
        self._closed = True

    def cache_info(self):
        from .assets import resolve_asset_path
        from .pipeline import InputFeats, Task

        input_feats = InputFeats.JUKEBOX if self.use_jukebox else InputFeats.HANDCRAFTED
        tags = [
            f"SHEETSAGE_V02_{input_feats.name}_{task.name}_{suffix}"
            for task in (Task.MELODY, Task.HARMONY)
            for suffix in ("CFG", "MODEL")
        ]
        assets = {tag: resolve_asset_path(tag) for tag in tags}
        return {
            "assets": {tag: {"path": str(path), "exists": path.is_file()} for tag, path in assets.items()}
        }

    def __enter__(self):
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        self.close()

__all__ = ["SheetSageSession"]
