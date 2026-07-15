"""Small, dependency-free Hyperparams value object used by the slice path."""


class Hyperparams(dict):
    """Attribute-accessible mapping compatible with Jukebox model builders."""

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError as exc:
            raise AttributeError(attr) from exc

    def __setattr__(self, attr, value):
        self[attr] = value
