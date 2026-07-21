"""Device selection for SheetSage model construction.

Reads: torch; read by: sheetsage.infer, sheetsage.session, representations.jukebox.
"""

import torch


def resolve_device(device="auto") -> torch.device:
    """Resolve and validate SheetSage's public device vocabulary."""
    if isinstance(device, torch.device):
        requested = device
    elif device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        try:
            requested = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid device {device!r}; use auto, cpu, cuda, or cuda:N") from exc
    else:
        raise ValueError(f"Invalid device {device!r}; use auto, cpu, cuda, or cuda:N")

    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was explicitly requested ({requested}) but is unavailable")
        if requested.index is not None and requested.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {requested.index} is unavailable")
    elif requested.type != "cpu":
        raise ValueError(f"Unsupported device {requested!s}; use cpu, cuda, or cuda:N")
    return requested
