"""Package-owned checkpoint configuration."""
from pathlib import Path
CONFIG_PATH = Path(__file__).with_name("checkpoints.toml")
__all__ = ["CONFIG_PATH"]
