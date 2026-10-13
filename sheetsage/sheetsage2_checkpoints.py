"""Verified runtime artifacts for the SheetSage2 backend.

The packaged manifest owns file identities for both the adapter and its MERT2
encoder. Inspection and loading share one resolver; upstream code stays in the
user cache, outside this distribution.
Reads: config/sheetsage2.json, sheetsage.CACHE_DIR; read by: sheetsage.sheetsage2.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from . import CACHE_DIR

_MANIFEST = Path(__file__).with_name("config") / "sheetsage2.json"


def _read_manifest(path):
    try:
        raw = Path(path).read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        if manifest["schema_version"] != 1:
            raise ValueError("unsupported schema_version")
        components = manifest["components"]
        if set(components) != {"sheetsage2", "mert2"}:
            raise ValueError("expected sheetsage2 and mert2 components")
        for component in components.values():
            if not re.fullmatch(r"[0-9a-f]{40}", component["revision"]):
                raise ValueError("revision must be a full commit SHA")
            for field in ("repository", "license", "provenance"):
                if not isinstance(component[field], str) or not component[field]:
                    raise ValueError(f"missing {field}")
            artifacts = component["artifacts"]
            if not {"config.json", "model.safetensors"} <= artifacts.keys():
                raise ValueError("config.json and model.safetensors are required")
            for name, artifact in artifacts.items():
                if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) or ".." in name:
                    raise ValueError(f"invalid artifact filename: {name!r}")
                if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
                    raise ValueError(f"invalid SHA-256: {name}")
                size = artifact["size"]
                if type(size) is not int or size <= 0:
                    raise ValueError(f"invalid size: {name}")
                parsed = urlparse(artifact["url"])
                if parsed.scheme not in {"https", "http", "file"}:
                    raise ValueError(f"unsupported artifact URL: {name}")
                if parsed.scheme != "file" and not parsed.netloc:
                    raise ValueError(f"missing artifact URL host: {name}")
        return manifest, hashlib.sha256(raw).hexdigest()
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid SheetSage2 checkpoint manifest {path}: {exc}") from exc


def _verify(path, artifact):
    if path.stat().st_size != artifact["size"]:
        raise ValueError(f"SheetSage2 artifact size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact["sha256"]:
        raise ValueError(f"SheetSage2 artifact SHA-256 mismatch: {path}")


def _download(path, artifact):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(artifact["url"], timeout=60) as response:
                for chunk in iter(lambda: response.read(8 << 20), b""):
                    stream.write(chunk)
        _verify(temporary, artifact)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CheckpointStore:
    """Resolve, inspect and verify both local checkpoint bundles without model imports."""

    def __init__(self, *, cache_dir=None, model_path=None, base_model_path=None, checkpoint_config=None):
        self.manifest, digest = _read_manifest(
            checkpoint_config if checkpoint_config is not None else _MANIFEST,
        )
        self._provenance = {
            "manifest_sha256": digest,
            "components": {
                key: {"repository": component["repository"], "revision": component["revision"]}
                for key, component in self.manifest["components"].items()
            },
        }
        default_cache = os.environ.get("SHEETSAGE2_CACHE_DIR", str(CACHE_DIR / "sheetsage2"))
        self.cache_dir = Path(cache_dir if cache_dir is not None else default_cache).expanduser().resolve()
        self.overrides = {"sheetsage2": model_path, "mert2": base_model_path}

    def provenance(self):
        """Return an independent copy of the validated manifest's original identity."""
        return deepcopy(self._provenance)

    def directory(self, key):
        override = self.overrides[key]
        if override is not None:
            return Path(override).expanduser().resolve()
        revision = self.manifest["components"][key]["revision"]
        return self.cache_dir / f"{key}_{revision}"

    def cache_info(self):
        """Report file presence/size only; load() verifies hashes before execution."""
        components = {}
        for key, component in self.manifest["components"].items():
            directory = self.directory(key)
            artifacts = {}
            for name, artifact in component["artifacts"].items():
                path = directory / name
                exists = path.is_file()
                artifacts[name] = {
                    "path": str(path), "exists": exists, "size": artifact["size"],
                    "size_matches": exists and path.stat().st_size == artifact["size"],
                    "sha256": artifact["sha256"], "url": artifact["url"],
                }
            components[key] = {
                "path": str(directory), "revision": component["revision"],
                "complete": all(item["size_matches"] for item in artifacts.values()),
                "integrity_verified": False, "artifacts": artifacts,
            }
        return {"cache_dir": str(self.cache_dir), "components": components}

    def prepare(self, *, local_files_only=False):
        """Verify every file before upstream code runs; explicit directories are read-only."""
        directories = {}
        for key, component in self.manifest["components"].items():
            directory = self.directory(key)
            for name, artifact in component["artifacts"].items():
                path = directory / name
                if path.is_file():
                    _verify(path, artifact)
                elif local_files_only or self.overrides[key] is not None:
                    raise FileNotFoundError(f"Missing SheetSage2 artifact: {path}")
                else:
                    _download(path, artifact)
            directories[key] = directory
        return directories
