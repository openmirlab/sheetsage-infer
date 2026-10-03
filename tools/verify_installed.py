"""Verify an installed wheel outside its checkout against the pre-delivery payload.

Reads the installed package, checkout, and immutable delivery hash manifest. Never
loads models or downloads assets. Fails on extra/missing/changed runtime files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    import sheetsage
    from sheetsage.__about__ import __version__
    from sheetsage.assets import get_asset_tags

    installed = Path(sheetsage.__file__).resolve().parent
    assert not installed.is_relative_to(checkout), installed
    assert sheetsage.__version__ == __version__ == importlib.metadata.version("sheetsage-infer")
    baseline = json.loads((checkout / "tests/fixtures/delivery_baseline.json").read_text())
    files = {"sheetsage/" + str(p.relative_to(installed)): p
             for p in installed.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".toml")}
    assert set(files) == set(baseline["runtime_sha256"]) | {"sheetsage/__about__.py"}
    for name, path in files.items():
        raw = path.read_bytes()
        assert raw == (checkout / name).read_bytes(), name
        if name == "sheetsage/__about__.py":
            continue
        if name == "sheetsage/__init__.py":
            assert raw.count(b"from .__about__ import __version__\n") == 1
            assert raw.count(b', "__version__"') == 1
            raw = raw.replace(b"from .__about__ import __version__\n", b"")
            raw = raw.replace(b', "__version__"', b"")
        assert hashlib.sha256(raw).hexdigest() == baseline["runtime_sha256"][name], name
    tags = get_asset_tags()
    assert len(tags) == 24
    assert not any(tag.startswith(("HOOKTHEORY", "RWC_")) for tag in tags)
    session = sheetsage.SheetSageSession(device="cpu")
    assert session.status == "new"
    session.close()
    assert session.status == "closed"
    try:
        session.load()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Closed installed session loaded")
    print(json.dumps({"installed": str(installed), "version": __version__,
                      "source_identical_files": len(files), "unchanged_runtime_files": 27,
                      "version_only_init": True, "asset_tags": len(tags)}))


if __name__ == "__main__":
    main()
