"""Capture or replay the original cached CPU SheetSage pipeline without downloads.

Reads: the local package, cached TEST_WAV/checkpoints, and baseline JSON/NPZ.
Writes only the explicit output directory; never writes audio or weights to the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]

ORIGINAL = "b37c4579a788123e408b80d011f876631daa101f"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def capture(*, cpu_assets_only=False):
    from sheetsage import LIB_DIR, assets
    from sheetsage.infer import sheetsage
    from sheetsage.representations import Handcrafted

    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    # Model acquisition must never turn a local baseline into a network operation.
    def no_download(*args, **kwargs):
        raise RuntimeError("Baseline requires independently cached assets; downloads disabled")
    assets._download = no_download
    source = {"sheetsage/" + str(p.relative_to(LIB_DIR)): digest(p.read_bytes())
              for p in sorted(LIB_DIR.rglob("*"))
              if p.is_file() and p.suffix in (".py", ".json", ".toml")}
    checkpoints = {}
    for tag, item in assets._ASSETS.items():
        if not tag.startswith("SHEETSAGE_"):
            continue
        path = item["path_abs"]
        if cpu_assets_only and "_JUKEBOX_" in tag and not path.is_file():
            continue
        raw = path.read_bytes()
        algorithm = item.get("integrity_algorithm", "sha256")
        assert hashlib.new(algorithm, raw).hexdigest() == item["checksum"], tag
        checkpoints[tag] = {"path": str(path), "size": len(raw), "sha256": digest(raw),
                            "manifest_algorithm": algorithm, "manifest_digest": item["checksum"]}
    from madmom_infer import models
    decoder_assets = {}
    for model in models._DOWNBEATS_BLSTM_FILES:
        path = models._cache_root() / model.relpath
        raw = path.read_bytes()
        assert digest(raw) == model.sha256, model.relpath
        decoder_assets[model.relpath] = {"path": str(path), "size": len(raw),
                                         "sha256": digest(raw)}
    path = Path(assets.retrieve_asset("TEST_WAV", log=False))
    real, rate = sf.read(path, dtype="float64", always_2d=True)
    silence = np.zeros_like(real)
    silence_file = io.BytesIO()
    sf.write(silence_file, silence, rate, format="WAV", subtype="FLOAT")
    inputs = [("real", path, real), ("silence", silence_file.getvalue(), silence)]
    metadata = {
        "original_source_commit": ORIGINAL, "source_sha256": source,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "packages": {d.metadata["Name"]: d.version
                                     for d in importlib.metadata.distributions()},
                        "torch_build": torch.__version__, "torch_config": torch.__config__.show(),
                        "device": "cpu", "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(), "seed": 0},
        "checkpoint_sha256": checkpoints, "madmom_assets_sha256": decoder_assets,
        "input_file": {"path": str(path), "sha256": digest(path.read_bytes())},
        "options": {"device": "cpu", "return_intermediaries": True}, "cases": {},
    }
    arrays = {}
    for name, audio, decoded in inputs:
        case = {"shape": list(decoded.shape), "sample_rate": rate,
                "decoded_sha256": digest(decoded.tobytes()),
                "rms": float(np.sqrt(np.mean(decoded ** 2))), "runs": []}
        if name == "silence":
            assert not np.any(decoded)
            case["encoded_sha256"] = digest(audio)
        started = time.perf_counter()
        feature_rate, features = Handcrafted()(audio)
        case["feature_seconds"] = time.perf_counter() - started
        arrays[name + "/handcrafted"] = features
        case["feature_rate"] = feature_rate
        for repeat in range(2):
            started = time.perf_counter()
            statuses = []
            try:
                result = sheetsage(audio, device="cpu", return_intermediaries=True,
                                   status_change_callback=lambda status, statuses=statuses: statuses.append(status.name))
            except Exception as exc:
                run = {"error_type": type(exc).__name__, "error": str(exc)}
            else:
                sheet, beats, times, chunks, melody, harmony = result
                run = {"lead_sheet": plain(sheet), "lilypond": sheet.as_lily(),
                       "chunks": plain(chunks), "melody_chunks": len(melody),
                       "harmony_chunks": len(harmony)}
                prefix = f"{name}/{repeat}"
                arrays[prefix + "/segment_beats"] = np.asarray(beats)
                arrays[prefix + "/segment_beats_times"] = np.asarray(times)
                for task, values in (("melody_logits", melody), ("harmony_logits", harmony)):
                    for index, value in enumerate(values):
                        arrays[f"{prefix}/{task}/{index}"] = value
            run["seconds"] = time.perf_counter() - started
            run["statuses"] = statuses
            case["runs"].append(run)
        metadata["cases"][name] = case
    metadata["arrays"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype),
              "sha256": digest(value.tobytes()), "finite": bool(np.isfinite(value).all()),
              "nan_count": int(np.isnan(value).sum()),
              "rms": float(np.sqrt(np.mean(value.astype(float) ** 2)))}
        for key, value in arrays.items()
    }
    return metadata, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cpu-assets-only", action="store_true",
                        help="Permit absent unused Jukebox transducer assets; verify any present")
    parser.add_argument("--portable", action="store_true",
                        help="Require exact discrete/segment outputs; report float differences without an equality claim")
    parser.add_argument("--installed", action="store_true",
                        help="Use the installed package instead of prepending the source checkout")
    args = parser.parse_args()
    if args.record and (args.cpu_assets_only or args.portable or args.installed):
        parser.error("--record requires the complete original source/profile")
    if not args.installed:
        sys.path.insert(0, str(ROOT))
    if args.record:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        changes = subprocess.check_output(["git", "diff", "HEAD", "--", "sheetsage"], cwd=ROOT)
        if head != ORIGINAL or changes:
            parser.error("--record requires the original known Git commit and unchanged runtime")
    elif args.reference is None:
        parser.error("replay requires --reference; use --record only before production edits")
    if args.output.exists():
        parser.error("output directory must not already exist")
    if args.installed:
        from sheetsage import LIB_DIR
        assert not LIB_DIR.resolve().is_relative_to(ROOT), LIB_DIR
    metadata, arrays = capture(cpu_assets_only=args.cpu_assets_only)
    if args.reference:
        reference = json.loads((args.reference / "metadata.json").read_text())
        from sheetsage import assets
        configured = {tag: item for tag, item in assets._ASSETS.items()
                      if tag.startswith("SHEETSAGE_")}
        assert set(configured) == set(reference["checkpoint_sha256"])
        for tag, item in configured.items():
            recorded = reference["checkpoint_sha256"][tag]
            assert item["checksum"] == recorded["manifest_digest"], tag
            assert item.get("integrity_algorithm", "sha256") == recorded["manifest_algorithm"], tag
            if tag in metadata["checkpoint_sha256"]:
                assert metadata["checkpoint_sha256"][tag]["sha256"] == recorded["sha256"], tag
        assert metadata["input_file"]["sha256"] == reference["input_file"]["sha256"]
        for name, case in metadata["cases"].items():
            assert case["decoded_sha256"] == reference["cases"][name]["decoded_sha256"], name
        metadata["optional_uncached_jukebox_assets"] = sorted(
            set(configured) - set(metadata["checkpoint_sha256"]))
        diagnostics = {}
        with np.load(args.reference / "arrays.npz", allow_pickle=False) as expected:
            assert set(expected.files) == set(arrays)
            for key, value in arrays.items():
                if args.portable:
                    assert value.shape == expected[key].shape, key
                    assert value.dtype == expected[key].dtype, key
                    assert np.isfinite(value).all(), key
                    assert np.sqrt(np.mean(value.astype(float) ** 2)) > 0, key
                    if "segment_beats" in key:
                        np.testing.assert_array_equal(value, expected[key], err_msg=key)
                    delta = value.astype(float) - expected[key].astype(float)
                    diagnostics[key] = {
                        "exact": bool(np.array_equal(value, expected[key])),
                        "max_absolute_difference": float(np.max(np.abs(delta))),
                        "relative_rms_difference": float(np.sqrt(np.mean(delta ** 2)) /
                            max(np.sqrt(np.mean(expected[key].astype(float) ** 2)), 1e-30)),
                    }
                    if key.startswith("real/1/"):
                        np.testing.assert_array_equal(value, arrays[key.replace("real/1/", "real/0/")])
                else:
                    np.testing.assert_array_equal(value, expected[key], err_msg=key)
        for name, case in metadata["cases"].items():
            for actual, expected in zip(case["runs"], reference["cases"][name]["runs"], strict=True):
                assert {k: v for k, v in actual.items() if k != "seconds"} == {
                    k: v for k, v in expected.items() if k != "seconds"}, name
        metadata["exact_replay"] = not args.portable
        if args.portable:
            metadata["portable_discrete_replay"] = True
            metadata["float_diagnostics"] = diagnostics
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: [{k: v for k, v in run.items() if k in
                            ("seconds", "error_type", "error", "statuses")} for run in case["runs"]]
                      for name, case in metadata["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
