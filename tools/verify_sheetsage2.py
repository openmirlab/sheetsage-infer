#!/usr/bin/env python3
"""Compare the packaged SheetSage2 session with its pinned upstream model.

Run from a checkout with the SheetSage2 optional dependencies installed. This
loads both copies independently from verified local files, compares complete
in-memory inference returns, and checks a compact JSON fingerprint reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VOLATILE_FIELDS = {"elapsed_seconds", "peak_gpu_mib"}
REAL_OPTIONS = {
    "export_logits": True,
    "export_scores": True,
    "export_embeddings": True,
}
BASE_CHECKOUT_SHA = "97ddf16f0de85951db8e03ab04013fbe7d81372a"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_summary(value: torch.Tensor) -> dict:
    tensor = value.detach().contiguous().cpu()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "__tensor__": True,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": sha256_bytes(raw),
    }


def canonical(value):
    """Reduce tensors and binary payloads to exact, JSON-friendly identities."""
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": sha256_bytes(array.tobytes()),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"__bytes__": len(raw), "sha256": sha256_bytes(raw)}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            name = str(key)
            if name in VOLATILE_FIELDS:
                continue
            result[name] = canonical(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, slice):
        return {"__slice__": [canonical(value.start), canonical(value.stop), canonical(value.step)]}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return {"__float__": repr(value)}
        return value
    raise TypeError(f"Unsupported inference output type: {type(value).__module__}.{type(value).__qualname__}")


def json_digest(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def first_differences(left, right, path="$", limit=24):
    differences = []

    def visit(a, b, current):
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append({"path": current, "left_type": type(a).__name__, "right_type": type(b).__name__})
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append({"path": f"{current}.{key}", "missing": "left" if key not in a else "right"})
                    if len(differences) >= limit:
                        break
                else:
                    visit(a[key], b[key], f"{current}.{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append({"path": current, "left_length": len(a), "right_length": len(b)})
            for index, (left_item, right_item) in enumerate(zip(a, b, strict=False)):
                visit(left_item, right_item, f"{current}[{index}]")
                if len(differences) >= limit:
                    break
        elif a != b:
            differences.append({"path": current, "left": a, "right": b})

    visit(left, right, path)
    return differences


def in_memory_differences(left, right, path="$", limit=24):
    """Compare returned values directly, using torch.equal for tensor leaves."""
    differences = []

    def visit(a, b, current):
        if len(differences) >= limit:
            return
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                differences.append({"path": current, "reason": "tensor type mismatch"})
            elif a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape) or not torch.equal(a, b):
                differences.append({
                    "path": current,
                    "reason": "tensor value, dtype, or shape mismatch",
                    "left_dtype": str(a.dtype), "right_dtype": str(b.dtype),
                    "left_shape": list(a.shape), "right_shape": list(b.shape),
                })
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                differences.append({"path": current, "reason": "array type mismatch"})
            elif a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                differences.append({
                    "path": current,
                    "reason": "array value, dtype, or shape mismatch",
                    "left_dtype": str(a.dtype), "right_dtype": str(b.dtype),
                    "left_shape": list(a.shape), "right_shape": list(b.shape),
                })
        elif isinstance(a, Mapping) or isinstance(b, Mapping):
            if not (isinstance(a, Mapping) and isinstance(b, Mapping)):
                differences.append({"path": current, "reason": "mapping type mismatch"})
                return
            left_items = {str(key): value for key, value in a.items() if str(key) not in VOLATILE_FIELDS}
            right_items = {str(key): value for key, value in b.items() if str(key) not in VOLATILE_FIELDS}
            for key in sorted(set(left_items) | set(right_items)):
                if key not in left_items or key not in right_items:
                    differences.append({"path": f"{current}.{key}", "missing": "left" if key not in left_items else "right"})
                    if len(differences) >= limit:
                        break
                else:
                    visit(left_items[key], right_items[key], f"{current}.{key}")
        elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
                differences.append({"path": current, "reason": "sequence type mismatch"})
                return
            if len(a) != len(b):
                differences.append({"path": current, "left_length": len(a), "right_length": len(b)})
            for index, (left_item, right_item) in enumerate(zip(a, b, strict=False)):
                visit(left_item, right_item, f"{current}[{index}]")
                if len(differences) >= limit:
                    break
        elif isinstance(a, (bytes, bytearray, memoryview)) or isinstance(b, (bytes, bytearray, memoryview)):
            if not isinstance(a, (bytes, bytearray, memoryview)) or not isinstance(b, (bytes, bytearray, memoryview)) or bytes(a) != bytes(b):
                differences.append({"path": current, "reason": "binary value mismatch"})
        elif type(a) is not type(b) or a != b:
            differences.append({"path": current, "left": repr(a), "right": repr(b)})

    visit(left, right, path)
    return differences


def artifact_fingerprint(session, audio_path: Path, device: str) -> dict:
    import huggingface_hub
    import scipy
    import torchaudio
    import transformers

    cache = session.cache_info()
    manifest_path = REPO_ROOT / "sheetsage" / "config" / "sheetsage2.json"
    local_files = {
        "sheetsage2.py": REPO_ROOT / "sheetsage" / "sheetsage2.py",
        "sheetsage2_checkpoints.py": REPO_ROOT / "sheetsage" / "sheetsage2_checkpoints.py",
        "manifest": manifest_path,
    }
    components = {}
    for name, component in cache["components"].items():
        components[name] = {
            "revision": component["revision"],
            "artifacts": {
                filename: {"size": item["size"], "sha256": item["sha256"]}
                for filename, item in sorted(component["artifacts"].items())
            },
        }
    return {
        "baseline_base_sha": BASE_CHECKOUT_SHA,
        "backend_files": {name: sha256_file(path) for name, path in local_files.items()},
        "components": components,
        "input": {
            "real_fixture": {
                "sha256": sha256_file(audio_path),
                "size": audio_path.stat().st_size,
            },
            "zeros": {
                "sha256": sha256_bytes(np.zeros(7 * 24_000, dtype=np.float32).tobytes()),
                "size": 7 * 24_000 * np.dtype(np.float32).itemsize,
                "duration_seconds": 7,
                "sampling_rate": 24_000,
                "dtype": "float32",
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchaudio": str(torchaudio.__version__),
            "transformers": str(transformers.__version__),
            "huggingface_hub": str(huggingface_hub.__version__),
            "numpy": str(np.__version__),
            "scipy": str(scipy.__version__),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": device,
        },
    }


def output_summary(result) -> dict:
    stable = canonical(result)
    fields = {key: json_digest(value) for key, value in stable.items()}
    tensor_metrics = []

    def inspect(value, path="$"):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            finite = torch.isfinite(tensor)
            finite_count = int(finite.sum().item())
            finite_values = tensor[finite].to(torch.float64)
            rms = float(torch.sqrt(torch.mean(finite_values.square())).item()) if finite_count else None
            metric = {
                "path": path,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "finite_values": finite_count,
                "num_values": tensor.numel(),
                "rms_finite": rms,
            }
            tensor_metrics.append(metric)
            # Grammar-masked scores intentionally contain -inf. Model logits,
            # audio embeddings and decoder embeddings must be finite and useful.
            if any(label in path for label in ("logits", "encoder_last_hidden_state", "backbone_last_hidden_state",
                                                "mixed_hidden_state", "input_hidden_state", "decoder_hidden_state")):
                if finite_count != tensor.numel() or not rms or rms <= 0:
                    raise AssertionError(f"Non-finite or trivial model tensor: {path}: {metric}")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                inspect(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                inspect(item, f"{path}[{index}]")

    inspect(result)
    return {
        "sha256": json_digest(stable),
        "field_sha256": fields,
        "event_count": len(result.get("events", [])),
        "melody_note_count": sum(
            len(event.get("values", {}).get("melody", [])) for event in result.get("events", [])
        ),
        "token_count": sum(int(torch.as_tensor(tokens).numel()) for tokens in result.get("tokens", [])),
        "tensor_windows": len(result.get("tensors", [])),
        "tensor_metrics": tensor_metrics,
        "warnings": list(result.get("warnings", [])),
    }


def compare_outputs(left, right, label):
    left_stable, right_stable = canonical(left), canonical(right)
    differences = in_memory_differences(left, right)
    return {
        "label": label,
        "equal": not differences,
        "left_sha256": json_digest(left_stable),
        "right_sha256": json_digest(right_stable),
        "differences": differences,
    }


def check_reference(reference_path: Path, fingerprint: dict, cases: dict) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("schema_version") != 1:
        raise ValueError(f"Unsupported reference schema in {reference_path}")
    reference_fingerprint = dict(reference.get("fingerprint") or {})
    current_fingerprint = dict(fingerprint)
    # Backend hashes are recorded to explain which source produced the golden,
    # but changed backend code is exactly what a regression replay should test.
    reference_fingerprint.pop("backend_files", None)
    current_fingerprint.pop("backend_files", None)
    differences = first_differences(reference_fingerprint, current_fingerprint)
    if differences:
        raise ValueError("Reference fingerprint mismatch: " + json.dumps(differences[:8], sort_keys=True))
    for name, details in cases.items():
        expected = reference.get("cases", {}).get(name, {}).get("output", {}).get("sha256")
        if expected is None:
            raise ValueError(f"Reference has no output hash for case {name!r}")
        if expected != details["output"]["sha256"]:
            raise ValueError(
                f"Golden output mismatch for {name}: expected {expected}, got {details['output']['sha256']}"
            )


def main() -> int:
    from transformers import AutoModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="real audio fixture to transcribe")
    parser.add_argument("--cache-dir", type=Path, required=True, help="verified package checkpoint cache")
    parser.add_argument("--device", default="cuda", help="inference device (default: cuda)")
    parser.add_argument("--output", type=Path, help="write the run report JSON to this path")
    parser.add_argument("--reference", type=Path, help="compare against a golden JSON reference")
    parser.add_argument("--record", type=Path, help="explicitly write/replace a golden JSON reference")
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve(strict=True)
    if not audio_path.is_file():
        parser.error(f"audio input is not a file: {audio_path}")
    cache_dir = args.cache_dir.expanduser().resolve(strict=True)

    # Import the project backend only after parsing command-line options.
    from sheetsage.sheetsage2 import SheetSage2Session
    from sheetsage.sheetsage2_checkpoints import CheckpointStore

    store = CheckpointStore(cache_dir=cache_dir)
    directories = store.prepare(local_files_only=True)
    cache_session = SheetSage2Session(cache_dir=cache_dir, device=args.device, local_files_only=True)
    fingerprint = artifact_fingerprint(cache_session, audio_path, args.device)

    torch.set_num_threads(min(4, torch.get_num_threads()))
    from sheetsage.device import resolve_device

    resolved_device = resolve_device(args.device)
    upstream = AutoModel.from_pretrained(
        str(directories["sheetsage2"]),
        base_model_path=str(directories["mert2"]),
        trust_remote_code=True,
        local_files_only=True,
    ).eval().to(resolved_device)

    session = SheetSage2Session(cache_dir=cache_dir, device=args.device, local_files_only=True)
    comparisons = []
    case_reports = {}
    started = time.monotonic()
    try:
        with patch.object(AutoModel, "from_pretrained", wraps=AutoModel.from_pretrained) as load_spy:
            session.load()
            loads_after_load = load_spy.call_count
            if loads_after_load != 2:
                raise AssertionError(
                    f"Expected 2 AutoModel.from_pretrained calls during wrapper load "
                    f"(adapter plus nested MERT), saw {loads_after_load}"
                )

            zero_audio = np.zeros(7 * 24_000, dtype=np.float32)
            cases = {
                "real": (audio_path, {}, REAL_OPTIONS),
                "zeros": (zero_audio, {"sampling_rate": 24_000}, {}),
            }
            for name, (audio, input_options, export_options) in cases.items():
                options = {**input_options, **export_options}
                direct_started = time.monotonic()
                direct = upstream.transcribe(audio, **options)
                direct_seconds = time.monotonic() - direct_started

                wrapper_started = time.monotonic()
                wrapped_first = session.infer(audio, **options)
                wrapper_first_seconds = time.monotonic() - wrapper_started
                comparisons.append(compare_outputs(direct, wrapped_first, f"{name}: upstream vs wrapper"))

                wrapper_repeat_started = time.monotonic()
                wrapped_second = session.infer(audio, **options)
                wrapper_repeat_seconds = time.monotonic() - wrapper_repeat_started
                comparisons.append(compare_outputs(wrapped_first, wrapped_second, f"{name}: wrapper repeat"))

                if load_spy.call_count != loads_after_load:
                    raise AssertionError(
                        "AutoModel.from_pretrained was called during infer(): "
                        f"count changed from {loads_after_load} to {load_spy.call_count}"
                    )
                case_reports[name] = {
                    "options": options,
                    "output": output_summary(direct),
                    "timings_seconds": {
                        "upstream": direct_seconds,
                        "wrapper_first": wrapper_first_seconds,
                        "wrapper_repeat": wrapper_repeat_seconds,
                    },
                }
    finally:
        session.close()
        del upstream
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "status": "passed" if all(item["equal"] for item in comparisons) else "failed",
        "fingerprint": fingerprint,
        "cache": {key: {"path": value["path"], "revision": value["revision"], "complete": value["complete"]}
                  for key, value in cache_session.cache_info()["components"].items()},
        "model_load_calls": {"during_wrapper_load": loads_after_load, "after_repeated_infer": load_spy.call_count},
        "cases": case_reports,
        "comparisons": comparisons,
        "total_seconds": time.monotonic() - started,
    }
    if report["status"] != "passed":
        raise AssertionError("Output comparison failed: " + json.dumps(comparisons, indent=2, sort_keys=True))
    if args.reference:
        check_reference(args.reference.expanduser().resolve(), fingerprint, case_reports)
        report["reference"] = str(args.reference.expanduser().resolve())
    if args.record:
        record_path = args.record.expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps({
            "schema_version": 1,
            "fingerprint": fingerprint,
            "cases": {
                name: {"options": details["options"], "output": details["output"]}
                for name, details in case_reports.items()
            },
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["recorded_reference"] = str(record_path)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
