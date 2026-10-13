"""Opt-in cached real-audio verification of shared vs native transcription.

Compares every native field in memory (excluding timing/memory measurements),
then verifies shared serialization, native preservation and structure-only use.
Reads: sheetsage public APIs and verify_sheetsage2 comparison helpers. No fixture
regeneration, weight publishing or quality benchmark is performed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from verify_sheetsage2 import compare_outputs

from sheetsage import (
    SheetSage1Options,
    SheetSage2Options,
    SheetSage2Session,
    SheetSageSession,
    SheetSageTranscriber,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", choices=("sheetsage1", "sheetsage2"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--preset", choices=("default", "paper"), default="default")
    parser.add_argument("--waveform", action="store_true", help="V2: decode via FFmpeg to a24kHz waveform")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audio = args.audio
    if args.waveform:
        if args.model != "sheetsage2":
            parser.error("waveform input applies only to sheetsage2")
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-ac", "1",
                                  "-ar", "24000", "-f", "f32le", "pipe:1"],
                                 capture_output=True, check=True)
        audio = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if args.model == "sheetsage1":
        if args.preset != "default":
            parser.error("preset applies only to sheetsage2")
        options = SheetSage1Options(return_intermediaries=True)
        session_class = SheetSageSession
    else:
        options = SheetSage2Options(cache_dir=args.cache_dir, local_files_only=True,
                                    preset=args.preset, export_logits=True,
                                    export_scores=True, export_embeddings=True,
                                    sampling_rate=24000 if args.waveform else None)
        session_class = SheetSage2Session
    args.output.mkdir(parents=True, exist_ok=True)
    with session_class(device=args.device, **options.session_kwargs()) as direct:
        reference = direct.infer(audio, **options.inference_kwargs())
    checks = []
    with SheetSageTranscriber(args.model, device=args.device, options=options) as session:
        result = session.infer(audio)
        checks.append(compare_outputs(reference, result.native, "direct vs shared.native"))
        repeated = session.infer(audio)
        checks.append(compare_outputs(result.native, repeated.native, "shared repeated native"))
        checks.append(compare_outputs(result.to_dict(), repeated.to_dict(), "shared repeated normalized"))
        assert result.notes, "Real fixture must produce nontrivial notes"
        assert all(s.state in ("ok", "unsupported") for s in result.statuses.values()), result.statuses
        result.to_json(args.output / "result.json")
        result.to_midi(args.output / "shared.mid")
        if args.model == "sheetsage2":
            structure = session.infer(audio, options=replace(
                options, prompts=("timestamp", "structure"), export_logits=False,
                export_scores=False, export_embeddings=False))
            assert structure.structure is not None
            assert structure.statuses["structure"].state == "ok"
            assert structure.notes is None
            structure.to_json(args.output / "structure.json")
    assert all(check["equal"] for check in checks), checks
    report = {"model": args.model, "device": args.device, "preset": args.preset,
              "input_kind": "waveform" if args.waveform else "file",
              "notes": len(result.notes), "checks": checks,
              "quality_benchmark": False}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
