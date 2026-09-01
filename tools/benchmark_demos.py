#!/usr/bin/env python3
"""Run a reproducible demo/backend matrix and publish machine-readable results."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leonardo_demos.backend import probe  # noqa: E402
from leonardo_demos.registry import DEMOS  # noqa: E402


def run_case(python: str, demo: str, profile: str, frames: int, backend: str,
             method: str, run_dir: Path) -> dict:
    command = [python, str(ROOT / "run_demo.py"), demo, "--profile", profile,
               "--frames", str(frames), "--backend", backend, "--method", method,
               "--timings", "--run-dir", str(run_dir)]
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    wall = time.perf_counter() - started
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    status = "complete" if process.returncode == 0 and meta.get("status") == "complete" else "failed"
    return {
        "demo": demo,
        "backend_requested": backend,
        "backend": meta.get("backend"),
        "method": method,
        "status": status,
        "wall_seconds": wall,
        "run_elapsed_seconds": meta.get("elapsed"),
        "timings": meta.get("timings", {}),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "returncode": process.returncode,
        "error": meta.get("error") or (process.stderr[-2000:] if process.returncode else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="benchmark")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--backends", default="cpu,gpu,hybrid")
    parser.add_argument("--demos", default=",".join(DEMOS))
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()

    demos = [value.strip() for value in args.demos.split(",") if value.strip()]
    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    unknown = sorted(set(demos) - set(DEMOS))
    if unknown:
        parser.error(f"unknown demos: {', '.join(unknown)}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output.resolve()
    artifact_root = output / "artifacts" / stamp
    artifact_root.mkdir(parents=True, exist_ok=True)

    results = []
    total = sum(len(backends) * (1 + (len(DEMOS[d].methods) - 1 if args.all_methods else 0)) for d in demos)
    index = 0
    for demo in demos:
        cls = DEMOS[demo]
        methods = [cls.default_method]
        if args.all_methods:
            methods += [m for m in cls.methods if m != cls.default_method]
        for method in methods:
            for backend in backends:
                index += 1
                print(f"[{index}/{total}] {demo} · {backend} · {method}", flush=True)
                if backend not in cls.supported_backends:
                    results.append({"demo": demo, "backend_requested": backend,
                                    "backend": None, "method": method,
                                    "status": "unsupported", "wall_seconds": None,
                                    "run_elapsed_seconds": None, "timings": {},
                                    "run_dir": None, "returncode": None,
                                    "error": f"supported backends: {', '.join(cls.supported_backends)}"})
                    continue
                run_dir = artifact_root / f"{demo}_{method}_{backend}"
                results.append(run_case(args.python, demo, args.profile, args.frames,
                                        backend, method, run_dir))

    payload = {
        "schema_version": 1,
        "created": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": args.profile,
        "frames": args.frames,
        "python": args.python,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "backends": probe(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_partition": os.getenv("SLURM_JOB_PARTITION"),
        },
        "results": results,
        "notes": [
            "The benchmark profile is a fixed workstation-scale workload; it is not a Leonardo-node performance claim.",
            "GPU stage boundaries synchronize CUDA for honest elapsed times.",
            "Hybrid JPEG and write stages overlap solver work, so stage totals are not additive.",
            "Unsupported means the demo is intentionally scheduled on Leonardo's CPU partition rather than pretending to use a GPU.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    report = output / f"results_{stamp}.json"
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.copyfile(report, output / "latest.json")
    if not args.keep_runs:
        shutil.rmtree(artifact_root, ignore_errors=True)
    print(report)
    return 0 if all(r["status"] in {"complete", "unsupported"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
