"""Fail a Leonardo job early when its allocation or Python stack is wrong."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import tempfile


def check(backend: str, run_root: Path, demo: str | None = None) -> dict:
    backend=backend.lower()
    report={
        "backend_requested":backend,
        "demo":demo,
        "hostname":os.getenv("HOSTNAME"),
        "slurm_job_id":os.getenv("SLURM_JOB_ID"),
        "slurm_partition":os.getenv("SLURM_JOB_PARTITION"),
        "slurm_cpus_per_task":os.getenv("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    for package in ("numpy","PIL"):
        module=importlib.import_module(package)
        report[package]=getattr(module,"__version__","available")

    run_root.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".preflight-",dir=run_root,delete=True):
        report["run_root_writable"]=True

    gpu_requested=backend in {"gpu","cuda","cupy","hybrid"}
    partition=report["slurm_partition"]
    if partition and gpu_requested and not partition.startswith("boost"):
        raise RuntimeError(f"GPU backend requested on non-Booster partition: {partition}")
    if partition and not gpu_requested and backend in {"cpu","numpy"} and partition.startswith("boost"):
        raise RuntimeError(f"CPU backend would waste a Booster allocation: {partition}")

    if gpu_requested:
        cupy=importlib.import_module("cupy")
        count=cupy.cuda.runtime.getDeviceCount()
        if count < 1:
            raise RuntimeError("GPU job has no CUDA device visible to CuPy")
        props=cupy.cuda.runtime.getDeviceProperties(0)
        name=props.get("name",props.get(b"name","unknown"))
        if isinstance(name,bytes):
            name=name.decode(errors="replace")
        report.update({"cupy":cupy.__version__,"cuda_devices":count,
                       "cuda_device_0":name})
        if demo == "neural_wall":
            torch=importlib.import_module("torch")
            if not torch.cuda.is_available():
                raise RuntimeError("neural_wall GPU job requires a CUDA-enabled PyTorch build")
            report["torch_cuda"]=torch.cuda.get_device_name(0)
    elif backend not in {"cpu","numpy"}:
        raise ValueError(f"Unsupported Leonardo backend: {backend}")
    return report


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",required=True)
    parser.add_argument("--demo")
    parser.add_argument("--run-root",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(check(args.backend,args.run_root,args.demo),indent=2))


if __name__ == "__main__":
    main()
