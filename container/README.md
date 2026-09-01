# Container note

A container is useful for making the Python/CUDA environment reproducible between a workstation and Leonardo. The supplied definition uses CUDA 12.2, matching Leonardo's documented driver/toolkit generation, and installs the CUDA 12 family of CuPy wheels.

Build the `.sif` ahead of time and transfer it to CINECA storage; do not rely on an external registry being reachable from a compute node. Set `LEONARDO_CONTAINER=/shared/path/leonardo_demos.sif` in `config/leonardo.env`; the SLURM scripts then use `singularity exec --nv` for Booster jobs.

Run the build from the repository root so the `%files` entries resolve:

```bash
singularity build leonardo_demos.sif container/leonardo_demos.def
```

The container does not install PyTorch. The galaxy collision and other CuPy demos are complete without it. If the neural-network wall must train on a GPU, validate a CUDA-compatible PyTorch build separately or use CINECA's `cineca-ai` module with a virtual environment.
