# Hybrid CPU + GPU execution

`--backend hybrid` is for a CUDA-capable machine where one GPU and the host CPU
should both have useful work. It is deliberately a pipeline, not an inaccurate
claim that one numerical array is being updated by NumPy and CuPy at once.

- The numerical solver stays on the GPU (CuPy, or CUDA PyTorch for the neural
  wall).
- Two CPU workers encode completed presentation frames and commit them to disk
  atomically while the GPU starts the next solver update.
- At most two frames wait in memory. Completion waits for the queue, so a run
  is never marked complete before every JPEG exists.

This is the appropriate arrangement for the existing demos: the GPU has
regular array/tensor work, while colour conversion, Pillow encoding, metadata
and filesystem I/O are CPU work. It produces real CPU/GPU overlap without
copying a simulation state between devices on every time step, which would
usually make the run slower.

## Local use

Choose **GPU + CPU pipeline** in the viewer, or run:

```bash
python run_demo.py reaction_diffusion --profile desktop --backend hybrid
```

Hybrid requires a functioning CUDA backend. It fails rather than quietly
falling back to CPU; choose `auto` or `cpu` if the machine has no CUDA device.

## Leonardo use

The Leonardo Booster has four A100 GPUs and a 32-core Intel CPU per node. The
provided job asks SLURM for one GPU and eight host CPU cores, which is a
well-contained slice of that node for one hybrid demo:

```bash
sbatch --account="$LEONARDO_ACCOUNT" \
  --export=ALL,DEMO=reaction_diffusion,FRAMES=90,PROFILE=leonardo,LEONARDO_DEMO_BACKEND=hybrid \
  slurm/run_demo.sbatch
```

The CPU cores are on the same Booster node as the requested GPU. They are not
a DCGP allocation, and CPU-only work on DCGP should be submitted as a separate
job rather than assumed to share a GPU job's memory. For a later MPI/multi-GPU
version, use one rank per GPU and give each rank its matching CPU-core share;
CINECA's current container example uses four tasks and eight CPUs per task for
the four-GPU node.

## What professional deep zoom does differently

The crystal viewer uses a tiled image pyramid for immediate coarse previews,
then creates and caches a native-resolution tile at each requested log2 zoom
level. The browser therefore never uses a 300x CSS-scaled JPEG as the final
image. This is the same broad image-pyramid approach used by Deep Zoom viewers.

For escape-time sets such as Mandelbrot, serious *extreme* deep zoom renderers
add arbitrary-precision reference orbits and perturbation/series methods. That
is a different numerical problem from this procedural crystal generator; its
fix is regenerating the geometry from the rule at each tile scale. The current
limit of level 40 is about 1.1 trillion times the base view, well beyond the
requested 300x, while keeping tile indices and IEEE double coordinates safe.

Sources consulted: [CINECA Leonardo hardware and partitions](https://docs.hpc.cineca.it/hpc/leonardo.html), [CINECA's GPU container resource example](https://docs.hpc.cineca.it/services/singularity.html), and [Ultra Fractal's perturbation overview](https://www.ultrafractal.com/help/formulas/perturbationcalculations.html).
