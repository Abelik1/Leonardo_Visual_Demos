# Container note

A container is useful for making the Python/CUDA environment reproducible between a workstation and Leonardo, but the exact base image and CUDA/PyTorch wheels should be chosen using the software versions available during the event.

`leonardo_demos.def` is a starting template, not a promise that a particular external registry will be reachable from a compute node. It is usually safer to build the `.sif` ahead of time and transfer it to CINECA storage.
