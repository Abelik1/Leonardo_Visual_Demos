# Neural-network wall

## Purpose

Trains a batch of coordinate networks to reproduce an RGB image. Each network
maps `(x, y)` to `(R, G, B)`; it paints the image but does not classify or
recognise its subject.

## Implementation map

- Training and rendering: `leonardo_demos/demos/neural_wall.py`
- Browser drawing/upload/camera target: `web/app.js`
- Input payload: bounded RGB `target.png` stored inside the run
- Main frames: current winning reconstruction
- Optional diagnostic: `overlays/network/frame_NNNN.jpg`
- Reveal: wall of independently parameterised networks

## Compute

PyTorch performs genuine batched MLP training on CPU or CUDA. The explicitly
labelled surrogate is only a fallback when PyTorch is unavailable and must never
claim to be learning.
