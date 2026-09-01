# AI Plasma Guardian

## Purpose

Trains a small feedback policy in a reduced differentiable plasma-control
environment and compares its controlled trajectory with an uncontrolled one.

## Implementation map

- Environment, training, and renderer: `leonardo_demos/demos/plasma_guardian.py`
- Parameter: instability drive
- Main frames: vessel, coils, controlled plasma, risk island, baseline outline
- Optional policy graph: `overlays/network/frame_NNNN.jpg`
- Readouts: per-frame JSON shown through the frontend overlay system

## Visual language

- Cyan plasma: controlled state
- Smooth amber island inside it: tearing-risk proxy; larger means less stable
- Faint red outline: uncontrolled reference trajectory
- Cyan/orange/purple banks: aggregate coil commands

## Scientific boundary

This is a reduced control environment, not a tokamak equilibrium, tearing-mode,
or disruption solver. PyTorch trains the policy when available; the analytical
fallback is explicitly labelled and is not learning.
