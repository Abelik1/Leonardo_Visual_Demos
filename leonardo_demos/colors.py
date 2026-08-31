from __future__ import annotations
import numpy as np


def normalize(a, lo=None, hi=None, gamma=1.0):
    a=np.asarray(a, dtype=np.float32)
    lo=float(np.nanmin(a) if lo is None else lo)
    hi=float(np.nanmax(a) if hi is None else hi)
    x=(a-lo)/(hi-lo+1e-12)
    x=np.clip(x,0,1)
    if gamma != 1.0:
        x=x**gamma
    return x


def palette(a, kind="plasma", normalize_input=True):
    """Small dependency-free scientific-style palettes returning uint8 RGB.

    Set `normalize_input=False` when the data is already in [0, 1]. Rescaling a
    field that is legitimately uniform maps it to zero: a crystal zoom that
    lands wholly inside one thick branch is a screen of 1.0, and renormalising
    turned that solid white into solid background colour.
    """
    x=normalize(a) if normalize_input else np.clip(np.asarray(a,dtype=np.float32),0,1)
    if kind == "ice":
        r=np.clip(0.08 + 0.75*x**2,0,1)
        g=np.clip(0.12 + 0.88*x,0,1)
        b=np.clip(0.28 + 0.72*np.sqrt(x),0,1)
    elif kind == "fire":
        r=np.clip(3.0*x,0,1)
        g=np.clip(3.0*x-1.0,0,1)
        b=np.clip(3.0*x-2.0,0,1)
    elif kind == "cosmic":
        r=np.clip(0.05 + 1.15*x,0,1)
        g=np.clip(0.03 + 0.85*x**1.4,0,1)
        b=np.clip(0.15 + 0.95*np.sqrt(x),0,1)
    elif kind == "mono":
        r=g=b=x
    else:
        # dark purple -> blue -> orange -> white
        r=np.clip(0.08 + 1.3*x**1.6,0,1)
        g=np.clip(0.02 + 1.0*x**2.2,0,1)
        b=np.clip(0.18 + 1.2*x*(1-x)+0.6*x,0,1)
    return (np.stack([r,g,b],axis=-1)*255).astype(np.uint8)


def alpha_blend(base, overlay, alpha):
    return (base*(1-alpha)+overlay*alpha).clip(0,255).astype(np.uint8)
