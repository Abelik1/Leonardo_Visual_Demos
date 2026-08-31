"""Pre-baked multi-resolution tile pyramid.

The viewer used to "zoom" by CSS-scaling a finished JPEG, which magnifies
pixels and reveals nothing. A pyramid stores independently rendered tiles per
level, so each level is a fresh render of the underlying model rather than an
upscale of the level above it. The exhibition machine stays a pure playback
device: everything is written at run time and the viewer only fetches files.

Layout inside a run directory:

    zoom/
      zoom.json                 manifest
      L0/0_0.jpg
      L1/0_0.jpg L1/1_0.jpg ...
"""
from __future__ import annotations
import json
from pathlib import Path


def build_pyramid(render, out_dir: Path, cx=0.0, cy=0.0, span=1.0, levels=4,
                  tile=256, quality=88, progress_cb=None):
    """Render a `levels`-deep pyramid.

    `render(cx, cy, span, size)` must return a PIL image of the requested
    square window. Level L is a 2**L x 2**L grid, so the finest effective
    resolution is `tile * 2**levels` across.
    """
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    total=sum(4**l for l in range(levels+1)); made=0
    for level in range(levels+1):
        n=2**level
        sub=span/n
        d=out_dir/f"L{level}"; d.mkdir(exist_ok=True)
        for col in range(n):
            for row in range(n):
                # Tile centre in world coordinates. Row 0 is the top of the
                # image, so the y axis is walked downward.
                tcx=cx-span/2+sub*(col+.5)
                tcy=cy+span/2-sub*(row+.5)
                im=render(tcx,tcy,sub,(tile,tile))
                im.save(d/f"{col}_{row}.jpg",quality=quality)
                made+=1
                if progress_cb and made%16==0: progress_cb(made,total)
    manifest={"tile":tile,"levels":levels,"cx":cx,"cy":cy,"span":span,
              "max_resolution":tile*(2**levels)}
    (out_dir/"zoom.json").write_text(json.dumps(manifest,indent=2))
    return manifest
