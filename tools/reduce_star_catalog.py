"""Reduce a deprojected stellar catalogue to mass-weighted M31 tracers.

Input is a CSV containing at least x/y positions in kpc relative to M31's
centre.  The optional weight column can be luminosity, inferred stellar mass,
or simply 1.  The tool combines every spatial cell into its weighted centre of
mass and writes the compact asset read by galaxy_collision.py.

Example (after downloading and deprojecting a PHAT/PHAST/Gaia catalogue):
    python tools/reduce_star_catalog.py m31_deprojected.csv \
        data/m31_catalog_reduced.npz --x x_kpc --y y_kpc --weight flux --bins 180
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np


def read_rows(path: Path, x_name: str, y_name: str, weight_name: str | None):
    rows=[]
    with path.open(newline='',encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            try:
                x=float(row[x_name]); y=float(row[y_name])
                w=float(row[weight_name]) if weight_name else 1.0
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(w) and w>0:
                rows.append((x,y,w))
    if not rows:
        raise ValueError('no finite, positive-weight rows found; check column names')
    return np.asarray(rows,dtype=np.float64)


def reduce(rows: np.ndarray, bins: int):
    xy=rows[:,:2]; w=rows[:,2]
    lo=xy.min(axis=0); hi=xy.max(axis=0)
    span=np.maximum(hi-lo,1e-8)
    cell=np.minimum(((xy-lo)/span*bins).astype(np.int64),bins-1)
    key=cell[:,1]*bins+cell[:,0]
    unique,inv=np.unique(key,return_inverse=True)
    total=np.bincount(inv,weights=w)
    cx=np.bincount(inv,weights=w*xy[:,0])/total
    cy=np.bincount(inv,weights=w*xy[:,1])/total
    return np.column_stack([cx,cy]).astype(np.float32),total.astype(np.float32),lo,hi,len(unique)


def main():
    parser=argparse.ArgumentParser(description='Grid-reduce a deprojected M31 stellar catalogue.')
    parser.add_argument('input',type=Path); parser.add_argument('output',type=Path)
    parser.add_argument('--x',default='x_kpc'); parser.add_argument('--y',default='y_kpc')
    parser.add_argument('--weight',default=None,help='optional positive flux/mass column')
    parser.add_argument('--bins',type=int,default=180,help='cells per side (default: 180)')
    args=parser.parse_args()
    if args.bins<8: raise ValueError('--bins must be at least 8')
    rows=read_rows(args.input,args.x,args.y,args.weight)
    xy,weight,lo,hi,cells=reduce(rows,args.bins)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,xy_kpc=xy,weight=weight,
                        source=str(args.input.name),input_rows=len(rows),cells=cells,
                        bounds_kpc=np.vstack([lo,hi]))
    print(f'{len(rows):,} catalogue rows -> {cells:,} mass-weighted representatives: {args.output}')


if __name__=='__main__':
    main()
