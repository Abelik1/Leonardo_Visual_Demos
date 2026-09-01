#!/usr/bin/env python3
"""Fetch small, reproducible Gaia DR3 and PHAT v3 morphology samples.

The output is deliberately a reduced seed catalogue, not a claim that the
simulation follows individual observed stars. Gaia distances provide genuine
3-D Milky Way positions. PHAT provides projected M31 positions; those are
deprojected into M31's disc and receive a documented modelled vertical depth.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import socket
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

QUERY_URL = "https://datalab.noirlab.edu/query/query"
ANON_TOKEN = "anonymous.0.0.anon_access"
GAIA_SOURCE = "https://gea.esac.esa.int/archive/"
PHAT_SOURCE = "https://datalab.noirlab.edu/data/phat"


def query(sql: str, timeout: int = 300) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({
        "sql": sql, "ofmt": "csv", "out": "", "async": "False",
        "drop": "False", "profile": "default",
    })
    request = urllib.request.Request(
        f"{QUERY_URL}?{params}",
        headers={
            "Content-Type": "text/ascii",
            "X-DL-AuthToken": ANON_TOKEN,
            "X-DL-TimeoutRequest": str(timeout),
            "X-DL-ClientVersion": "catalog-reducer-1",
            "X-DL-OriginIP": "127.0.0.1",
            "X-DL-OriginHost": socket.gethostname(),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    if text.startswith("Error"):
        raise RuntimeError(text.strip())
    return list(csv.DictReader(io.StringIO(text)))


def fetch_gaia(per_sector: int = 350) -> np.ndarray:
    rows: list[dict[str, str]] = []
    # Longitude sectors prevent a small reduced download being dominated by
    # one scanning-law/table-order region. random_index is Gaia's own random
    # row ordering key, not a locally invented selection.
    for lower in range(0, 360, 30):
        upper = lower + 30
        rows.extend(query(
            f"select top {per_sector} source_id,l,b,distance_gspphot,phot_g_mean_mag "
            "from gaia_dr3.gaia_source "
            "where distance_gspphot between 100 and 15000 "
            "and phot_g_mean_mag < 17 "
            f"and l >= {lower} and l < {upper} and random_index < 120000000"
        ))
    l = np.radians([float(row["l"]) for row in rows])
    b = np.radians([float(row["b"]) for row in rows])
    distance = np.array([float(row["distance_gspphot"]) for row in rows]) / 1000.0
    cb = np.cos(b)
    # Right-handed Galactocentric frame: Sun at (+R0, 0, z_sun).
    x = 8.277 - distance * cb * np.cos(l)
    y = distance * cb * np.sin(l)
    z = 0.0208 + distance * np.sin(b)
    xyz = np.column_stack((x, y, z))
    good = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz[:, :2], axis=1) < 32) & (np.abs(z) < 5)
    return xyz[good].astype(np.float32)


def fetch_phat(limit: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    rows = query(
        f"select top {limit} ra,dec,f475w_vega,f814w_vega,random_id "
        "from phat_v3.phot_mod where random_id < 0.01 "
        "and (f475w_gst_flag=1 or f814w_gst_flag=1)"
    )
    ra = np.array([float(row["ra"]) for row in rows])
    dec = np.array([float(row["dec"]) for row in rows])
    f475 = np.array([float(row["f475w_vega"]) for row in rows])
    f814 = np.array([float(row["f814w_vega"]) for row in rows])
    # Tangent-plane projection at the M31 centre, followed by deprojection
    # with position angle 38 degrees and inclination 77 degrees. PHAT has no
    # useful per-star line-of-sight depth at M31's distance.
    ra0, dec0, distance = 10.6847083, 41.26875, 785.0
    east = np.radians(ra - ra0) * math.cos(math.radians(dec0)) * distance
    north = np.radians(dec - dec0) * distance
    pa = math.radians(38.0)
    major = east * math.sin(pa) + north * math.cos(pa)
    minor_projected = east * math.cos(pa) - north * math.sin(pa)
    minor = minor_projected / math.cos(math.radians(77.0))
    xy = np.column_stack((major, minor))
    colour = f475 - f814
    good = np.isfinite(xy).all(axis=1) & np.isfinite(colour) & (np.linalg.norm(xy, axis=1) < 35)
    return xy[good].astype(np.float32), np.clip(colour[good], -1, 5).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "galaxy_observations.npz")
    args = parser.parse_args()
    mw_xyz = fetch_gaia()
    m31_xy, m31_colour = fetch_phat()
    metadata = {
        "schema_version": 1,
        "gaia": {"release": "DR3", "source": GAIA_SOURCE,
                 "meaning": "observed 3-D Galactocentric seed positions",
                 "rows": int(len(mw_xyz))},
        "phat": {"release": "v3", "source": PHAT_SOURCE,
                 "meaning": "observed sky pattern deprojected at PA=38 deg, inclination=77 deg; depth modelled later",
                 "rows": int(len(m31_xy))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, mw_xyz=mw_xyz, m31_xy=m31_xy,
                        m31_colour=m31_colour,
                        metadata=np.array(json.dumps(metadata)))
    print(json.dumps(metadata, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
