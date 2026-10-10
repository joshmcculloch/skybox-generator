"""SRTM elevation data (1 arc-second, "skadi" public S3 bucket).

Tiles are 1x1 degree, 3601x3601 signed big-endian int16 samples.  Values are
metres above the EGM96 geoid; -32768 marks void (e.g. open water).
"""

from __future__ import annotations

import gzip
import math

import numpy as np

from . import download

BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
GRID = 3601  # samples per 1x1 degree tile (1 arc-second)
VOID = -32768.0
LAT_LIMIT = 60.0  # SRTM only covers -60..+60


def _tile_path(lat_int: int, lon_int: int) -> str:
    lat_dir = "N" if lat_int >= 0 else "S"
    lon_dir = "E" if lon_int >= 0 else "W"
    name = f"{lat_dir}{abs(lat_int):02d}{lon_dir}{abs(lon_int):03d}.hgt.gz"
    return f"{lat_dir}{abs(lat_int):02d}/{name}"


def _tile_url(path: str) -> str:
    return f"{BASE_URL}/{path}"


def _load_tile(lat_int: int, lon_int: int, cache_dir: str) -> np.ndarray | None:
    """Return a 3601x3601 float64 array, or None if the tile is unavailable."""
    if abs(lat_int) >= LAT_LIMIT:
        return None
    path = _tile_path(lat_int, lon_int)
    try:
        local = download.fetch(_tile_url(path), cache_dir, path)
    except download.DownloadError:
        return None

    with gzip.open(local, "rb") as fh:
        raw = fh.read()
    arr = np.frombuffer(raw, dtype=">i2").reshape(GRID, GRID).astype(np.float64)
    arr[arr == VOID] = np.nan
    return arr


def sample_srtm(lat: np.ndarray, lon: np.ndarray, cache_dir: str) -> np.ndarray:
    """Bilinearly sample SRTM elevation (m) at arbitrary lat/lon (degrees).

    `lat`/`lon` must be broadcast-compatible float arrays.  Positions with no
    data resolve to NaN.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    out = np.full(np.broadcast_shapes(lat.shape, lon.shape), np.nan, dtype=np.float64)

    lat_int_lo = math.floor(np.min(lat))
    lat_int_hi = math.floor(np.max(lat))
    lon_int_lo = math.floor(np.min(lon))
    lon_int_hi = math.floor(np.max(lon))

    for la in range(lat_int_lo, lat_int_hi + 1):
        if abs(la) >= LAT_LIMIT:
            continue
        for lo in range(lon_int_lo, lon_int_hi + 1):
            tile = _load_tile(la, lo, cache_dir)
            if tile is None:
                continue
            inside = (
                (lat >= la) & (lat < la + 1) & (lon >= lo) & (lon < lo + 1)
            )
            if not np.any(inside):
                continue

            # Row 0 is the *north* edge; col 0 is the *west* edge.
            fr = (la + 1 - lat[inside]) * (GRID - 1)
            fc = (lon[inside] - lo) * (GRID - 1)
            r0 = np.floor(fr).astype(np.int64)
            c0 = np.floor(fc).astype(np.int64)
            tr = fr - r0
            tc = fc - c0
            r1 = r0 + 1
            c1 = c0 + 1
            r0c = np.clip(r0, 0, GRID - 1)
            r1c = np.clip(r1, 0, GRID - 1)
            c0c = np.clip(c0, 0, GRID - 1)
            c1c = np.clip(c1, 0, GRID - 1)

            h = (
                tile[r0c, c0c] * (1 - tr) * (1 - tc)
                + tile[r0c, c1c] * (1 - tr) * tc
                + tile[r1c, c0c] * tr * (1 - tc)
                + tile[r1c, c1c] * tr * tc
            )
            out[inside] = h

    return out