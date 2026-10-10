"""Satellite imagery from Esri World Imagery (slippy-map tiles)."""

from __future__ import annotations

import math

import numpy as np

from . import download, geo

TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE = geo.TILE_SIZE
MAX_TILES = 4000
MAX_ZOOM = 18


def choose_zoom(radius: float, grid_n: int, lat0: float, min_zoom: int = 1) -> int:
    """Pick the coarsest zoom whose ground texture beats the terrain grid.

    The dominant cost is tile downloads, so we target roughly 1.5x the terrain
    grid resolution and never finer than that.
    """
    target_m_per_px = (2.0 * radius) / (grid_n * 1.5)
    for z in range(min_zoom, MAX_ZOOM + 1):
        ground_m_per_px = geo.meters_per_pixel(z) * math.cos(math.radians(lat0))
        if ground_m_per_px <= target_m_per_px:
            return z
    return MAX_ZOOM


def sample_imagery(
    lat: np.ndarray, lon: np.ndarray, zoom: int, cache_dir: str
) -> np.ndarray:
    """Return an (..., 3) float32 RGB texture in [0, 1] for given lat/lon.

    Downloads the tile mosaic covering the bounding box and bilinearly
    samples each position.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))

    n = float(2 ** zoom)
    tx_min = math.floor((lon_min + 180.0) / 360.0 * n)
    tx_max = math.floor((lon_max + 180.0) / 360.0 * n)
    ty_min = math.floor(geo.tile_xy(lat_max, lon_min, zoom)[1])
    ty_max = math.floor(geo.tile_xy(lat_min, lon_min, zoom)[1])
    tx_min = max(tx_min, 0)
    ty_min = max(ty_min, 0)
    tx_max = min(tx_max, int(n) - 1)
    ty_max = min(ty_max, int(n) - 1)

    n_tiles = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
    if n_tiles > MAX_TILES:
        raise ValueError(
            f"Imagery request would need {n_tiles} tiles (>{MAX_TILES}); "
            "reduce --extent-km or --terrain-res"
        )

    width = (tx_max - tx_min + 1) * TILE
    height = (ty_max - ty_min + 1) * TILE
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)

    jobs = []
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            url = TILE_URL.format(z=zoom, y=ty, x=tx)
            key = f"esri/{zoom}/{tx}/{ty}.jpg"
            jobs.append((url, key))

    results = download.fetch_many(jobs, cache_dir)

    for row in range(ty_max - ty_min + 1):
        for col in range(tx_max - tx_min + 1):
            got = results[row * (tx_max - tx_min + 1) + col]
            if got is None:
                continue
            tile = _decode_tile(got)
            if tile is None:
                continue
            oy = row * TILE
            ox = col * TILE
            mosaic[oy : oy + TILE, ox : ox + TILE] = tile

    mpp = geo.meters_per_pixel(zoom)
    left_x = -geo.MERCATOR_HALF + tx_min * TILE * mpp
    top_y = geo.MERCATOR_HALF - ty_min * TILE * mpp

    mx, my = geo.mercator(lat, lon)
    px = (mx - left_x) / mpp
    py = (top_y - my) / mpp
    return _bilinear_sample_rgb(mosaic, px, py)


def _decode_tile(path: str) -> np.ndarray | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception:  # noqa: BLE001
        return None


def _bilinear_sample_rgb(
    mosaic: np.ndarray, px: np.ndarray, py: np.ndarray
) -> np.ndarray:
    """Bilinearly sample an (H, W, 3) uint8 mosaic at float pixel coords.

    Positions outside the mosaic resolve to black (0,0,0); callers generally
    never sample outside the bounding box they requested.
    """
    H, W = mosaic.shape[:2]
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    tx = (px - x0).astype(np.float32)
    ty = (py - y0).astype(np.float32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    c00 = mosaic[y0c, x0c].astype(np.float32)
    c10 = mosaic[y0c, x1c].astype(np.float32)
    c01 = mosaic[y1c, x0c].astype(np.float32)
    c11 = mosaic[y1c, x1c].astype(np.float32)

    tx = tx[..., None]
    ty = ty[..., None]
    result = (
        c00 * (1 - tx) * (1 - ty)
        + c10 * tx * (1 - ty)
        + c01 * (1 - tx) * ty
        + c11 * tx * ty
    ) / 255.0
    result[~valid] = 0.0
    return result.astype(np.float32)