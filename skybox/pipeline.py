"""Top-level pipeline: geolocate -> sample terrain -> render skybox."""

from __future__ import annotations

import os

import numpy as np

from . import dem, geo, imagery, render, sky


def default_cache_dir() -> str:
    base = os.environ.get("SKYBOX_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "skybox"
    )
    return base


def generate(
    lat: float,
    lon: float,
    sky_name: str = "clear",
    out: str | None = None,
    width: int = 2048,
    radius_km: float = 60.0,
    terrain_res: int = 1024,
    eye_height: float = 2.0,
    heading_deg: float = 0.0,
    sun_azimuth: float | None = None,
    sun_elevation: float | None = None,
    zoom: int | None = None,
    cache_dir: str | None = None,
    seed: int | None = None,
    n_steps: int = 192,
    visibility: float | None = None,
    ambient: float = 0.4,
    lod_count: int = 8,
):
    """Generate an equirectangular skybox image (H, W, 3) uint8.

    Returns the image array; writes a PNG to `out` when provided.
    """
    cache = cache_dir or default_cache_dir()
    radius = radius_km * 1000.0
    H, W = width // 2, width
    if H * 2 != W:
        raise ValueError("width must be even (equirectangular 2:1 aspect)")

    ground = dem.sample_srtm(
        np.array([[lat]], dtype=np.float64),
        np.array([[lon]], dtype=np.float64),
        cache,
    )[0, 0]
    if not np.isfinite(ground):
        ground = 0.0
    cam_z = ground + eye_height

    lat2d, lon2d = geo.latlon_grid(lat, lon, radius, terrain_res)

    heights = dem.sample_srtm(lat2d, lon2d, cache).astype(np.float32)
    n_nan = int(np.isnan(heights).sum())
    heights = np.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
    if n_nan:
        print(f"  filled {n_nan} water/void samples with sea level")

    base_zoom = zoom if zoom is not None else imagery.choose_zoom(
        radius, terrain_res, lat
    )

    lod_count = max(1, int(lod_count))
    levels = []
    for level in range(lod_count):
        r = radius / (2 ** level)
        if level == 0:
            l_lat, l_lon = lat2d, lon2d
            l_heights = heights
        else:
            l_lat, l_lon = geo.latlon_grid(lat, lon, r, terrain_res)
            l_heights = dem.sample_srtm(l_lat, l_lon, cache).astype(np.float32)
            n_nan = int(np.isnan(l_heights).sum())
            l_heights = np.nan_to_num(
                l_heights, nan=0.0, posinf=0.0, neginf=0.0
            )
            if n_nan:
                print(f"  LOD {level}: filled {n_nan} water/void samples")
        z = min(base_zoom + level, imagery.MAX_ZOOM)
        colors = imagery.sample_imagery(l_lat, l_lon, z, cache)
        print(
            f"  LOD {level}: radius {r / 1000.0:.1f} km, "
            f"grid {terrain_res}px, imagery zoom {z}"
        )
        levels.append(render.Terrain(l_heights, colors, r))

    terrain = render.TerrainLOD(levels)

    _, _, dx, dy, dz = render.build_directions(width, heading_deg)
    theta = np.arctan2(dy, dx).astype(np.float32)
    phi = np.arcsin(np.clip(dz, -1.0, 1.0)).astype(np.float32)
    sun_azimuth, sun_elevation = sky.resolve_sun(sky_name, sun_azimuth, sun_elevation)
    sun_dir = sky.sun_direction(sun_azimuth, sun_elevation).astype(np.float32)
    sky_rgb = sky.render_sky(
        theta, phi, sky_name, sun_azimuth, sun_elevation, seed
    )

    image = render.render_equirect(
        terrain,
        sky_rgb,
        cam_z,
        width=width,
        heading_deg=heading_deg,
        n_steps=n_steps,
        visibility=visibility,
        sun_dir=sun_dir,
        ambient=ambient,
    )

    if out:
        from PIL import Image

        Image.fromarray(image).save(out)
        print(f"  wrote {out}")

    return image