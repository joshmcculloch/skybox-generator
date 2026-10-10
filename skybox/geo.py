"""Geodesy helpers: Web-Mercator / slippy-tile math and a local ENU frame."""

from __future__ import annotations

import numpy as np

# WGS84 (well, spherical-mercator convention) constants.
R_EARTH = 6378137.0
MERCATOR_HALF = 20037508.342789244  # pi * R_EARTH
TILE_SIZE = 256
# Web-Mercator clips to roughly +-85.05 degrees latitude.
MAX_LAT = 85.05112878


# --------------------------------------------------------------------------- #
# Web Mercator (EPSG:3857)
# --------------------------------------------------------------------------- #
def mercator(lat: float | np.ndarray, lon: float | np.ndarray):
    """Forward projection to Web-Mercator metres. lat/lon in degrees."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.clip(lat, -MAX_LAT, MAX_LAT)
    x = R_EARTH * np.radians(lon)
    y = R_EARTH * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def inverse_mercator(x, y):
    """Inverse projection. Returns lat, lon in degrees."""
    lon = np.degrees(np.asarray(x, dtype=np.float64) / R_EARTH)
    lat = np.degrees(
        2.0 * np.arctan(np.exp(np.asarray(y, dtype=np.float64) / R_EARTH))
        - np.pi / 2.0
    )
    return lat, lon


def tile_xy(lat, lon, zoom):
    """Slippy-map tile coordinates (float, not integer) for a zoom level."""
    lat = np.clip(np.asarray(lat, dtype=np.float64), -MAX_LAT, MAX_LAT)
    lon = np.asarray(lon, dtype=np.float64)
    n = float(2 ** int(zoom))
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n
    return xtile, ytile


def meters_per_pixel(zoom):
    """Web-Mercator metres per pixel at a given integer zoom."""
    return (2.0 * MERCATOR_HALF) / (TILE_SIZE * (2.0 ** int(zoom)))


# --------------------------------------------------------------------------- #
# Local tangent frame (approximate equirectangular ENU about a centre point)
# --------------------------------------------------------------------------- #
def degrees_per_metre(lat: float):
    """Metres per degree of latitude/longitude at a given latitude.

    Uses standard small-circle approximations of the WGS84 ellipsoid.
    """
    lat = float(lat)
    phi = np.radians(lat)
    m_lat = (
        111132.954
        - 559.822 * np.cos(2.0 * phi)
        + 1.175 * np.cos(4.0 * phi)
        - 0.0023 * np.cos(6.0 * phi)
    )
    m_lon = (
        111415.13 * np.cos(phi)
        - 94.55 * np.cos(3.0 * phi)
        + 0.12 * np.cos(5.0 * phi)
    )
    return m_lat, m_lon


def enu_axes(radius: float, count: int):
    """Evenly spaced axis in local metres, centred on the camera.

    Returned axis covers [-radius, radius] with `count` samples.
    """
    return np.linspace(-radius, radius, int(count), dtype=np.float64)


def enu_to_latlon(east: np.ndarray, north: np.ndarray, lat0: float, lon0: float):
    """Map local east/north (metres) to geographic lat/lon (degrees).

    `east`/`north` may be scalars or broadcastable arrays.
    """
    m_lat, m_lon = degrees_per_metre(lat0)
    lat = lat0 + np.asarray(north) / m_lat
    lon = lon0 + np.asarray(east) / m_lon
    return lat, lon


def latlon_grid(lat0: float, lon0: float, radius: float, count: int):
    """Build the shared lat/lon grid for the terrain sampling stage.

    Returns two (count, count) float64 arrays: lat2d, lon2d.
    """
    axis = enu_axes(radius, count)
    east, north = np.meshgrid(axis, axis)
    return enu_to_latlon(east, north, lat0, lon0)