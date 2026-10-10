"""Procedural sky rendering (clear day / cloudy / evening) as a RGB field."""

from __future__ import annotations

import numpy as np

# Default sun positions (compass azimuth in degrees from north, elevation).
DEFAULT_SUN = {
    "clear": (225.0, 40.0),
    "cloudy": (220.0, 32.0),
    "evening": (262.0, 3.0),
}

PRESETS = {
    "clear": {
        "zenith": (0.10, 0.28, 0.55),
        "horizon": (0.66, 0.76, 0.88),
        "ground": (0.87, 0.91, 0.96),
        "sun": (1.0, 0.98, 0.90),
        "sun_scale": 0.06,
        "clouds": None,
    },
    "cloudy": {
        "zenith": (0.28, 0.40, 0.55),
        "horizon": (0.72, 0.78, 0.86),
        "ground": (0.40, 0.40, 0.40),
        "sun": (1.0, 0.97, 0.88),
        "sun_scale": 0.12,
        "clouds": {"density": 0.6, "scale": 3.0, "octaves": 5, "min_angle": 0.04},
    },
    "evening": {
        "zenith": (0.06, 0.09, 0.20),
        "horizon": (0.92, 0.42, 0.22),
        "ground": (0.16, 0.12, 0.14),
        "sun": (1.0, 0.55, 0.25),
        "sun_scale": 0.09,
        "clouds": {"density": 0.55, "scale": 3.5, "octaves": 5, "min_angle": 0.02},
    },
}

_ALIASES = {
    "blue": "clear",
    "day": "clear",
    "sunset": "evening",
    "dusk": "evening",
    "overcast": "cloudy",
}


def resolve_preset(name: str) -> str:
    name = (name or "clear").strip().lower()
    return _ALIASES.get(name, name)


def resolve_sun(preset: str, sun_azimuth, sun_elevation) -> tuple[float, float]:
    preset = resolve_preset(preset)
    if sun_azimuth is None or sun_elevation is None:
        az0, el0 = DEFAULT_SUN[preset]
        sun_azimuth = az0 if sun_azimuth is None else sun_azimuth
        sun_elevation = el0 if sun_elevation is None else sun_elevation
    return float(sun_azimuth), float(sun_elevation)


def sun_direction(sun_azimuth: float, sun_elevation: float) -> np.ndarray:
    """Unit vector toward the sun in the east/north/up frame used by render."""
    az = np.radians(sun_azimuth)
    el = np.radians(sun_elevation)
    return np.array(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)],
        dtype=np.float64,
    )


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _mix(a, b, t):
    t = np.asarray(t)[..., None]
    return a * (1.0 - t) + b * t


# --------------------------------------------------------------------------- #
# Value noise / FBM (deterministic, patch-free in 3D direction space)
# --------------------------------------------------------------------------- #
def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    s = ix * 127.1 + iy * 311.7 + iz * 74.7
    return (np.sin(s) * 43758.5453123) % 1.0


def _value_noise(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    iz = np.floor(z).astype(np.int64)
    fx = x - np.floor(x)
    fy = y - np.floor(y)
    fz = z - np.floor(z)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)

    c000 = _hash3(ix, iy, iz)
    c100 = _hash3(ix + 1, iy, iz)
    c010 = _hash3(ix, iy + 1, iz)
    c110 = _hash3(ix + 1, iy + 1, iz)
    c001 = _hash3(ix, iy, iz + 1)
    c101 = _hash3(ix + 1, iy, iz + 1)
    c011 = _hash3(ix, iy + 1, iz + 1)
    c111 = _hash3(ix + 1, iy + 1, iz + 1)

    x00 = c000 * (1 - ux) + c100 * ux
    x10 = c010 * (1 - ux) + c110 * ux
    x01 = c001 * (1 - ux) + c101 * ux
    x11 = c011 * (1 - ux) + c111 * ux
    y0 = x00 * (1 - uy) + x10 * uy
    y1 = x01 * (1 - uy) + x11 * uy
    return y0 * (1 - uz) + y1 * uz


def _fbm(x: np.ndarray, y: np.ndarray, z: np.ndarray, octaves: int = 5) -> np.ndarray:
    value = np.zeros_like(x)
    amp = 0.5
    scale = 1.0
    for _ in range(octaves):
        value += amp * _value_noise(x * scale, y * scale, z * scale)
        amp *= 0.5
        scale *= 2.0
    return value  # roughly in [0, 1)


def _cloud_field(theta, phi, params: dict) -> np.ndarray:
    """Cloud coverage in [0, 1) sampled over 3D unit directions."""
    H, W = theta.shape
    dx = np.cos(phi) * np.cos(theta).astype(np.float32)
    dy = np.cos(phi) * np.sin(theta).astype(np.float32)
    dz = np.sin(phi).astype(np.float32)
    s = params["scale"]
    n = _fbm(dx * s, dy * s, dz * s, params["octaves"])
    band = _smoothstep(params["min_angle"], params["min_angle"] * 3.0, phi)
    return n * band


def render_sky(
    theta: np.ndarray,
    phi: np.ndarray,
    preset: str,
    sun_azimuth: float | None = None,
    sun_elevation: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Produce an (H, W, 3) float32 sky image in [0, 1].

    theta: azimuth in radians (0 = +east, +pi/2 = +north).
    phi:   elevation in radians (-pi/2 below, +pi/2 zenith).
    """
    if seed is not None:
        np.random.seed(seed)
    preset = resolve_preset(preset)
    cfg = PRESETS[preset]

    sun_azimuth, sun_elevation = resolve_sun(preset, sun_azimuth, sun_elevation)
    az = np.radians(sun_azimuth)
    el = np.radians(sun_elevation)
    sun_dir = sun_direction(sun_azimuth, sun_elevation)

    dx = np.cos(phi) * np.cos(theta)
    dy = np.cos(phi) * np.sin(theta)
    dz = np.sin(phi)
    cos_ang = np.clip(dx * sun_dir[0] + dy * sun_dir[1] + dz * sun_dir[2], -1.0, 1.0)
    ang = np.arccos(cos_ang)

    zenith = np.array(cfg["zenith"], dtype=np.float32)
    horizon = np.array(cfg["horizon"], dtype=np.float32)
    ground = np.array(cfg["ground"], dtype=np.float32)
    sun_col = np.array(cfg["sun"], dtype=np.float32)

    # Base gradient: horizon colour at phi=0 -> zenith colour at top.
    above = phi > 0.0
    f = np.clip(phi / (np.pi / 2.0), 0.0, 1.0)
    f = f * f * (3.0 - 2.0 * f)  # smooth (C1) so the horizon blends softly
    base = _mix(horizon, zenith, f)

    # Below the horizon: continue the horizon colour into a soft haze.
    b = np.clip(np.abs(phi) / (np.pi / 2.0), 0.0, 1.0)
    below_factor = b * b * (3.0 - 2.0 * b)
    base_below = _mix(horizon, ground, below_factor)
    base = np.where(above[..., None], base, base_below)

    sky = base

    # Evening warmth near the sun azimuth along the horizon.
    if preset == "evening":
        warm = np.array([1.0, 0.35, 0.15], dtype=np.float32)
        azim_diff = np.abs(np.angle(np.exp(1j * (theta - az))))
        warmness = (
            np.exp(-(azim_diff / 0.6) ** 2)
            * np.exp(-(np.clip(phi, 0.0, None) / 0.25) ** 2)
        )
        sky = sky + warm * warmness[..., None] * 0.4

    # Sun disc + glow.
    disc = (ang < 0.009).astype(np.float32)
    glow = np.exp(-(ang / cfg["sun_scale"]) ** 2)
    sky = sky + (disc[..., None] + glow[..., None] * 0.65) * sun_col

    # Clouds.
    clouds = cfg["clouds"]
    if clouds is not None:
        coverage = _cloud_field(theta, phi, clouds)
        cloud_mask = _smoothstep(1.0 - clouds["density"], 0.85, coverage)
        shade = np.clip(cos_ang * 0.5 + 0.5, 0.0, 1.0)
        cloud_col = 0.55 + 0.45 * shade[..., None]
        sky = sky * (1.0 - cloud_mask[..., None]) + (
            sky * cloud_col
        ) * cloud_mask[..., None]

    return np.clip(sky, 0.0, 1.0).astype(np.float32)