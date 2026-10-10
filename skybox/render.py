"""Equirectangular terrain renderer: heightfield ray-marching in numpy."""

from __future__ import annotations

import numpy as np


def build_directions(width: int, heading_deg: float = 0.0):
    """Return (theta, phi, dx, dy, dz) fields for an equirectangular image.

    theta: azimuth in radians (0 = +east, +pi/2 = +north).
    phi:   elevation in radians.
    dx/dy/dz are unit direction components for every (row, col).

    Columns sweep clockwise viewed from above (left edge = behind the camera,
    right edge = to the camera's right), so a due-east landmark at heading 0
    lands on the right of the frame. `heading_deg` is a compass bearing from
    north (0 = north, 90 = east).
    """
    H = width // 2
    W = width
    u = (np.arange(W, dtype=np.float64) + 0.5) / W
    bearing = np.radians(heading_deg) + (u - 0.5) * 2.0 * np.pi
    theta = 0.5 * np.pi - bearing
    v = (np.arange(H, dtype=np.float64) + 0.5) / H
    phi = (0.5 - v) * np.pi

    TH, PH = np.meshgrid(theta, phi)
    dx = (np.cos(PH) * np.cos(TH)).astype(np.float32)
    dy = (np.cos(PH) * np.sin(TH)).astype(np.float32)
    dz = np.sin(PH).astype(np.float32)
    return theta, phi, dx, dy, dz


class Terrain:
    """Height + colour fields on a regular local east/north grid."""

    def __init__(self, heights: np.ndarray, colors: np.ndarray, radius: float):
        """
        heights: (Ny, Nx) float array, metres (absolute elevation)
        colors:  (Ny, Nx, 3) float array in [0, 1]
        radius:  half-extent of the grid in metres (grid spans [-radius, radius])
        """
        self.heights = np.asarray(heights, dtype=np.float32)
        self.colors = np.asarray(colors, dtype=np.float32)
        self.Ny, self.Nx = self.heights.shape
        self.radius = float(radius)
        axis = np.linspace(-radius, radius, self.Nx, dtype=np.float64)
        self.east0 = axis[0]
        self.north0 = axis[0]
        self.dx = axis[1] - axis[0]
        self.dy = self.dx  # square, isotropic grid
        self.color_ndim = self.colors.ndim == 3
        self.grad_e, self.grad_n = self._gradients()

    def _gradients(self):
        """dh/d(east) and dh/d(north) via central differences of the heightfield."""
        h = self.heights
        ge = np.zeros_like(h)
        gn = np.zeros_like(h)
        if self.Nx > 1:
            ge[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * (0.5 / self.dx)
            ge[:, 0] = (h[:, 1] - h[:, 0]) / self.dx
            ge[:, -1] = (h[:, -1] - h[:, -2]) / self.dx
        if self.Ny > 1:
            gn[1:-1, :] = (h[2:, :] - h[:-2, :]) * (0.5 / self.dy)
            gn[0, :] = (h[1, :] - h[0, :]) / self.dy
            gn[-1, :] = (h[-1, :] - h[-2, :]) / self.dy
        return ge.astype(np.float32), gn.astype(np.float32)

    def _indices(self, east, north):
        xf = (east - self.east0) / self.dx
        yf = (north - self.north0) / self.dy
        x0 = np.floor(xf).astype(np.int64)
        y0 = np.floor(yf).astype(np.int64)
        tx = (xf - x0).astype(np.float32)
        ty = (yf - y0).astype(np.float32)
        x1 = x0 + 1
        y1 = y0 + 1
        valid = (x0 >= 0) & (x1 < self.Nx) & (y0 >= 0) & (y1 < self.Ny)
        x0c = np.clip(x0, 0, self.Nx - 1)
        x1c = np.clip(x1, 0, self.Nx - 1)
        y0c = np.clip(y0, 0, self.Ny - 1)
        y1c = np.clip(y1, 0, self.Ny - 1)
        return (y0c, x0c), (y0c, x1c), (y1c, x0c), (y1c, x1c), tx, ty, valid

    def sample_height(self, east, north):
        i00, i10, i01, i11, tx, ty, valid = self._indices(east, north)
        # i00=(y0,x0) i10=(y0,x1) i01=(y1,x0) i11=(y1,x1)
        h = (
            self.heights[i00] * (1 - tx) * (1 - ty)
            + self.heights[i10] * tx * (1 - ty)
            + self.heights[i01] * (1 - tx) * ty
            + self.heights[i11] * tx * ty
        )
        h = h.astype(np.float32)
        h[~valid] = np.nan
        return h

    def sample_color(self, east, north):
        i00, i10, i01, i11, tx, ty, valid = self._indices(east, north)
        tx = tx[..., None]
        ty = ty[..., None]
        c = (
            self.colors[i00] * (1 - tx) * (1 - ty)
            + self.colors[i10] * tx * (1 - ty)
            + self.colors[i01] * (1 - tx) * ty
            + self.colors[i11] * tx * ty
        )
        c[~valid] = 0.0
        return c

    def sample_normal(self, east, north):
        """Unit surface normal (len-3 last axis) from the heightfield gradient."""
        i00, i10, i01, i11, tx, ty, valid = self._indices(east, north)
        ge = (
            self.grad_e[i00] * (1 - tx) * (1 - ty)
            + self.grad_e[i10] * tx * (1 - ty)
            + self.grad_e[i01] * (1 - tx) * ty
            + self.grad_e[i11] * tx * ty
        )
        gn = (
            self.grad_n[i00] * (1 - tx) * (1 - ty)
            + self.grad_n[i10] * tx * (1 - ty)
            + self.grad_n[i01] * (1 - tx) * ty
            + self.grad_n[i11] * tx * ty
        )
        nx = -ge
        ny = -gn
        nz = np.ones_like(ge)
        length = np.sqrt(nx * nx + ny * ny + nz * nz)
        return np.stack([nx / length, ny / length, nz / length], axis=-1)


class TerrainLOD:
    """Hierarchical terrain: coarse at the horizon, fine near the camera.

    ``levels[k]`` is a :class:`Terrain` covering ``radius / 2**k`` with the
    same grid resolution as the base level, so the sample density doubles each
    level.  At render time each sample point is routed to the finest level
    whose extent still contains it.
    """

    def __init__(self, levels: list[Terrain]):
        if not levels:
            raise ValueError("TerrainLOD requires at least one level")
        self.levels = list(levels)
        self.radius = self.levels[0].radius
        self.max_level = len(self.levels) - 1

    def _split(self, east, north):
        east, north = np.broadcast_arrays(
            np.asarray(east, dtype=np.float32), np.asarray(north, dtype=np.float32)
        )
        rho = np.hypot(east, north)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.floor(np.log2(self.radius / np.maximum(rho, 1e-9)))
        level = np.clip(raw, 0, self.max_level).astype(np.int64)
        return east, north, level

    def _gather(self, east, north, level, sample, out):
        for L, t in enumerate(self.levels):
            mask = level == L
            if mask.any():
                out[mask] = sample(t, east[mask], north[mask])
        return out

    def sample_height(self, east, north):
        east, north, level = self._split(east, north)
        out = np.full(east.shape, np.nan, dtype=np.float32)
        return self._gather(east, north, level, Terrain.sample_height, out)

    def sample_color(self, east, north):
        east, north, level = self._split(east, north)
        out = np.zeros(east.shape + (3,), dtype=np.float32)
        return self._gather(east, north, level, Terrain.sample_color, out)

    def sample_normal(self, east, north):
        east, north, level = self._split(east, north)
        out = np.zeros(east.shape + (3,), dtype=np.float32)
        return self._gather(east, north, level, Terrain.sample_normal, out)


def raymarch(terrain: Terrain, dx, dy, dz, cam_z, n_steps=192, bisect=6):
    """Return (t_hit, has_hit) per pixel.

    t_hit is the distance to the first terrain intersection (inf when the ray
    escapes the grid).
    """
    shape = dx.shape
    lo = np.zeros(shape, dtype=np.float32)
    hi = np.full(shape, np.inf, dtype=np.float32)
    alive = np.ones(shape, dtype=bool)

    diag = np.hypot(terrain.radius, terrain.radius)
    t_min = 0.2
    t_max = 1.6 * diag
    for t in np.geomspace(t_min, t_max, n_steps, dtype=np.float64):
        tf = t.astype(np.float32)
        px = dx * tf
        py = dy * tf
        pz = cam_z + dz * tf
        h = terrain.sample_height(px, py)
        hit = alive & (pz < h)
        if np.any(hit):
            hi[hit] = tf
            alive[hit] = False
        lo[alive] = tf
        if not np.any(alive):
            break

    has_hit = np.isfinite(hi)

    # Bisection refinement only on rays that actually intersected.
    idx = np.nonzero(has_hit.ravel())[0]
    if idx.size:
        lo_f = lo.ravel()[idx].astype(np.float32)
        hi_f = hi.ravel()[idx].astype(np.float32)
        dxf = dx.ravel()[idx]
        dyf = dy.ravel()[idx]
        dzf = dz.ravel()[idx]
        for _ in range(bisect):
            mid = 0.5 * (lo_f + hi_f)
            h = terrain.sample_height(dxf * mid, dyf * mid)
            below = (cam_z + dzf * mid) < h
            hi_f = np.where(below, mid, hi_f)
            lo_f = np.where(below, lo_f, mid)
        hi.ravel()[idx] = hi_f

    return hi, has_hit


def render_equirect(
    terrain: Terrain,
    sky_rgb: np.ndarray,
    cam_z: float,
    width: int = 2048,
    heading_deg: float = 0.0,
    n_steps: int = 192,
    visibility: float | None = None,
    sun_dir: np.ndarray | None = None,
    ambient: float = 0.4,
):
    """Render a full equirectangular panorama (H, W, 3) uint8.

    When ``sun_dir`` is provided, terrain is shaded with a diffuse Lambertian
    term (``ambient + (1 - ambient) * max(dot(normal, sun), 0)``); no
    self-shadowing is computed.
    """
    _, _, dx, dy, dz = build_directions(width, heading_deg)
    H, W = sky_rgb.shape[0], sky_rgb.shape[1]
    assert (H, W) == dx.shape, "sky_rgb dimensions must match width"

    t_hit, has_hit = raymarch(terrain, dx, dy, dz, cam_z, n_steps=n_steps)

    out = np.array(sky_rgb, dtype=np.float32)

    if np.any(has_hit):
        px = np.where(has_hit, dx * t_hit, np.float32(0.0))
        py = np.where(has_hit, dy * t_hit, np.float32(0.0))
        color = terrain.sample_color(px, py)
        if sun_dir is not None:
            normal = terrain.sample_normal(px, py)
            cos_sun = np.clip(np.einsum("...i,i->...", normal, sun_dir), 0.0, None)
            light = ambient + (1.0 - ambient) * cos_sun
            color = color * light[..., None]
        vis = visibility if visibility else terrain.radius / 3.0
        transmittance = np.exp(-t_hit / vis)
        haze = 1.0 - transmittance
        blended = color * transmittance[..., None] + sky_rgb * haze[..., None]
        out = np.where(has_hit[..., None], blended, out)

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)