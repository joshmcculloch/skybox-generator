"""Command-line interface for the skybox generator."""

from __future__ import annotations

import argparse

from . import pipeline, sky


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skybox",
        description="Render an equirectangular skybox from satellite imagery "
        "and SRTM elevation at any location on Earth.",
    )
    p.add_argument("--lat", type=float, default=46.0216, help="centre latitude")
    p.add_argument("--lon", type=float, default=7.7480, help="centre longitude")
    p.add_argument("--sky", default="clear", help="sky preset: clear, cloudy, evening")
    p.add_argument("-o", "--out", default="skybox.png", help="output PNG path")
    p.add_argument("--width", type=int, default=2048, help="panorama width (2:1)")
    p.add_argument("--radius-km", type=float, default=60.0, help="terrain surround radius")
    p.add_argument("--terrain-res", type=int, default=1024, help="terrain grid size")
    p.add_argument("--eye-height", type=float, default=2.0, help="metres above ground")
    p.add_argument("--heading", type=float, default=0.0, help="panorama centre heading (deg)")
    p.add_argument("--sun-azimuth", type=float, default=None, help="deg from north")
    p.add_argument("--sun-elevation", type=float, default=None, help="deg above horizon")
    p.add_argument("--zoom", type=int, default=None, help="imagery zoom override")
    p.add_argument("--cache", default=None, help="cache directory")
    p.add_argument("--seed", type=int, default=None, help="cloud/sky RNG seed")
    p.add_argument(
        "--n-steps",
        type=int,
        default=192,
        help="ray-march steps (log-spaced; lowers cost, bisection refines)",
    )
    p.add_argument(
        "--visibility",
        type=float,
        default=None,
        help="haze e-folding distance in metres (default: radius/3)",
    )
    p.add_argument(
        "--ambient",
        type=float,
        default=0.4,
        help="terrain ambient light (0..1); shadows = ambient, sun-facing = 1",
    )
    p.add_argument(
        "--lod-count",
        type=int,
        default=8,
        help="terrain LOD levels; each halves the radius and doubles detail",
    )
    p.add_argument(
        "--list-skies", action="store_true", help="list sky presets and exit"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_skies:
        print("Available sky presets:")
        for name in sorted(sky.PRESETS):
            print(f"  {name}")
        print("Aliases:", ", ".join(sorted(sky._ALIASES)))
        return 0

    preset = sky.resolve_preset(args.sky)
    if preset not in sky.PRESETS:
        print(
            f"Unknown sky preset '{args.sky}'. "
            f"Choose from: {', '.join(sorted(sky.PRESETS))}"
        )
        return 2

    print(
        f"Skybox @ ({args.lat:.4f}, {args.lon:.4f}), sky={preset}, "
        f"radius={args.radius_km}km, {args.width}x{args.width // 2}"
    )
    pipeline.generate(
        lat=args.lat,
        lon=args.lon,
        sky_name=preset,
        out=args.out,
        width=args.width,
        radius_km=args.radius_km,
        terrain_res=args.terrain_res,
        eye_height=args.eye_height,
        heading_deg=args.heading,
        sun_azimuth=args.sun_azimuth,
        sun_elevation=args.sun_elevation,
        zoom=args.zoom,
        cache_dir=args.cache,
        seed=args.seed,
        n_steps=args.n_steps,
        visibility=args.visibility,
        ambient=args.ambient,
        lod_count=args.lod_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())