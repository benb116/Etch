#!/usr/bin/env python
"""Benchmarking and profiling for the etchgen pipeline.

Tests rendering performance across styles and image sizes to identify
bottlenecks and verify performance targets.
"""

import sys
import time
from pathlib import Path

import numpy as np

from .pipeline import Pipeline


def bench_image(image_path, styles=None, qualities=None):
    """Benchmark rendering an image across styles and qualities.

    Args:
        image_path: path to image file
        styles: list of style names (default: all)
        qualities: list of 'preview' or 'final' (default: ['preview'])

    Prints: table of style × quality × timing
    """
    if styles is None:
        styles = ['sketch', 'hatch', 'pencil', 'squiggle']
    if qualities is None:
        qualities = ['preview']

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    print(f"\n{'='*60}")
    print(f"Benchmarking: {image_path.name}")
    print(f"{'='*60}\n")

    pipe = Pipeline()
    pipe.load_image(str(image_path))

    results = []

    for style in styles:
        for quality in qualities:
            try:
                print(f"Testing {style:10s} ({quality:7s})...", end=" ", flush=True)

                pipe.set_style_params(style)
                t0 = time.time()
                result = pipe.render(quality=quality)
                elapsed = time.time() - t0

                metrics = result["metrics"]
                timings = result["stage_timings"]

                print(f"{elapsed:6.2f}s  ({metrics['points']:5d} pts, "
                      f"{metrics['ink_fraction']*100:5.1f}% ink)")

                results.append({
                    'style': style,
                    'quality': quality,
                    'elapsed': elapsed,
                    'points': metrics['points'],
                    'ink_fraction': metrics['ink_fraction'],
                    'timings': timings,
                })

            except Exception as e:
                print(f"FAILED: {e}")

    print(f"\n{'='*60}")
    print("Summary by stage:")
    print(f"{'='*60}\n")

    for result in results:
        print(f"{result['style']:10s} ({result['quality']:7s}):")
        for stage, t in result['timings'].items():
            pct = 100 * t / result['elapsed']
            print(f"  {stage:12s}: {t:6.3f}s ({pct:5.1f}%)")
        print()


def bench_sizes(image_path, styles=None):
    """Benchmark different working resolutions.

    Args:
        image_path: path to image file
        styles: list of style names (default: ['sketch'])
    """
    if styles is None:
        styles = ['sketch']

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    details = [320, 640, 1024, 1280, 1920]

    print(f"\n{'='*60}")
    print(f"Scaling benchmark: {image_path.name}")
    print(f"{'='*60}\n")

    pipe = Pipeline()
    pipe.load_image(str(image_path))

    for style in styles:
        print(f"\nStyle: {style}")
        print(f"{'Detail':>8s}  {'Time':>8s}  {'Points':>8s}  {'Ink %':>8s}")
        print("-" * 40)

        for detail in details:
            try:
                pipe.set_style_params(style, {'detail': detail})
                t0 = time.time()
                result = pipe.render(quality='preview')
                elapsed = time.time() - t0

                metrics = result["metrics"]
                print(f"{detail:8d}  {elapsed:8.2f}s  {metrics['points']:8d}  "
                      f"{metrics['ink_fraction']*100:7.1f}%")

            except Exception as e:
                print(f"{detail:8d}  ERROR: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m Art.etchgen.bench <image> [--size-bench] [--styles=sketch,hatch,...]")
        print("       python -m Art.etchgen.bench Art/Pics/Mario.jpg")
        print("       python -m Art.etchgen.bench Art/Pics/Mario.jpg --size-bench")
        return 1

    image_path = sys.argv[1]
    size_bench = '--size-bench' in sys.argv

    # Parse styles argument
    styles = None
    for arg in sys.argv[2:]:
        if arg.startswith('--styles='):
            styles = arg.split('=', 1)[1].split(',')

    if size_bench:
        bench_sizes(image_path, styles)
    else:
        bench_image(image_path, styles)

    return 0


if __name__ == "__main__":
    sys.exit(main())
