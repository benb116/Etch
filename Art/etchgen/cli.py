"""Command-line interface for etchgen image-to-lineset generation."""

import argparse
import sys
from pathlib import Path

from .pipeline import Pipeline
from .serialize import save_lineset_json


def main():
    parser = argparse.ArgumentParser(
        prog="python -m Art.etchgen",
        description="Convert images to single-line Etch-a-Sketch linesets",
    )
    parser.add_argument("image", help="Input image file")
    parser.add_argument("--style", default="sketch", choices=["sketch", "hatch", "pencil", "squiggle"],
                        help="Drawing style (default: sketch)")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--name", help="Lineset name (default: filename)")
    parser.add_argument("--quality", default="final", choices=["preview", "final"],
                        help="Render quality (default: final)")
    parser.add_argument("--set", action="append", default=[],
                        help="Override a parameter: --set key=value (can be repeated)")

    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: {image_path} not found", file=sys.stderr)
        return 1

    # Validate output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse parameter overrides
    param_overrides = {}
    for kv in args.set:
        if "=" not in kv:
            print(f"Error: --set must be key=value, got {kv}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        param_overrides[k] = v

    # Default name
    lineset_name = args.name or image_path.stem

    print(f"Loading {image_path}...")
    pipe = Pipeline()
    pipe.load_image(str(image_path))
    pipe.set_style_params(args.style, param_overrides)

    print(f"Rendering ({args.style}, quality={args.quality})...")
    lineset = pipe.to_lineset(name=lineset_name, quality=args.quality)

    print(f"Saving to {out_path}...")
    save_lineset_json(lineset, str(out_path))

    metrics = pipe.render(quality=args.quality)["metrics"]
    print(f"Done. Points: {metrics['points']}, Ink: {metrics['ink_fraction']:.1%}, "
          f"Draw time: {metrics['draw_seconds']:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
