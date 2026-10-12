# -*- coding: utf-8 -*-
"""Gate: our numpy+Pillow preprocessing vs the processor's own pixel_values.

artifacts/golden_pixels.npz holds the pixel_values and image_grid_thw the
reference's processor produced, so no transformers is needed here -- only Pillow
on top of the runtime dependencies. Geometry must match exactly; pixels are
compared numerically (Pillow's resampling and torchvision's are not bit-equal
when an image has to be resized).
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from onejev_decide.image import ImageConfig, prepare_images  # noqa: E402
from onejev_decide.paths import MODEL_SIZES, default_model_dir  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGES = ROOT / "artifacts" / "images"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="0.8b", choices=MODEL_SIZES,
                        help="0.8b (models/onnx, $ONEJEV_MODEL_DIR) or 4b (models/onnx-4b, $ONEJEV4_MODEL_DIR)")
    parser.add_argument("--model", default=None, help="model directory (overrides --model-size and the environment)")
    args = parser.parse_args()
    golden_file = "artifacts/golden.json" if args.model_size == "0.8b" else "artifacts/golden_4b.json"
    golden = json.loads((ROOT / golden_file).read_text(encoding="utf-8"))
    pixels = np.load(ROOT / "artifacts/golden_pixels.npz")      # the 0.8B and 4B processors give identical pixels
    config = ImageConfig.from_pretrained(default_model_dir(args.model_size, ROOT, args.model))

    failures = 0
    worst = 0.0
    for case in golden["cases"]:
        media = case["request"].get("media") or []
        if not media:
            continue
        ours = prepare_images([str(IMAGES / m["path"]) for m in media], config)
        ref_pixels = pixels["%s__pixel_values" % case["id"]]
        ref_grid = pixels["%s__image_grid_thw" % case["id"]]

        ok_grid = ours.image_grid_thw.tolist() == ref_grid.tolist()
        ok_shape = ours.pixel_values.shape == ref_pixels.shape
        delta = float(np.abs(ours.pixel_values - ref_pixels).max()) if ok_shape else -1.0
        worst = max(worst, delta)
        good = ok_grid and ok_shape and 0 <= delta < 0.02
        failures += not good
        print("%s %-18s grid=%-16s shape=%s  max|dp|=%.4f"
              % ("ok " if good else "XX ", case["id"], ref_grid.tolist(),
                 "OK" if ok_shape else "DIFF", delta))

    print("\nworst |pixel delta|: %.4f (range is [-1, 1])" % worst)
    print("failures: %d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
