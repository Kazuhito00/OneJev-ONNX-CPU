# -*- coding: utf-8 -*-
"""Image decisions with OneJev, on onnxruntime + numpy + Pillow.

    python demo_inference_image.py                        # built-in example
    python demo_inference_image.py --image shot.png \
        --choice "status=paid,pending,failed"

Images are referenced from the state as <image:1>, <image:2>, ... once each, in
order. The image is encoded by a separate ONNX graph and its features are
scattered into the token embeddings; the decision is then read exactly as for
text.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onejev_decide import Decider, GpuUnavailableError, add_device_args, choice, noul, providers_from_args, score  # noqa: E402
from onejev_decide.devices import exit_on_gpu_error, report  # noqa: E402
from onejev_decide.paths import MODEL_SIZES, default_model_dir  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
SAMPLES = ROOT / "artifacts" / "images"


def built_in_example():
    """A picture we can draw here, so the demo runs with no assets."""
    from PIL import Image, ImageDraw

    SAMPLES.mkdir(parents=True, exist_ok=True)
    path = SAMPLES / "demo_circle.png"
    if not path.exists():
        canvas = Image.new("RGB", (224, 224), "white")
        ImageDraw.Draw(canvas).ellipse((40, 40, 184, 184), fill=(200, 40, 40))
        canvas.save(path)
    state = {"photo": "<image:1>", "context": "a drawing uploaded by a user"}
    questions = {
        "shape": choice("What shape is drawn?", {"circle": "a circle", "square": "a square"}),
        "colour": choice("What colour is the shape?",
                         {"red": "mostly red", "green": "mostly green", "blue": "mostly blue"}),
        "drawing": noul("The picture contains a drawn shape"),
    }
    return state, questions, [str(path)]


def parse_questions(args):
    questions = {}
    for spec in args.choice or []:
        name, _, rest = spec.partition("=")
        options = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = choice(args.instructions or name.strip(),
                                         {o: o for o in options})
    for spec in args.score or []:
        name, _, rest = spec.partition("=")
        levels = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = score(args.instructions or name.strip(), levels)
    for spec in args.noul or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = noul(rest.strip() or name.strip())
    return questions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="append", default=[], metavar="PATH")
    parser.add_argument("--state", default=None,
                        help="text referring to the images as <image:1>, <image:2>, ...")
    parser.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    parser.add_argument("--score", action="append", metavar="NAME=low,...,high")
    parser.add_argument("--noul", action="append", metavar="NAME=statement")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--model-size", default="0.8b", choices=MODEL_SIZES,
                        help="0.8b (models/onnx, $ONEJEV_MODEL_DIR) or 4b (models/onnx-4b, $ONEJEV4_MODEL_DIR)")
    parser.add_argument("--model", default=None, help="model directory (overrides --model-size and the environment)")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--threads", type=int, default=8)
    add_device_args(parser)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.image:
        state, questions, images = built_in_example()
    else:
        images = list(args.image)
        # By default each image is named in order; --state can say more.
        state = args.state or {"images": ["<image:%d>" % (i + 1) for i in range(len(images))]}
        questions = parse_questions(args)
        if not questions:
            parser.error("give at least one --choice, --score or --noul")

    providers, require_gpu = providers_from_args(args)
    started = time.perf_counter()
    try:
        decider = Decider(default_model_dir(args.model_size, ROOT, args.model), variant=args.variant, intra_op_threads=args.threads,
                          providers=providers, require_gpu=require_gpu)
    except GpuUnavailableError as exc:
        exit_on_gpu_error(exc)
    load = time.perf_counter() - started

    started = time.perf_counter()
    answers = decider.decide(state, questions, images=images)
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0

    print("images: %s\n" % ", ".join(pathlib.Path(p).name for p in images))
    for name, answer in answers.items():
        if answer["type"] == "noul":
            print("%-10s noul   p(yes)=%.3f" % (name, answer["noul"]))
            continue
        spread = "  ".join(
            "%s=%.3f" % kv
            for kv in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])
        )
        head = answer["choice"] if answer["type"] == "choice" else "%.2f" % answer["score"]
        print("%-10s %-6s %-20s confidence %.3f\n           %s"
              % (name, answer["type"], head, answer["confidence"], spread))
    print("\n%d questions, one prefill each" % len(answers))
    print("load %.1f s, inference %.2f s (%s)" % (load, elapsed, decider.provider))
    report("providers actually used", decider.providers_used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
