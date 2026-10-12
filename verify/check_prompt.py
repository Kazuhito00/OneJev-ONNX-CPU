# -*- coding: utf-8 -*-
"""Gate: our prompt string and input_ids vs the reference implementation's.

artifacts/golden.json carries, for every question, the prompt text the reference
rendered and the token ids it ran (vision blocks expanded). Matching both means
the model is being fed exactly what it was trained on. Runs on the runtime
dependencies alone -- plus Pillow, because image cases need the visual token
count.
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402

from onejev_decide import Decider  # noqa: E402
from onejev_decide.paths import MODEL_SIZES, default_model_dir  # noqa: E402
from onejev_decide.protocol import question_from_spec, render_question  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGES = ROOT / "artifacts" / "images"


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="0.8b", choices=MODEL_SIZES,
                        help="0.8b (models/onnx, $ONEJEV_MODEL_DIR) or 4b (models/onnx-4b, $ONEJEV4_MODEL_DIR)")
    parser.add_argument("--model", default=None, help="model directory (overrides --model-size and the environment)")
    args = parser.parse_args()
    golden_file = "artifacts/golden.json" if args.model_size == "0.8b" else "artifacts/golden_4b.json"
    golden = json.loads((ROOT / golden_file).read_text(encoding="utf-8"))
    # the image cases need the vision geometry (preprocessor_config.json), not the vision encoder graph
    decider = Decider(default_model_dir(args.model_size, ROOT, args.model), intra_op_threads=1, load_vision=False)

    failures = 0
    for case in golden["cases"]:
        request = case["request"]
        media = request.get("media") or []
        counts = []
        if media:
            from onejev_decide.image import prepare_images

            prepared = prepare_images([str(IMAGES / m["path"]) for m in media],
                                      decider.image_config)
            counts = prepared.token_counts

        problems = []
        for branch in case["branches"]:
            spec = request["questions"][branch["qid"]]
            rendered = render_question(question_from_spec(spec))
            prompt = decider.render_prompt(request["state"], rendered, counts)
            ids = decider.tokenizer.encode(prompt)

            # The stored prompt is what apply_chat_template emits: one
            # <|image_pad|> per image, which the processor expands while
            # tokenising. We expand up front, so expand the reference too.
            # One block per image, in order. Replace by position, not by
            # "first match": after the first expansion the first match would
            # land inside the pads just written.
            blocks = iter(counts)
            expected = re.sub(
                r"<\|image_pad\|>",
                lambda _match: "<|image_pad|>" * next(blocks),
                branch["prompt"],
            )

            if prompt != expected:
                k = first_diff(prompt, expected)
                problems.append("%s: prompt differs at %d: ours=%r ref=%r"
                                % (branch["qid"], k, prompt[max(0, k - 30):k + 30],
                                   expected[max(0, k - 30):k + 30]))
            if ids != branch["input_ids"]:
                k = first_diff(ids, branch["input_ids"])
                problems.append("%s: input_ids differ at %d (len ours=%d ref=%d)"
                                % (branch["qid"], k, len(ids), len(branch["input_ids"])))

        failures += bool(problems)
        print("%s %-18s %d questions%s"
              % ("ok " if not problems else "XX ", case["id"], len(case["branches"]),
                 "  (%d image%s)" % (len(media), "s" if len(media) != 1 else "") if media else ""))
        for problem in problems:
            print("      %s" % problem)

    print("\nfailures: %d / %d" % (failures, len(golden["cases"])))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
