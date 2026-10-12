# -*- coding: utf-8 -*-
"""Gate: ONNX option logits and typed answers vs the PyTorch reference.

artifacts/golden.json holds, per question, the raw option-slot logits the
reference (qev, fp32, CPU) produced and the answer it built from them. This runs
the ONNX graphs on the same requests and compares both stages.

fp16 and q4f16 are not bit-identical to fp32, so logits are compared with a
tolerance and the decision must agree; --variant fp32 should be tight.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from onejev_decide import Decider  # noqa: E402
from onejev_decide.paths import MODEL_SIZES, default_model_dir  # noqa: E402
from onejev_decide.protocol import (  # noqa: E402
    make_answer,
    question_from_spec,
    render_question,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGES = ROOT / "artifacts" / "images"

#: per-variant tolerance on the slot logits, and on the final probabilities
TOLERANCE = {"fp32": (5e-3, 5e-3), "fp16": (0.25, 0.08), "q4f16": (1.5, 0.35)}


def pick(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return "yes" if answer["noul"] >= 0.5 else "no"
    probs = answer["probabilities"]
    return max(probs, key=probs.get)


def margin(answer):
    """How far the reference is from a tie: top-1 minus top-2 probability
    (for noul, distance from 0.5 times two, which is the same quantity)."""
    if answer["type"] == "noul":
        return abs(2 * answer["noul"] - 1)
    ranked = sorted(answer["probabilities"].values(), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="0.8b", choices=MODEL_SIZES,
                        help="0.8b (models/onnx, $ONEJEV_MODEL_DIR) or 4b (models/onnx-4b, $ONEJEV4_MODEL_DIR)")
    parser.add_argument("--model", default=None, help="model directory (overrides --model-size and the environment)")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--golden", default=None, help="default: artifacts/golden.json (0.8b) or golden_4b.json (4b)")
    args = parser.parse_args()

    logit_tol, prob_tol = TOLERANCE[args.variant]
    golden_path = args.golden or ("artifacts/golden.json" if args.model_size == "0.8b" else "artifacts/golden_4b.json")
    golden = json.loads((ROOT / golden_path).read_text(encoding="utf-8"))
    decider = Decider(default_model_dir(args.model_size, ROOT, args.model), variant=args.variant,
                      intra_op_threads=args.threads, media_root=IMAGES)

    worst_logit = worst_prob = 0.0
    agreed = total = failures = ties = 0

    for case in golden["cases"]:
        request = case["request"]
        media = request.get("media") or []
        images = [m["path"] for m in media]

        prepared = None
        counts = []
        if images:
            from onejev_decide.image import prepare_images

            prepared = prepare_images(decider._resolve_images(images), decider.image_config)
            counts = prepared.token_counts

        bad = False
        lines = []
        for branch in case["branches"]:
            question = question_from_spec(request["questions"][branch["qid"]])
            rendered = render_question(question)
            prompt = decider.render_prompt(request["state"], rendered, counts)
            logits = decider.last_logits(decider.tokenizer.encode(prompt), prepared)
            slots = logits[np.asarray(decider.slot_ids[: rendered.n_slots])]

            ref_slots = np.asarray(branch["slot_logits"])
            # softmax is shift-invariant, so compare logits up to a constant
            d_logit = float(np.abs((slots - slots.mean()) - (ref_slots - ref_slots.mean())).max())

            probs = np.exp(slots - slots.max())
            probs /= probs.sum()
            ours = make_answer(rendered, probs.tolist())
            expected = case["answers"][branch["qid"]]
            if ours["type"] == "noul":
                d_prob = abs(ours["noul"] - expected["noul"])
            else:
                d_prob = max(abs(ours["probabilities"][k] - expected["probabilities"][k])
                             for k in expected["probabilities"])

            worst_logit = max(worst_logit, d_logit)
            worst_prob = max(worst_prob, d_prob)
            same = pick(ours) == pick(expected)
            # A different label is only a real disagreement if the reference was
            # not already within the numeric tolerance of a tie: at p=0.498 vs
            # 0.502 the label is decided by rounding, not by the model.
            near_tie = (not same) and margin(expected) <= 2 * prob_tol
            total += 1
            agreed += same or near_tie
            ties += near_tie
            ok = (same or near_tie) and d_logit < logit_tol and d_prob < prob_tol
            bad |= not ok
            lines.append("      %-9s %s ours=%-12s ref=%-12s max|dlogit|=%.3f max|dp|=%.4f%s"
                         % (branch["qid"][:9], "ok " if ok else "XX ", pick(ours),
                            pick(expected), d_logit, d_prob,
                            "  (near-tie: reference margin %.3f)" % margin(expected)
                            if near_tie else ""))
        failures += bad
        print("%s %-18s" % ("ok " if not bad else "XX ", case["id"]))
        for line in lines:
            print(line)

    print("\nvariant=%s  decisions %d/%d agree   worst |dlogit| %.3f   worst |dp| %.4f"
          % (args.variant, agreed, total, worst_logit, worst_prob))
    print("failures: %d / %d" % (failures, len(golden["cases"])))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
