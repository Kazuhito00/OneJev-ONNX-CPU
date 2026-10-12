# -*- coding: utf-8 -*-
"""Does the model answer from the content, or from where the option sits?

The same question is asked with the options in both orders. A content-driven
model gives the same answer; a position-biased one follows the position. The
0.8B Qwen3.5 decision models in this family differ a lot here, so it is measured
rather than assumed.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from onejev_decide import Decider, choice  # noqa: E402
from onejev_decide.paths import MODEL_SIZES, default_model_dir  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

CASES = [
    ("Thanks so much for the quick fix yesterday, everything works great now!",
     ("the customer is angry", "the customer is calm"), "calm"),
    ("This is the third time this week my payouts failed and nobody answers. Unacceptable.",
     ("the customer is angry", "the customer is calm"), "angry"),
    ("The deploy finished, all health checks are green and latency is back to normal.",
     ("there is an ongoing incident", "everything is healthy"), "healthy"),
    ("Buy cheap watches now!!! Click http://spam.example.com for 90% off!!!",
     ("this message is spam", "this message is legitimate"), "spam"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="0.8b", choices=MODEL_SIZES,
                        help="0.8b (models/onnx, $ONEJEV_MODEL_DIR) or 4b (models/onnx-4b, $ONEJEV4_MODEL_DIR)")
    parser.add_argument("--model", default=None, help="model directory (overrides --model-size and the environment)")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    decider = Decider(default_model_dir(args.model_size, ROOT, args.model), variant=args.variant,
                      intra_op_threads=args.threads, load_vision=False)
    print("variant=%s" % args.variant)
    stable = correct = 0
    for state, (first, second), expected in CASES:
        forward = decider.decide(state, {"q": choice("Which statement holds?",
                                                     {"a": first, "b": second})})["q"]
        reverse = decider.decide(state, {"q": choice("Which statement holds?",
                                                     {"b": second, "a": first})})["q"]
        # same two options, opposite order; compare by which statement won
        fwd_pick = first if forward["choice"] == "a" else second
        rev_pick = first if reverse["choice"] == "a" else second
        agree = fwd_pick == rev_pick
        right = agree and expected in fwd_pick
        stable += agree
        correct += right
        print("%s %-52s  A-first -> %-26s | B-first -> %-26s"
              % ("ok " if right else ("~~ " if agree else "XX "), state[:50],
                 fwd_pick[:24], rev_pick[:24]))
        print("      p(first listed) fwd=%.3f rev=%.3f"
              % (forward["probabilities"]["a"], reverse["probabilities"]["b"]))

    print("\norder-stable: %d/%d   and correct: %d/%d"
          % (stable, len(CASES), correct, len(CASES)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
