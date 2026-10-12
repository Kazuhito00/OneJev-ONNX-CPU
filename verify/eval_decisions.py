# -*- coding: utf-8 -*-
"""Twelve decisions a competent reader agrees on.

Not a benchmark: a low score means either the prompt contract is wrong or the
variant is too degraded to use. The same set is used in the sibling repositories
so the models can be compared like for like.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from onejev_decide import Decider, choice, noul, score  # noqa: E402
from onejev_decide.paths import MODEL_SIZES, default_model_dir  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

TEAM = {"billing": "charges, refunds, invoices",
        "technical support": "bugs, outages, errors",
        "sales": "pricing, upgrades, new contracts"}
INTENT = {"cancel_subscription": "wants to cancel the subscription",
          "order_status": "asking where an order is",
          "login_problem": "cannot sign in",
          "bug_report": "reporting a defect"}
PRIORITY = ["low", "medium", "high", "critical"]

CASES = [
    ("Thanks so much for the quick fix yesterday, everything works great now!",
     noul("The customer is angry", "the customer is angry", "the customer is calm"), "no"),
    ("This is the third time this week my payouts failed and nobody answers my emails. Unacceptable.",
     noul("The customer is angry", "the customer is angry", "the customer is calm"), "yes"),
    ("Hi, I was charged twice for my Pro subscription this month, please fix.",
     choice("Which team should handle this?", TEAM), "billing"),
    ("The app crashes every time I open the settings screen on Android 14.",
     choice("Which team should handle this?", TEAM), "technical support"),
    ("URGENT: production database is down, all customers affected, CEO asking for updates every 5 minutes.",
     score("How urgent is this?", PRIORITY), "3"),
    ("Minor typo on the pricing page footer, says 'recieve' instead of 'receive'.",
     score("How urgent is this?", PRIORITY), "0"),
    ("The export button crashes in Safari but works in Chrome. Not blocking, we told users to switch browsers.",
     choice("How severe is this bug?", {"cosmetic": "cosmetic only, nothing is broken",
                                        "degraded": "degraded but there is a workaround",
                                        "blocking": "blocking, no workaround exists"}), "degraded"),
    ("Buy cheap watches now!!! Click http://spam.example.com for 90% off!!!",
     noul("This message is spam", "this message is spam", "this message is legitimate"), "yes"),
    ("Can you send me the invoice for March? Also my login stopped working on the mobile app.",
     score("How many separate requests are in this message?",
           ["one request", "two requests", "three or more requests"]), "1"),
    ("I would like to cancel my subscription and get a refund for this month.",
     choice("What is the customer's intent?", INTENT), "cancel_subscription"),
    ("My package was supposed to arrive on Tuesday and the tracking has not moved since Friday.",
     choice("What is the customer's intent?", INTENT), "order_status"),
    ("The deploy finished, all health checks are green and latency is back to normal.",
     noul("There is an ongoing incident", "there is an ongoing incident", "everything is healthy"), "no"),
]


def decision(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return "yes" if answer["noul"] >= 0.5 else "no"
    probs = answer["probabilities"]
    return max(probs, key=probs.get)


def confidence_of(answer):
    if answer["type"] == "noul":
        return max(answer["noul"], 1 - answer["noul"])
    return max(answer["probabilities"].values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="0.8b", choices=MODEL_SIZES,
                        help="0.8b (models/onnx, $ONEJEV_MODEL_DIR) or 4b (models/onnx-4b, $ONEJEV4_MODEL_DIR)")
    parser.add_argument("--model", default=None, help="model directory (overrides --model-size and the environment)")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    decider = Decider(default_model_dir(args.model_size, ROOT, args.model), variant=args.variant,
                      intra_op_threads=args.threads, load_vision=False)
    correct = 0
    records = []
    started = time.perf_counter()
    for state, question, expected in CASES:
        t0 = time.perf_counter()
        answer = decider.decide(state, {"q": question})["q"]
        elapsed = (time.perf_counter() - t0) * 1000
        got = decision(answer)
        ok = got == expected
        correct += ok
        records.append({"type": question.type, "expected": expected, "got": got,
                        "confidence": confidence_of(answer), "ms": round(elapsed, 1)})
        print("%s %-7s got=%-20s expected=%-20s p=%.3f  %5.0f ms"
              % ("ok " if ok else "XX ", question.type, got, expected,
                 confidence_of(answer), elapsed))

    total = time.perf_counter() - started
    print("\nvariant=%s  %d/%d correct  (%.1fs total, %.0f ms/decision)"
          % (args.variant, correct, len(CASES), total, total * 1000 / len(CASES)))
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"variant": args.variant, "correct": correct,
                        "total": len(CASES), "cases": records}, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
