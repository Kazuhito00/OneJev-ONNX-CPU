# -*- coding: utf-8 -*-
"""Golden outputs from the reference implementation (qev, PyTorch, CPU).

Run in an environment with the `export` group (uv sync --group export). Everything downstream is checked against this file, so it
must be readable without torch: JSON and .npz only, never a pickle.

Needs the reference package and the original checkpoint:

    git clone https://github.com/OmniJev/OneJev ref/OneJev
    uv pip install --no-deps -e ref/OneJev   # plus the `export` dependency group (pydantic, jinja2, fastapi ...)
    python export/prepare_model.py --checkpoint

For every question it records the rendered prompt, the expanded input ids, the
raw option-slot logits and the final typed answer, so each stage of the port can
be checked on its own.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
HF = ROOT / "models" / "hf"
IMAGES = ROOT / "artifacts" / "images"

CHOICE_30 = {"opt%02d" % i: "option number %d" % i for i in range(30)}

CASES = [
    {
        "id": "support_json",
        "state": {"ticket": "My card was charged twice for the same order and support has not "
                            "replied in 3 days.",
                  "customer": {"tenure_months": 14, "vip": False}},
        "questions": {
            "angry": {"type": "noul", "instructions": "The customer is angry"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "charges and refunds", "tech": "bugs and outages",
                                  "sales": "pricing and contracts"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["low", "medium", "high", "critical"]},
        },
    },
    {
        "id": "bug_text",
        "state": "The export button crashes in Safari but works in Chrome. "
                 "Not blocking, we told users to switch browsers.",
        "questions": {
            "severity": {"type": "choice", "instructions": "How severe is this bug?",
                         "criteria": {"cosmetic": "cosmetic only", "degraded": "degraded, workaround exists",
                                      "blocking": "blocking, no workaround"}},
            "escalate": {"type": "noul", "instructions": "This should be escalated immediately",
                         "criteria": {"true": "escalate now", "false": "can wait"}},
        },
    },
    {
        "id": "incident",
        "state": "URGENT: production database is down, all customers affected, "
                 "CEO asking for updates every 5 minutes.",
        "questions": {
            "priority": {"type": "score", "instructions": "How urgent is this?",
                         "criteria": ["low", "medium", "high", "critical"]},
            "security": {"type": "noul", "instructions": "This is a security incident"},
            "page": {"type": "noul", "instructions": "The on-call engineer should be paged"},
        },
    },
    {
        "id": "japanese",
        "state": "先週の納品が遅れた件、まだ連絡がありません。至急対応してください。",
        "questions": {
            "angry": {"type": "noul", "instructions": "The customer is angry"},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["low", "medium", "high"]},
        },
    },
    {
        "id": "many_options",
        "state": "Pick the option whose number is closest to twelve.",
        "questions": {
            "pick": {"type": "choice", "instructions": "Which option is it?", "criteria": CHOICE_30},
        },
    },
    {
        "id": "image_colour_red",
        "state": {"task": "Describe the picture", "picture": "<image:1>"},
        "media": [{"type": "image", "path": "red.png"}],
        "questions": {
            "colour": {"type": "choice", "instructions": "What colour fills the picture?",
                       "criteria": {"red": "mostly red", "green": "mostly green", "blue": "mostly blue"}},
        },
    },
    {
        "id": "image_colour_blue",
        "state": {"task": "Describe the picture", "picture": "<image:1>"},
        "media": [{"type": "image", "path": "blue.png"}],
        "questions": {
            "colour": {"type": "choice", "instructions": "What colour fills the picture?",
                       "criteria": {"red": "mostly red", "green": "mostly green", "blue": "mostly blue"}},
        },
    },
    {
        "id": "image_shape",
        "state": {"photo": "<image:1>", "context": "a drawing uploaded by a user"},
        "media": [{"type": "image", "path": "circle.png"}],
        "questions": {
            "shape": {"type": "choice", "instructions": "What shape is drawn?",
                      "criteria": {"circle": "a circle", "square": "a square", "triangle": "a triangle"}},
            "drawing": {"type": "noul", "instructions": "The picture contains a drawn shape"},
        },
    },
    {
        "id": "two_images",
        "state": "Compare <image:1> with <image:2>.",
        "media": [{"type": "image", "path": "red.png"}, {"type": "image", "path": "blue.png"}],
        "questions": {
            "same": {"type": "noul", "instructions": "Both pictures have the same colour"},
            "first": {"type": "choice", "instructions": "What colour is the first picture?",
                      "criteria": {"red": "red", "blue": "blue"}},
        },
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/golden.json")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--hf", default=str(HF), help="checkpoint directory (default: models/hf, the 0.8B)")
    parser.add_argument("--name", default="OneJev-0.8B")
    parser.add_argument("--pixels", default="artifacts/golden_pixels.npz")
    args = parser.parse_args()

    from PIL import Image
    from qev.mm_engine import MMDecisionEngine
    from qev.prompt import render_question
    from qev.schema import SystemOneRequest

    engine = MMDecisionEngine(args.hf, device="cpu", dtype=args.dtype,
                              fork_mode="sequential", media_root=str(IMAGES))

    captured = []
    original_prepare = engine._prepare

    def spy(state, rendered):
        prep = original_prepare(state, rendered)
        captured.append(prep)
        return prep

    engine._prepare = spy

    records = []
    pixels = {}
    for case in CASES:
        request = {"state": case["state"], "questions": case["questions"]}
        if case.get("media"):
            request["media"] = case["media"]
        parsed = SystemOneRequest.model_validate(request)

        captured.clear()
        response, meta = engine.decide(parsed, debug=True)
        prep = captured[-1]

        branches = []
        for index, (qid, question) in enumerate(parsed.questions.items()):
            rendered = render_question(qid, question)
            engine._set_media(list(case.get("media") or []))
            try:
                prompt = engine._render_prompt(parsed.state, rendered.suffix)
            finally:
                engine._set_media([])
            branches.append({
                "qid": qid,
                "prompt": prompt,
                "input_ids": list(prep.prefix_ids) + list(prep.suffixes[index]),
                "slot_logits": [float(x) for x in meta["diagnostics"][qid]["slot_logits"][0]],
                "n_slots": rendered.n_slots,
            })

        if case.get("media"):
            images = [Image.open(IMAGES / m["path"]).convert("RGB") for m in case["media"]]
            out = engine.processor(text=[branches[0]["prompt"]], images=images, return_tensors="np")
            pixels["%s__pixel_values" % case["id"]] = np.asarray(out["pixel_values"], dtype=np.float32)
            pixels["%s__image_grid_thw" % case["id"]] = np.asarray(out["image_grid_thw"], dtype=np.int64)

        answers = {qid: json.loads(answer.model_dump_json()) for qid, answer in response.answers.items()}
        records.append({"id": case["id"], "request": request, "branches": branches,
                        "answers": answers})
        print("  %-18s %s" % (case["id"], {
            k: (v.get("choice") if v["type"] == "choice" else
                round(v["noul"], 3) if v["type"] == "noul" else round(v["score"], 2))
            for k, v in answers.items()}))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.name, "dtype": args.dtype, "cases": records},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(ROOT / args.pixels, **pixels)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
