# -*- coding: utf-8 -*-
"""OneJev typed decisions on onnxruntime + numpy.

One prefill per question, no generation. The prompt ends at the assistant turn;
the next-token logits are read at the option letters' token ids and softmaxed
among themselves. Because the answer is one of the options supplied, it can
never go off-schema.

Three ONNX graphs are involved:

    embed_tokens          input_ids  -> inputs_embeds
    vision_encoder        pixel_values, image_grid_thw -> image_features
    decoder_model_merged  inputs_embeds, attention_mask, position_ids, state -> logits

The decoder is a hybrid: linear-attention layers carry conv/recurrent state and
the full-attention layers carry a KV cache. For a single prefill every state
input is fed empty, and its shape is read from the graph rather than assumed.

Runtime dependencies: onnxruntime and numpy; Pillow only when images are used.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from .devices import CPU, GpuUnavailableError, check_used, select_providers
from .chat import IMAGE_PLACEHOLDER, position_ids, render
from .protocol import (
    LETTERS,
    PLACEHOLDER,
    SLOT_LABELS,
    Question,
    build_messages,
    make_answer,
    media_order,
    question_from_spec,
    render_question,
)
from .tokenizer import QwenBPE

__all__ = ["Decider", "VARIANTS", "gpu_providers"]

VARIANTS = ("fp32", "fp16", "q4f16")

_SUFFIX = {"fp32": "", "fp16": "_fp16", "q4f16": "_q4f16"}

_NUMPY_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def gpu_providers() -> List[str]:
    """[CUDA, CPU] if this onnxruntime build has CUDA, else [CPU] (for callers that pass ``providers=`` themselves)."""
    return select_providers("auto")


def _session(path: pathlib.Path, threads: Optional[int], providers: Optional[Sequence[str]] = None,
             require_gpu: bool = False) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError("%s not found (run `uv run download_model.py`, or set ONEJEV_MODEL_DIR)" % path)
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    wanted = list(providers) if providers else [CPU]
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=wanted)
    except Exception as exc:        # e.g. CUDA out of memory while loading the weights
        if require_gpu:
            raise GpuUnavailableError("%s could not load %s: %s"
                                      % (wanted[0], path.name, str(exc).splitlines()[0][:200])) from exc
        if wanted == [CPU]:
            raise
        print("warning: %s could not load %s (%s); using the CPU"
              % (wanted[0], path.name, str(exc).splitlines()[0][:160]), file=sys.stderr)
        return ort.InferenceSession(str(path), sess_options=options, providers=[CPU])
    # onnxruntime does not raise when CUDA cannot be set up: it builds the session on the CPU. Look at what it really chose.
    warning = check_used(session.get_providers(), wanted, require_gpu, name=path.name)
    if warning:
        print("warning: " + warning, file=sys.stderr)
    return session


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - values.max())
    return shifted / shifted.sum()


class Decider:
    """Loads one variant of the three graphs and answers typed questions."""

    def __init__(
        self,
        model_dir: Union[str, pathlib.Path],
        variant: str = "fp16",
        intra_op_threads: Optional[int] = None,
        load_vision: bool = True,
        image_config=None,
        media_root: Optional[Union[str, pathlib.Path]] = None,
        providers: Optional[Sequence[str]] = None,
        require_gpu: bool = False,
    ):
        """``providers``: onnxruntime execution providers, e.g. ``select_providers("cuda")`` (default: CPU only).

        ``require_gpu``: raise GpuUnavailableError unless every loaded graph really runs on CUDA (no silent CPU fallback).
        """
        if variant not in VARIANTS:
            raise ValueError("unknown variant %r; pick one of %s" % (variant, list(VARIANTS)))
        model_dir = pathlib.Path(model_dir)
        self.model_dir = model_dir
        self.variant = variant
        self.media_root = pathlib.Path(media_root) if media_root else None
        suffix = _SUFFIX[variant]

        self.embed = _session(model_dir / "onnx" / ("embed_tokens%s.onnx" % suffix), intra_op_threads, providers, require_gpu)
        # Text-only callers can skip the vision encoder entirely.
        self.vision = (
            _session(model_dir / "onnx" / ("vision_encoder%s.onnx" % suffix), intra_op_threads, providers, require_gpu)
            if load_vision else None
        )
        self.decoder = _session(
            model_dir / "onnx" / ("decoder_model_merged%s.onnx" % suffix), intra_op_threads, providers, require_gpu
        )

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.image_token_id: int = config["image_token_id"]
        self.merge_size: int = config["vision_config"]["spatial_merge_size"]
        self.hidden_size: int = config["text_config"]["hidden_size"]

        self.tokenizer = QwenBPE(str(model_dir / "tokenizer.json"))
        self._image_config = image_config
        self._state_spec = self._empty_state_spec()
        # Newer exports can project only the last K positions through the
        # output head. The answer lives at the last position, so K=1 saves the
        # [1, S, 248320] logits tensor (hundreds of MB at long prompts).
        self._keep_input = any(i.name == "num_logits_to_keep" for i in self.decoder.get_inputs())
        self.slot_ids = self._verify_slots()

    @property
    def provider(self) -> str:
        """The execution provider the main graph actually runs on."""
        return self.decoder.get_providers()[0]

    @property
    def providers_used(self) -> Dict[str, str]:
        """The provider each loaded graph really runs on, e.g. {"embed": "CUDAExecutionProvider", ...}."""
        out = {"embed": self.embed.get_providers()[0]}
        if self.vision is not None:
            out["vision"] = self.vision.get_providers()[0]
        out["decoder"] = self.decoder.get_providers()[0]
        return out

    # ------------------------------------------------------------- plumbing

    @property
    def image_config(self):
        """Lazy: importing image.py pulls in Pillow, which text-only use avoids."""
        if self._image_config is None:
            from .image import ImageConfig

            self._image_config = ImageConfig.from_pretrained(self.model_dir)
        return self._image_config

    def _verify_slots(self) -> List[int]:
        """Token id of every answer label that is one round-trip token.

        A..Z are required. The two-letter labels are all-or-nothing: if the
        tokenizer splits any of them the runtime keeps 26 slots, and questions
        with more options are rejected.
        """
        def one(label: str) -> Optional[int]:
            ids = self.tokenizer.encode(label)
            return ids[0] if len(ids) == 1 else None

        ids = [one(label) for label in LETTERS]
        if any(i is None for i in ids):
            bad = [label for label, i in zip(LETTERS, ids) if i is None]
            raise ValueError("answer slots %s are not single tokens in this tokenizer" % bad)
        extra = [one(label) for label in SLOT_LABELS[len(LETTERS):]]
        if all(i is not None for i in extra):
            ids += extra
        if len(set(ids)) != len(ids):
            raise ValueError("answer slot tokens collide")
        return ids

    def _empty_state_spec(self) -> List[Tuple[str, Tuple[int, ...], np.dtype]]:
        """Shapes for every decoder state input, with batch=1 and no past tokens."""
        spec = []
        for node in self.decoder.get_inputs():
            if node.name in ("inputs_embeds", "attention_mask", "position_ids",
                             "num_logits_to_keep"):
                continue
            shape = []
            for axis, dim in enumerate(node.shape):
                if isinstance(dim, int):
                    shape.append(dim)
                elif axis == 0:
                    shape.append(1)  # batch
                else:
                    shape.append(0)  # past_sequence_length
            dtype = _NUMPY_DTYPE.get(node.type)
            if dtype is None:
                raise RuntimeError("unhandled decoder input type %s" % node.type)
            spec.append((node.name, tuple(shape), dtype))
        return spec

    def _empty_state(self) -> Dict[str, np.ndarray]:
        return {name: np.zeros(shape, dtype=dtype) for name, shape, dtype in self._state_spec}

    def _embed(self, input_ids: Sequence[int]) -> np.ndarray:
        ids = np.asarray([list(input_ids)], dtype=np.int64)
        return self.embed.run(["inputs_embeds"], {"input_ids": ids})[0]

    def _image_features(self, prepared) -> np.ndarray:
        if self.vision is None:
            raise RuntimeError("this Decider was built with load_vision=False; images are unavailable")
        return self.vision.run(
            ["image_features"],
            {
                "pixel_values": prepared.pixel_values.astype(np.float32),
                "image_grid_thw": prepared.image_grid_thw.astype(np.int64),
            },
        )[0]

    # --------------------------------------------------------------- prompts

    def _resolve_images(self, images: Sequence[Any]) -> List[Any]:
        root = self.media_root
        out = []
        for source in images:
            if isinstance(source, (str, pathlib.Path)):
                text = str(source)
                is_uri = text.startswith("data:image/")
                if not is_uri and not pathlib.Path(text).is_absolute() and root is not None:
                    text = str(root / text)
                out.append(text)
            else:
                out.append(source)
        return out

    def render_prompt(self, state: Any, rendered, counts: Sequence[int] = ()) -> str:
        """The chat-rendered prompt for one question."""
        messages = build_messages(state, rendered.suffix)
        # <image:N> becomes the chat template's vision block, expanded to the
        # number of visual tokens that image really costs.
        text = PLACEHOLDER.sub(IMAGE_PLACEHOLDER, messages[-1]["content"])
        messages = messages[:-1] + [{"role": "user", "content": text}]
        return render(messages, counts)

    # --------------------------------------------------------------- forward

    def last_logits(self, prompt_ids: Sequence[int], prepared=None,
                    features: Optional[np.ndarray] = None) -> np.ndarray:
        """Logits at the last position, one float per vocabulary entry.

        ``features`` are the vision encoder's output for ``prepared``. Every
        question about the same images shares them, so decide() computes them
        once; they are computed here only when the caller did not.
        """
        embeds = self._embed(prompt_ids)

        if prepared is not None and len(prepared.image_grid_thw):
            if features is None:
                features = self._image_features(prepared)
            slots = np.flatnonzero(np.asarray(prompt_ids) == self.image_token_id)
            if slots.size != features.shape[0]:
                raise RuntimeError(
                    "vision encoder produced %d features for %d image tokens"
                    % (features.shape[0], slots.size)
                )
            embeds[0, slots, :] = features.astype(embeds.dtype)

        length = len(prompt_ids)
        feeds = self._empty_state()
        feeds["inputs_embeds"] = embeds
        feeds["attention_mask"] = np.ones((1, length), dtype=np.int64)
        feeds["position_ids"] = position_ids(
            prompt_ids,
            prepared.image_grid_thw if prepared is not None else None,
            self.image_token_id,
            self.merge_size,
        )
        if self._keep_input:
            feeds["num_logits_to_keep"] = np.asarray(1, dtype=np.int64)
        logits = self.decoder.run(["logits"], feeds)[0]
        return logits[0, -1, :].astype(np.float64)

    # --------------------------------------------------------------- decide

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Union[Question, Mapping[str, Any]]],
        images: Sequence[Any] = (),
    ) -> Dict[str, Dict[str, Any]]:
        """Answer every question about ``state``.

        ``state`` may be text or any JSON-able object. Images are referenced
        from it as ``<image:1>``, ``<image:2>``, ... once each, in order.
        """
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        # a missing / extra / out-of-order placeholder is a request bug
        media_order(text, len(images))

        prepared = None
        features = None
        counts: List[int] = []
        if images:
            from .image import prepare_images

            prepared = prepare_images(self._resolve_images(images), self.image_config)
            counts = prepared.token_counts
            # one vision pass per request, not per question
            features = self._image_features(prepared)

        answers: Dict[str, Dict[str, Any]] = {}
        for qid, spec in questions.items():
            question = spec if isinstance(spec, Question) else question_from_spec(spec)
            rendered = render_question(question)
            if PLACEHOLDER.search(rendered.suffix):
                raise ValueError("question %r: image placeholders belong in the state, not a question" % qid)
            if rendered.n_slots > len(self.slot_ids):
                raise ValueError(
                    "question %r has %d options but this tokenizer has %d single-token slots"
                    % (qid, rendered.n_slots, len(self.slot_ids))
                )
            prompt = self.render_prompt(state, rendered, counts)
            logits = self.last_logits(self.tokenizer.encode(prompt), prepared, features)
            probabilities = _softmax(logits[np.asarray(self.slot_ids[: rendered.n_slots])])
            answers[qid] = make_answer(rendered, probabilities.tolist())
        return answers

    def predict(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """A System One style request in, a response out.

        ``{"state": ..., "media": [{"type": "image", "path": ...}], "questions":
        {id: {"type": ..., "instructions": ..., "criteria": ...}}}``
        """
        media = list(request.get("media") or [])
        images = []
        for index, item in enumerate(media):
            if item.get("type") != "image":
                raise ValueError("media[%d]: this build handles images only" % index)
            ref = next((item[k] for k in ("image", "path", "data", "url") if k in item), None)
            if ref is None:
                raise ValueError("media[%d] needs one of image, path, data or url" % index)
            if isinstance(ref, str) and ref.startswith(("http://", "https://")):
                raise ValueError("media[%d]: fetching URLs is not supported; pass a path or data" % index)
            if isinstance(ref, str) and "data" in item and not ref.startswith("data:"):
                ref = "data:image/png;base64," + ref
            images.append(ref)
        answers = self.decide(request.get("state"), request["questions"], images)
        return {"answers": answers, "model": "OneJev-0.8B", "backend": "onnxruntime"}

