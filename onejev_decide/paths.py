# -*- coding: utf-8 -*-
"""Where the model files are: one rule shared by the demos, verify/ and the benchmark adapter.

The argument wins over the environment, the environment over the checkout's ``models/`` folder:

    model size 0.8b   $ONEJEV_MODEL_DIR    else <root>/models/onnx
    model size 4b     $ONEJEV4_MODEL_DIR   else <root>/models/onnx-4b

(``ONEJEV_MODEL_DIR`` / ``ONEJEV4_MODEL_DIR`` are also the names the wikipedia-vision-decision-bench adapter reads.)
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional, Union

MODEL_SIZES = ("0.8b", "4b")

_ENV = {"0.8b": "ONEJEV_MODEL_DIR", "4b": "ONEJEV4_MODEL_DIR"}
_FOLDER = {"0.8b": "onnx", "4b": "onnx-4b"}


def default_model_dir(model_size: str = "0.8b", root: Union[str, os.PathLike] = ".",
                      explicit: Optional[Union[str, os.PathLike]] = None) -> pathlib.Path:
    """``explicit`` (relative paths are taken from ``root``), else the environment variable, else ``<root>/models/<folder>``."""
    if model_size not in _ENV:
        raise ValueError("model size must be one of %s, not %r" % (", ".join(MODEL_SIZES), model_size))
    root = pathlib.Path(root)
    if explicit:
        return root / explicit          # an absolute ``explicit`` replaces ``root``
    if os.environ.get(_ENV[model_size]):
        return pathlib.Path(os.environ[_ENV[model_size]])
    return root / "models" / _FOLDER[model_size]
