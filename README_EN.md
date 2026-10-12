[[Japanese](README.md)/[English](README_EN.md)]

# OneJev-ONNX-CPU

A minimal-dependency inference implementation of [`OmniJev/OneJev-0.8B`](https://huggingface.co/OmniJev/OneJev-0.8B) and [`OmniJev/OneJev-4B`](https://huggingface.co/OmniJev/OneJev-4B) (typed decision models for text and images) that runs on onnxruntime and numpy only. GPU (CUDA) inference is possible too.<br>
It does not use torch, transformers, pydantic, fastapi, etc., which the reference implementation (`qev`) requires.

# Features
- Minimal dependencies: only onnxruntime and numpy at run time (plus pillow when images are used)
- Two scripts are enough to run it: `download_model.py` (fetch the model) and `demo_inference_text.py` (inference). Images: `demo_inference_image.py`; camera: `demo_inference_webcam.py`
- Both sizes, 0.8B and 4B, in one repository (`--model-size 0.8b|4b`)
- Decisions agree with the reference implementation (PyTorch fp32): 17/17 for both sizes (max logit difference 0.010 for 0.8B fp16, 0.006 for 4B)
- One forward pass per question, no generation. The answer is always one of the options given (the logits at the option-letter tokens are read and softmaxed)
- Images supported (referenced from the state as `<image:1>`, `<image:2>`, ...)
- The preprocessing (Qwen byte-level BPE, chat template, 3-D M-RoPE, image patching) is ported to pure Python, with checks against the reference included
- Runs on the CPU, and on the GPU (CUDA) with `--device cuda`. The providers actually used are printed; a silent fallback to the CPU is a warning (an error with `--require-gpu`)

# Purpose of This Repository
- How far can OneJev's decisions be run with only 2-3 runtime dependencies?
- Can the reference preprocessing be ported to pure Python so that the same inputs are built?
- What are the speed and memory on the CPU (and the GPU)?

See [artifacts/report.md](artifacts/report.md) for measurements and the evidence for the port (a record from the time of the work: 0.8B, CPU only; Japanese).

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later   (onnxruntime-gpu to run on the GPU; never both, see below)
pillow       10.0 or later   (only when images are used)
```
That is all that is needed to run. Downloading the model uses only the standard library.<br>
Environments are built with [uv](https://docs.astral.sh/uv/).<br>
Optional groups and extras:
- `cpu` (default group): onnxruntime (CPU build), installed by `uv sync`
- `gpu` (extra): onnxruntime-gpu; it cannot be installed together with `cpu` (see "Running on the GPU")
- `test`: pytest for `tests/`
- `verify`: `tokenizers`, for the tokenizer cross-check (`check_tokenizer_fuzz`)
- `webcam`: OpenCV, for `demo_inference_webcam.py`
- `export`: torch, transformers, etc., to export ONNX yourself or to rebuild the golden data (never used at run time)

# Installation

```bash
git clone https://github.com/Kazuhito00/OneJev-ONNX-CPU
cd OneJev-ONNX-CPU

# runtime dependencies only (onnxruntime, numpy, pillow)
uv sync
```

# Download Model
The ONNX files are on the [Releases](https://github.com/Kazuhito00/OneJev-ONNX-CPU/releases) page (they are not in the repository).<br>
This script fetches them into `models/`. It needs only the standard library (HTTPS), no git, no git-lfs.
```bash
uv run download_model.py                       # 0.8B (fp16 and q4f16, tokenizer, settings; about 2.9 GB)
uv run download_model.py --model-size 0.8b     # 0.8B only (about 2.9 GB)
uv run download_model.py --model-size 4b       # 4B only (fp16 only; about 10.4 GB)
```
- **Without arguments it fetches every file of both 0.8B and 4B** (about 13 GB; 0.8B alone is 21 files, about 2.9 GB). 4B needs a lot of RAM to run, so add `--model-size 0.8b` to save disk.
- The default source is the latest release of `Kazuhito00/OneJev-ONNX-CPU` (use `--repo OWNER/NAME` or the environment variable `ONEJEV_ONNX_REPO` for another one; priority: argument > environment > default)
- 0.8B goes to `models/onnx`, 4B to `models/onnx-4b`
- Every file is checked against the SHA-256 in the release's `manifest-0.8b.json` / `manifest-4b.json` (a mismatch deletes the file and stops)
- Files split to stay under GitHub's limit (below 2 GiB per file) are joined and checked after the download. In the current release the largest file is 2,095,788,032 bytes (1.95 GiB), so nothing is split
- A file already in place is skipped; an interrupted download continues where it stopped when the same command is run again

```bash
uv run download_model.py --list                             # list the files
uv run download_model.py --variant fp16                     # skip q4f16 (about 2.1 GB)
uv run download_model.py --exclude "vision_encoder*"        # text-only: skip the vision encoder
uv run download_model.py --tag v0.0.0                       # a given tag (default: latest)
```

Release assets (built by `release/make_release.py`; list them with `uv run release/make_release.py --plan`. v0.0.0 is intended; nothing has been uploaded yet). Asset names are the paths with `/` replaced by `__`:

| What | Files | Size |
|---|---|---:|
| 0.8B fp16 | `onnx__onnx__decoder_model_merged_fp16.onnx` + `.onnx_data` | 600,469 B + 1,504,886,784 B |
| | `onnx__onnx__embed_tokens_fp16.onnx` + `.onnx_data` | 637 B + 508,559,360 B |
| | `onnx__onnx__vision_encoder_fp16.onnx` + `.onnx_data` | 187,476 B + 204,798,464 B |
| 0.8B q4f16 | `onnx__onnx__{decoder_model_merged,embed_tokens,vision_encoder}_q4f16.onnx` + `.onnx_data` | about 0.65 GB in total |
| 0.8B shared | `onnx__tokenizer.json` (about 20 MB), `onnx__config.json`, `onnx__preprocessor_config.json`, etc., `onnx__LICENSE-ONEJEV`, `onnx__LICENSE-QWEN`, `manifest-0.8b.json`, `SHA256SUMS-0.8b` | about 20 MB |
| 4B fp16 | `onnx-4b__onnx__decoder_model_merged_fp16.onnx` (1.3 MB) + `.onnx_data`, `_data_1` ... `_data_4` | 2,095,788,032 / 2,058,608,640 / 2,094,993,408 / 892,618,752 / 1,271,398,400 B |
| | `onnx-4b__onnx__embed_tokens_fp16.onnx` + `.onnx_data` | 1,271,398,400 B |
| | `onnx-4b__onnx__vision_encoder_fp16.onnx` + `.onnx_data` | 671,943,680 B |
| 4B shared | `onnx-4b__tokenizer.json` etc., `onnx-4b__LICENSE-*`, `manifest-4b.json`, `SHA256SUMS-4b` | about 20 MB |

To prepare the files yourself:
```bash
# 0.8B: fetch the public onnx-community/OneJev-0.8B-ONNX from Hugging Face (the same files as the 0.8B release)
uv run export/prepare_model.py
# 4B: no public ONNX exists, so the OneJev-4B weights are swapped into the graphs of the public Qwen3.5-4B ONNX
# (where the inputs go is at the top of export/build_onnx_4b.py)
uv sync --group export
uv run export/build_onnx_4b.py verify    # first check that the swap rules reproduce the public Qwen3.5-4B ONNX
uv run export/build_onnx_4b.py build
```

# Usage

### Examples (Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
uv run demo_inference_text.py --model-size 4b       # 4B ($ONEJEV4_MODEL_DIR or models/onnx-4b)
```
The model location is: `--model` > environment (`ONEJEV_MODEL_DIR` for 0.8B, `ONEJEV4_MODEL_DIR` for 4B) > `models/onnx` (`models/onnx-4b` for 4B).<br>
Your own input goes in `--state` and `--choice` / `--score` / `--noul`.
```bash
uv run demo_inference_text.py --state "I was charged twice." --choice "team=billing,technical" --noul "angry=The customer is angry"
uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
```
The image demo draws a circle (`artifacts/images/demo_circle.png`, generated when it runs) and uses it by default. Images are referenced from the state as `<image:1>`, `<image:2>`, ... (each once, in order).

```python
from onejev_decide import Decider, choice, noul, score

d = Decider("models/onnx", variant="fp16")        # "models/onnx-4b" for 4B; load_vision=False for text only

d.decide(
    {"ticket": "My card was charged twice for the same order and support has not replied in 3 days."},
    {
        "angry": noul("The customer is angry"),
        "team": choice("Which team should handle this?",
                       {"billing": "charges and refunds", "tech": "bugs and outages"}),
        "urgency": score("How urgent is this?", ["low", "medium", "high", "critical"]),
    },
)
# {"angry":   {"type": "noul",   "noul": 0.19},
#  "team":    {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": ...},
#  "urgency": {"type": "score",  "score": 1.99, "legend": {...}, "probabilities": {...}, ...}}

d.decide({"screen": "<image:1>", "task": "Pay the open invoice"},
         {"done": noul("The invoice has been paid")}, images=["screenshot.png"])
```
Requests in the reference implementation's System One format are accepted too: `d.predict({"state": ..., "media": [{"type": "image", "path": ...}], "questions": {...}})`.

**One forward pass answers a question.** The prompt ends at the assistant turn; the next-token logits are read at the option-letter tokens (`A`, `B`, ...) and softmaxed. Nothing is generated. `noul` has `yes` as A (the answer is p(yes)), `score` is the expected value over the levels, and more than 26 options use two-letter labels (up to 255). Each question prefills the whole prompt, so the time grows roughly with the number of questions (the reference implementation's reuse of the shared prefix as a KV cache is not implemented). The vision encoder runs once per request and its output is reused.

### Is there a person in the camera image?
Grabs camera frames with OpenCV, decides whether a person is in the newest frame and draws the result on the live image.
```bash
uv run --group webcam demo_inference_webcam.py                        # camera 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/circle.png   # an image or video file instead of a camera
```
- Decisions run in their own thread, so the live image never stalls. The banner (green = yes, red = no) and the probability of "yes" update whenever a decision is ready
- Change the question with `--question` (a yes/no question). The image given to the model is shrunk to 224 px on the long edge (`--size`)
- Quit with `q` or `Esc`. OpenCV is a dependency of this demo only. `--no-window` prints the decisions without opening a window

### Question types
| `type` | Answer |
|---|---|
| `choice` | one of the options (`choice`, `confidence`, `probabilities`) |
| `score` | expected value on an ordinal scale (`score`, `legend`); the first level is the lowest (up to 10 levels) |
| `noul` | whether a statement holds (`noul` = probability of yes) |

### Choosing a model
| | Size | Notes |
|---|---:|---|
| 0.8B `fp16` | about 2.1 GB | **the CPU default, and the fastest**. 17/17 agree, logit difference <= 0.010 |
| 0.8B `q4f16` | about 0.65 GB | small, but slower than fp16 on the CPU and a little worse (17/17 agree, but logit difference <= 1.29; 11/12 on the clear 12-question set) |
| 4B `fp16` | about 10.4 GB | more accurate but heavy; needs a lot of RAM (`--model-size 4b`). 17/17 agree, max logit difference 0.006 |

`--variant q4f16` selects the 0.8B q4f16. There is no fp32 ONNX (the decoder would be one 3.0 GB file, over the release's 2 GiB per-file limit).

### Running on the GPU (optional)
`--device cuda` (or `auto`) runs on CUDA. Without it everything runs on the CPU (the default is `--device cpu`).

| Option | Behaviour |
|---|---|
| `--device cpu` (default) | CPU only |
| `--device cuda` | `CUDAExecutionProvider`, then `CPUExecutionProvider`. If this onnxruntime has no CUDA, or CUDA cannot be loaded, it **warns and runs on the CPU** |
| `--device auto` | CUDA if available, else the CPU |
| `--require-gpu` | require every graph to really run on CUDA; if CUDA cannot be used it **stops with an error** (prevents the accident of silently running on the CPU while thinking it is the GPU; implies `--device cuda`) |

**The providers actually used are printed on the last lines of the demos (`providers actually used: embed=CUDA, vision=CUDA, decoder=CUDA`) and in the output of `verify/bench.py` (`providers_used`).** onnxruntime does not raise when it cannot set up CUDA; it builds the session on the CPU. Asking for CUDA is therefore no proof that the GPU was used.

```bash
# 1. swap in the GPU build (drop the cpu group, add the gpu extra)
uv sync --extra gpu --no-group cpu
# 2. run (plain `uv run` would sync back to the default cpu group, so pass --no-sync; UV_NO_SYNC=1 does the same)
uv run --no-sync demo_inference_text.py --device cuda --require-gpu
uv run --no-sync verify/bench.py --device cuda --require-gpu
```
```python
from onejev_decide import Decider, select_providers

d = Decider("models/onnx", providers=select_providers("cuda"), require_gpu=True)   # GpuUnavailableError unless on CUDA
print(d.providers_used)   # {"embed": "CUDAExecutionProvider", "vision": ..., "decoder": ...}
```

Requirements and notes:
- **`onnxruntime` and `onnxruntime-gpu` cannot be installed together** (both write into the same `onnxruntime/` directory). `pyproject.toml` puts the CPU build in the `cpu` group and the GPU build in the `gpu` extra and declares them as conflicting, so `uv sync --extra gpu` alone fails (the default `cpu` group would be installed too). Always add `--no-group cpu`. To go back to the CPU, run `uv sync`
- **CUDA 13 and cuDNN 9 are required** (for `onnxruntime-gpu` 1.31). Put both `bin` folders on PATH. `onnxruntime-gpu` does not bundle the CUDA DLLs and looks for them on PATH (`CUDA_PATH` may point at an older CUDA). When they are missing, a warning that `cublasLt64_13.dll` is not found appears and it runs on the CPU (an error with `--require-gpu`). Windows example (adapt the versions); reopen the terminal after changing PATH:
  ```text
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
  C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
  ```
- Probabilities on the GPU do not match the CPU exactly (order of operations; the maximum difference in the measurements below is 2.3e-03). The checks against the reference (`verify/`) are CPU only
- DirectML (`onnxruntime-directml`) has not been tried with this model (**unverified**). There is no measurement of q4f16 on the GPU either (**unverified**)

**VRAM guide** (measured on an RTX 3050 Ti Laptop, 4 GB; the increase seen by `nvidia-smi`): 0.8B fp16 uses about 3.2 GiB for text and about 3.7 GiB for images (it fits in 4 GB). **4B fp16 has about 8 GB of weights and does not fit in 4 GB.** A CUDA session can still be created, but a question takes 15-19 s, slower than the CPU (about 6.5 s). Why the session is created although the weights do not fit has not been investigated, and `--require-gpu` cannot detect it. On a GPU with little VRAM, use the CPU for 4B.

# Speed and accuracy
Measured on this work's PC (Core i7-12800H, RTX 3050 Ti Laptop 4 GB, CPU with 8 threads), 224 px images, fp16. **The numbers change a lot with the state of the machine at the time.** Only existing measurements that have a source are listed.

### Time per question (source: the comparison table in the README of `wikipedia-vision-decision-bench`; median over 1268 Commons questions, calibrated blocks only)
| Model | CPU (8 threads) | GPU (RTX 3050 Ti Laptop 4GB, CUDA) |
|---|---:|---:|
| OneJev 0.8B fp16 | 1.2 s | 87 ms |
| OneJev 4B fp16 | 6.5-7.1 s | not measured (slower than the CPU in the separate measurement below) |

### GPU measurement (source: `eval_common/gpu_bench/report.md`, 2026-10-10, onnxruntime-gpu 1.31.0, CUDA 13.4, cuDNN 9.20, CPU with 8 threads)
Three fixed questions each (3 text, 3 image at 224 px). "Per question" is the median of all timings.

| Model | Input | CPU per question | CUDA per question | Ratio | Peak VRAM | Agreement with CPU (max probability difference) |
|---|---|---:|---:|---:|---:|---|
| 0.8B fp16 | text | 830 ms | 66 ms | 12.6x | 3192 MiB | 3/3 (5.8e-04) |
| 0.8B fp16 | image | 1277 ms | 80 ms | 16.0x | 3722 MiB | 3/3 (2.3e-03) |
| 4B fp16 | text | 6456 ms | 19172 ms | 0.3x (slower) | 3820 MiB (capped) | 3/3 (5.4e-05) |
| 4B fp16 | image | not measured (the CPU run was aborted for lack of RAM) | 15260 ms | - | 3820 MiB | - |

**0.8B is 12-16x faster on the GPU. 4B is slower on a 4 GB GPU** (not enough VRAM; it has not been measured on a GPU with more VRAM: **unverified**).

### Decision accuracy (%; source: the same bench README; 1268 Commons / 382 synthetic / 225 reasoning questions, CPU)
| Model | Commons en | Commons ja | synthetic en | synthetic ja | reasoning en | reasoning ja |
|---|---:|---:|---:|---:|---:|---:|
| OneJev 0.8B | 93.0 | 87.0 | 91.6 | 89.0 | 45.8 | 47.6 |
| OneJev 4B | 97.9 | 96.8 | 95.3 | 95.0 | 68.9 | 67.6 |

The reasoning set includes formats with few options where guessing already scores well; read it with chance accuracy in mind.

### Memory
0.8B peaks at about 3.9 GB RAM with fp16 and about 0.85 GB with q4f16 (measured during the work). 4B has about 10 GB of weights alone; in the bench's text measurement its peak RAM was 18.6 GB (other processes were also using RAM during that measurement).

To measure speed: `uv run verify/bench.py --model-size 0.8b` (one measurement per process; `--questions N`, `--image 224`, `--device cuda`).

# Verification
```bash
uv run verify/run_all.py                                 # the gates that need only the runtime dependencies (0.8B)
uv run --group verify verify/run_all.py --full-fuzz      # with tokenizers, run the fuzz at full size too
uv run verify/run_all.py --model-size 4b                 # 4B (uses about 10 GB of RAM)
uv run --group test pytest                               # unit tests that need no model (and no GPU)
```

| Gate | What it checks | Needs |
|---|---|---|
| check_prompt | prompt strings and expanded `input_ids` equal the reference's (9/9) | runtime dependencies |
| check_image | preprocessing equals the processor's `pixel_values` down to the pixels | + pillow |
| verify | option logits and typed answers equal PyTorch (fp32), 17 questions | runtime dependencies |
| eval_decisions | an unambiguous 12-question set (12/12 for 0.8B fp16) | runtime dependencies |
| check_bias | answers do not flip when the options are reordered (4/4) | runtime dependencies |
| check_tokenizer_fuzz | cross-check against the Rust `tokenizers` (`--full-fuzz`: 187,689 cases with the 0.8B `tokenizer.json`, 0 mismatches) | + tokenizers |

`artifacts/golden.json` (0.8B) and `golden_4b.json` (4B) are ground truth made by actually running the reference implementation (`qev`, PyTorch fp32, CPU). Both run the same weights, so a difference can only be a bug in the port. `artifacts/golden_pixels.npz` holds the `pixel_values` the processor made for the three images in `artifacts/images/`.

`tests/` has unit tests that need no model and no GPU (provider selection logic, how the model location is decided, building prompts and answers, the tokenizer, and a local round trip of `download_model.py` and `release/make_release.py`).

# Limitations
- One prefill per question (reuse of the shared prefix as a KV cache is not implemented); many questions are correspondingly slower
- Still images only; video (`<video:N>`) is not supported
- Images cannot be fetched from URLs (give a path or a data URI)
- `tokenizer.py` / `chat.py` are specific to this checkpoint and reject unexpected input with an exception
- 4B is heavy and does not fit a 4 GB GPU; q4f16 exists for 0.8B only
- Probabilities on the GPU do not match the CPU exactly

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # Apache-2.0
pyproject.toml           # dependencies (runtime: onnxruntime, numpy, pillow; groups: cpu / test / verify / webcam / export; extra: gpu)
uv.lock
download_model.py        # fetch the model (GitHub Releases, SHA-256 check, 0.8B / 4B selection)
demo_inference_text.py   # text inference demo
demo_inference_image.py  # image inference demo
demo_inference_webcam.py # "is there a person?" on a camera (OpenCV)
onejev_decide/           # the inference code (light dependencies only; never imports export/)
  runtime.py             #   ORT sessions, prefill, reading the option letters
  devices.py             #   --device / --require-gpu, the check of the providers actually used
  paths.py               #   where the model is (argument > environment > models/)
  tokenizer.py           #   pure-Python Qwen byte-level BPE
  protocol.py            #   prompt building and answer construction (a port of the reference)
  chat.py                #   chat template (no jinja2) and 3-D M-RoPE
  image.py               #   Pillow decoding + smart_resize + patching
  _regex_translate.py    #   translates tokenizer.json's regex for Python
  _uniprops.py           #   generated: Unicode range tables
export/                  # heavy dependencies (torch / transformers / onnx)
  prepare_model.py       #   0.8B: fetch the public ONNX (onnx-community)
  build_onnx_4b.py       #   4B: swap OneJev-4B's weights into the public Qwen3.5-4B ONNX graphs
  make_golden.py         #   makes artifacts/golden*.json
  gen_uniprops.py        #   regenerates _uniprops.py
release/
  make_release.py        # stage release files from models/ (splitting, SHA-256, manifests)
  licenses/              # licence texts shipped with the release (LICENSE-ONEJEV, LICENSE-QWEN)
  notes.md               # the release description
verify/                  # verification gates (run_all.py runs them all), bench.py
tests/                   # unit tests (no model needed)
artifacts/               # golden*.json (ground truth), report.md, sample images
models/                  # ONNX, tokenizer, settings (gitignored)
```

# Reference
- [OmniJev/OneJev-0.8B](https://huggingface.co/OmniJev/OneJev-0.8B), [OmniJev/OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B) (the original models, OmniJev Team), [OmniJev/OneJev](https://github.com/OmniJev/OneJev) (the reference implementation `qev`)
- [onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX) (the public 0.8B ONNX)
- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (the base models, Alibaba Cloud)

# License
OneJev-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
The model weights follow the licence of the original models ([OmniJev/OneJev-0.8B](https://huggingface.co/OmniJev/OneJev-0.8B) and [OmniJev/OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B), both stated as Apache-2.0 on their model cards; the base Qwen3.5 is Apache-2.0 too). The release ships both licence texts unchanged (`LICENSE-ONEJEV` = the LICENSE of OmniJev/OneJev, `LICENSE-QWEN` = the LICENSE of Qwen3.5).

**A note on the training data (a statement of fact, not legal advice).** The licence of OneJev's training data, [OmniJev/OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data), is "per source" (`license: other`) and includes sources marked research-only or non-commercial (for example agent_reward_bench, aqa7, parts of videofeedback, konvid1k, fgvc_aircraft, flickr30k; see the dataset's [licenses.csv](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/main/licenses.csv)). The model's provider licenses the model itself as Apache-2.0, but this repository does not judge whether weights trained on that data may be used commercially. If you use it commercially, check for yourself.

## About the converted files
The ONNX files in the release are OneJev's weights in ONNX form.
- **0.8B** (`onnx/`): the ONNX published as [onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX) (Apache-2.0), redistributed unchanged (`export/prepare_model.py` fetches the same files directly). That repository describes it as the graphs of [onnx-community/Qwen3.5-0.8B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT) (Apache-2.0) with OneJev-0.8B's weights put in. fp16 was checked against the original implementation (PyTorch fp32): 17/17 decisions agree, max logit difference 0.010, probability difference 0.002.
- **4B** (`onnx-4b/`): no public ONNX exists, so `export/build_onnx_4b.py` swaps OneJev-4B's weights into the graphs of the public Qwen3.5-4B ONNX (fp16; no torch export). The swap rules were first shown to reproduce the public ONNX of the original Qwen3.5-4B; against PyTorch fp32, 17/17 decisions agree (max logit difference 0.006). **The graphs come from [onnx-community/Qwen3.5-4B-ONNX](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX) (not the `-OPT` repository: judged from the shipped decoder, which has no `num_logits_to_keep` input and uses `GroupQueryAttention`), which declares no licence** (the base model Qwen/Qwen3.5-4B is Apache-2.0). Unlike the `-OPT` graphs used for 0.8B, these cannot project only the last position through the output head; the effect of that on 4B's speed has not been measured.
- Neither the providers of the original models, nor onnx-community, give any warranty for these converted files.
