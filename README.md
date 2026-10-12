[[Japanese](README.md)/[English](README_EN.md)]

# OneJev-ONNX-CPU

[`OmniJev/OneJev-0.8B`](https://huggingface.co/OmniJev/OneJev-0.8B) と [`OmniJev/OneJev-4B`](https://huggingface.co/OmniJev/OneJev-4B)（型付き判定モデル。テキストと画像）を、onnxruntime と numpy だけで推論する最小依存の推論実装です。 ※GPU（CUDA）推論も可<br>
参照実装（`qev`）が要求する torch・transformers・pydantic・fastapi などは使いません。

# Features
以下の特徴があります。
- 最小依存：実行時に必要なパッケージは onnxruntime と numpy の2つだけ（画像を使う場合は pillow を足した3つ）
- 動かすのに必要なスクリプトは `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）の2つ。画像は `demo_inference_image.py`、カメラは `demo_inference_webcam.py`
- 0.8B と 4B の2サイズを1つのリポジトリで扱う（`--model-size 0.8b|4b`）
- 判定は参照実装（PyTorch fp32）と一致（0.8B・4B とも 17/17。0.8B fp16 の最大ロジット差 0.010、4B は 0.006）
- 1問につき1回の順伝播で、生成は走らない。答えは必ず与えた選択肢のどれかになる（選択肢の文字のトークンで logit を読んで softmax）
- 画像対応（状態の中から `<image:1>`, `<image:2>`, … で参照）
- 前処理（Qwen byte-level BPE、チャットテンプレート、3次元 M-RoPE、画像のパッチ化）を純Pythonで移植し、参照実装との突き合わせ検証を同梱
- CPU のほか、GPU（CUDA）でも動かせる（`--device cuda`）。実際に使われたプロバイダを表示し、CPU への静かなフォールバックは警告（`--require-gpu` ならエラー）

# Purpose of This Repository
以下の検証を目的としています。
- OneJev の判定を、実行時依存2〜3つだけでどこまで動かせるか
- 参照実装の前処理を純Pythonに移植して、同じ入力が作れるか
- CPU（と GPU）で速度・メモリがどうなるか

実測値と移植の根拠は[artifacts/report.md](artifacts/report.md)を参照してください（0.8B・CPU のみの作業時点の記録です）。

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later   （GPU で動かすときは onnxruntime-gpu。両方は入れない。後述）
pillow       10.0 or later   （画像を使う場合のみ）
```
実行に必要なのはこれだけです。モデルの取得も標準ライブラリのみで動きます。<br>
環境構築は[uv](https://docs.astral.sh/uv/)を使います。<br>
オプションのグループ・extra があります。
- `cpu`（既定のグループ）：onnxruntime（CPU 版）。`uv sync` で入ります
- `gpu`（extra）：onnxruntime-gpu。`cpu` とは同時に入れられません（後述「GPUで実行する」）
- `test`：`tests/` 用の pytest
- `verify`：トークナイザの突き合わせ検証（`check_tokenizer_fuzz`）用の `tokenizers`
- `webcam`：`demo_inference_webcam.py` 用の OpenCV
- `export`：ONNX を自分で書き出すときや、ゴールデンを作り直すときの torch・transformers など（実行時には使いません）

# Installation

```bash
git clone https://github.com/Kazuhito00/OneJev-ONNX-CPU
cd OneJev-ONNX-CPU

# 実行時の依存（onnxruntime・numpy・pillow）のみインストール
uv sync
```

# Download Model
ONNX は[Releases](https://github.com/Kazuhito00/OneJev-ONNX-CPU/releases)に置いています（リポジトリには含めていません）。<br>
以下のスクリプトで `models/` に取得します。標準ライブラリのHTTPSのみで動き、gitもgit-lfsも不要です。
```bash
uv run download_model.py                       # 0.8B（fp16 と q4f16、トークナイザ、設定。約 2.9 GB）
uv run download_model.py --model-size 0.8b     # 0.8B だけ（約 2.9 GB）
uv run download_model.py --model-size 4b       # 4B だけ（fp16 のみ。約 10.4 GB）
```
- **引数なしでは 0.8B と 4B の全ファイルを取得します**（約 13 GB。0.8B は 21 ファイル・約 2.9 GB）。4B は動かすのに RAM も多く要るため、ディスクを節約したい場合は `--model-size 0.8b` を付けてください
- 取得元は、既定で `Kazuhito00/OneJev-ONNX-CPU` の最新の Release です（`--repo OWNER/NAME` か、環境変数 `ONEJEV_ONNX_REPO` で別のリポジトリにできます。優先順位は 引数 > 環境変数 > 既定）
- 0.8B は `models/onnx`、4B は `models/onnx-4b` に置かれます
- 取得したファイルは、Release の `manifest-0.8b.json` / `manifest-4b.json` に書かれた SHA-256 と照合します（不一致ならファイルを削除して停止）
- GitHub の制限（1ファイル 2 GiB 未満）で分割されたファイルは、取得後に結合して照合します（現状の Release は最大のファイルが 2,095,788,032 バイト = 1.95 GiB なので分割はありません）
- 取得済みなら何もせず、中断した場合は同じコマンドで続きから再開します

```bash
uv run download_model.py --list                             # ファイル一覧
uv run download_model.py --variant fp16                     # q4f16 を取らない（約 2.1 GB）
uv run download_model.py --exclude "vision_encoder*"        # テキスト専用（視覚エンコーダを取らない）
uv run download_model.py --tag v0.0.0                       # タグを指定（既定は latest）
```

Release のアセット（`release/make_release.py` が作ります。一覧は `uv run release/make_release.py --plan`。Release は v0.0.0 を想定、まだアップロードしていません）:

| 対象 | ファイル（アセット名は `/` を `__` にしたもの） | サイズ |
|---|---|---:|
| 0.8B fp16 | `onnx__onnx__decoder_model_merged_fp16.onnx` + `.onnx_data` | 600,469 B + 1,504,886,784 B |
| | `onnx__onnx__embed_tokens_fp16.onnx` + `.onnx_data` | 637 B + 508,559,360 B |
| | `onnx__onnx__vision_encoder_fp16.onnx` + `.onnx_data` | 187,476 B + 204,798,464 B |
| 0.8B q4f16 | `onnx__onnx__{decoder_model_merged,embed_tokens,vision_encoder}_q4f16.onnx` + `.onnx_data` | 合計 約 0.65 GB |
| 0.8B 共通 | `onnx__tokenizer.json`（約 20 MB）、`onnx__config.json`、`onnx__preprocessor_config.json` など、`onnx__LICENSE-ONEJEV`、`onnx__LICENSE-QWEN`、`manifest-0.8b.json`、`SHA256SUMS-0.8b` | 約 20 MB |
| 4B fp16 | `onnx-4b__onnx__decoder_model_merged_fp16.onnx`（1.3 MB）+ `.onnx_data`、`_data_1` … `_data_4` | 2,095,788,032 / 2,058,608,640 / 2,094,993,408 / 892,618,752 / 1,271,398,400 B |
| | `onnx-4b__onnx__embed_tokens_fp16.onnx` + `.onnx_data` | 1,271,398,400 B |
| | `onnx-4b__onnx__vision_encoder_fp16.onnx` + `.onnx_data` | 671,943,680 B |
| 4B 共通 | `onnx-4b__tokenizer.json` など、`onnx-4b__LICENSE-*`、`manifest-4b.json`、`SHA256SUMS-4b` | 約 20 MB |

自分で用意する場合：
```bash
# 0.8B: 公開されている onnx-community/OneJev-0.8B-ONNX を Hugging Face から取得（Release の 0.8B と同じファイル）
uv run export/prepare_model.py
# 4B: 公開 ONNX は無いので、Qwen3.5-4B の公開 ONNX のグラフに OneJev-4B の重みを差し替えて作る（入力の置き場所は export/build_onnx_4b.py の冒頭）
uv sync --group export
uv run export/build_onnx_4b.py verify    # 差し替え規則が、元の Qwen3.5-4B の公開 ONNX を再現できるかを先に確認
uv run export/build_onnx_4b.py build
```

# Usage

### 実行例(Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
uv run demo_inference_text.py --model-size 4b       # 4B（$ONEJEV4_MODEL_DIR か models/onnx-4b）
```
モデルの場所は、引数 `--model` > 環境変数（`ONEJEV_MODEL_DIR`＝0.8B、`ONEJEV4_MODEL_DIR`＝4B）> `models/onnx`（4B は `models/onnx-4b`）の順で決まります。<br>
自分の入力は `--state` と `--choice` / `--score` / `--noul` で指定します。
```bash
uv run demo_inference_text.py --state "I was charged twice." --choice "team=billing,technical" --noul "angry=The customer is angry"
uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
```
画像のデモは、既定で描いた円の画像（`artifacts/images/demo_circle.png`。実行時に生成）を使います。画像は状態の中から `<image:1>`, `<image:2>`, … で参照します（各画像を1回ずつ、順番どおり）。

```python
from onejev_decide import Decider, choice, noul, score

d = Decider("models/onnx", variant="fp16")        # 4B は "models/onnx-4b"。テキストのみなら load_vision=False

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
参照実装と同じ System One 形式のリクエスト（`d.predict({"state": ..., "media": [{"type": "image", "path": ...}], "questions": {...}})`）も受け付けます。

**1問につき1回の順伝播で決まります。** プロンプトは assistant ターンで終わり、次トークンの logit を選択肢の文字（`A`, `B`, …）のトークンで読んで softmax します。生成は走りません。`noul` は `yes` が A（回答は p(yes)）、`score` は各レベルを並べた期待値、選択肢が 27 個以上なら 2 文字ラベルを使います（最大 255）。質問ごとに全体をプレフィルするので、時間は問い数にほぼ比例します（参照実装のような共通部分の KV キャッシュ再利用は未実装です）。視覚エンコーダは1リクエストにつき1回だけ実行して使い回します。

### カメラで「人が映っているか」を判定する
OpenCV でカメラの映像を取得し、最新のフレームについて「人が映っているか」を判定して、結果を映像に重ねて表示します。
```bash
uv run --group webcam demo_inference_webcam.py                        # カメラ 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/circle.png   # カメラの代わりに画像・動画ファイル
```
- 判定は別スレッドで動かし、映像の表示は止めません。判定が出るたびに、上部の帯（緑＝はい／赤＝いいえ）と「はい」の確率が更新されます
- 質問は `--question`（はい・いいえの質問）で変えられます。モデルに渡す画像は長辺 224 px に縮小します（`--size`）
- `q` または `Esc` で終了します。OpenCV は、このデモだけの依存です。`--no-window` ではウィンドウを開かずに判定結果を表示します

### 質問の種類
| `type` | 答え |
|---|---|
| `choice` | 選択肢から1つ（`choice`、`confidence`、`probabilities`） |
| `score` | 順序尺度の期待値（`score`、`legend`）。先頭が最低（最大 10 段階） |
| `noul` | 文が成り立つか（`noul` = はいの確率） |

### モデルの選び方
| | サイズ | 備考 |
|---|---:|---|
| 0.8B `fp16` | 約 2.1 GB | **CPU の既定。最速**。17/17 一致、logit 差 ≤ 0.010 |
| 0.8B `q4f16` | 約 0.65 GB | 小さいが CPU では fp16 より遅く、少し劣化する（17/17 一致だが logit 差 ≤ 1.29、明確な 12 問で 11/12） |
| 4B `fp16` | 約 10.4 GB | 精度は高いが重い。RAM を多く使う（`--model-size 4b`）。17/17 一致、最大ロジット差 0.006 |

`--variant q4f16` で 0.8B の q4f16 を選べます。fp32 の ONNX は用意していません（デコーダが 3.0 GB の単一ファイルになり、Release の1ファイル 2 GiB 制限も超えるため）。

### GPUで実行する（任意）
`--device cuda`（または `auto`）を付けると、CUDA で推論します。何も付けなければ CPU だけです（既定は `--device cpu`）。

| オプション | 動作 |
|---|---|
| `--device cpu`（既定） | CPU だけ |
| `--device cuda` | `CUDAExecutionProvider` → `CPUExecutionProvider` の順。この onnxruntime に CUDA が無い、または CUDA が読み込めなかったときは、**警告を出して CPU で動きます** |
| `--device auto` | CUDA があれば CUDA、無ければ CPU |
| `--require-gpu` | 全グラフが本当に CUDA で動くことを要求します。CUDA が使えなければ**エラーで止まります**（CPU に静かに切り替わって GPU のつもりになる事故の防止。`--device cuda` を含意） |

**実際に使われたプロバイダは、デモの最終行（`providers actually used: embed=CUDA, vision=CUDA, decoder=CUDA`）と `verify/bench.py` の出力（`providers_used`）に表示されます。** onnxruntime は、CUDA を用意できなくても例外を出さずに CPU でセッションを作るので、指定しただけでは GPU で動いた証拠になりません。

```bash
# 1. GPU 版に入れ替える（cpu グループを外し、gpu extra を入れる）
uv sync --extra gpu --no-group cpu
# 2. 実行（uv run は既定の cpu グループに戻してしまうので --no-sync を付ける。UV_NO_SYNC=1 でも同じ）
uv run --no-sync demo_inference_text.py --device cuda --require-gpu
uv run --no-sync verify/bench.py --device cuda --require-gpu
```
```python
from onejev_decide import Decider, select_providers

d = Decider("models/onnx", providers=select_providers("cuda"), require_gpu=True)   # CUDA でなければ GpuUnavailableError
print(d.providers_used)   # {"embed": "CUDAExecutionProvider", "vision": ..., "decoder": ...}
```

要件と注意点です。
- **`onnxruntime` と `onnxruntime-gpu` は同時に入れられません**（どちらも同じ `onnxruntime/` ディレクトリに書き込むため）。`pyproject.toml` は、CPU 版を `cpu` グループ、GPU 版を `gpu` extra にして、両方を同時に解決できないように宣言しています。そのため `uv sync --extra gpu` だけだと（既定で `cpu` グループも入ろうとして）エラーになります。必ず `--no-group cpu` を付けます。CPU に戻すときは `uv sync` です
- **CUDA 13 と cuDNN 9 が必要です**（`onnxruntime-gpu` 1.31 の場合）。両方の `bin` フォルダを PATH に入れてください。`onnxruntime-gpu` は CUDA の DLL を同梱せず、PATH から探します（`CUDA_PATH` が古い CUDA を指していることがあります）。見つからないと `cublasLt64_13.dll` が無いという警告が出て、CPU で動きます（`--require-gpu` ならエラーになります）。Windows の例です（バージョンは環境に合わせてください）。PATH を変えたあとは、ターミナルを開き直します
  ```text
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
  C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
  ```
- GPU での確率は、CPU と完全には一致しません（演算の順序の違い。下の測定では最大差 2.3e-03）。参照実装との一致検証（`verify/`）は CPU のみです
- DirectML（`onnxruntime-directml`）は、このモデルでは試していません（**未検証**）。q4f16 を GPU で動かした測定もありません（**未検証**）

**VRAM の目安**（RTX 3050 Ti Laptop 4 GB での測定。`nvidia-smi` の増分）：0.8B fp16 はテキストで約 3.2 GiB、画像で約 3.7 GiB（4 GB に収まります）。**4B fp16 は重みが約 8 GB あり 4 GB に載りません**。CUDA のセッションは作れてしまいますが、1問 15〜19 秒で、CPU（約 6.5 s）より遅くなります（載りきらずにセッションが成立する理由は未調査です。`--require-gpu` でもこれは検出できません。VRAM が小さい GPU で 4B を使うときは CPU を選んでください）。

# 性能と精度
測定は、この作業の PC（Core i7-12800H、RTX 3050 Ti Laptop 4 GB、CPU は 8 スレッド）で、224 px の画像、fp16 です。**測定時のマシンの状態で数値は大きく変わります。** 下の数値は、既存の測定結果のうち、根拠（出典）があるものだけを載せています。

### 1問あたりの時間（出典: `wikipedia-vision-decision-bench` の README の比較表。Commons 1268 問の中央値、較正が合ったブロックのみ）
| モデル | CPU（8 スレッド） | GPU（RTX 3050 Ti Laptop 4GB、CUDA） |
|---|---:|---:|
| OneJev 0.8B fp16 | 1.2 s | 87 ms |
| OneJev 4B fp16 | 6.5〜7.1 s | 未測定（下の別測定では CPU より遅い） |

### GPU の測定（出典: `eval_common/gpu_bench/report.md`、2026-10-10、onnxruntime-gpu 1.31.0、CUDA 13.4、cuDNN 9.20、CPU 8 スレッド）
問題は固定の3問ずつ（テキスト3問、画像3問・224 px）。「1問」は全計測の中央値です。

| モデル | 入力 | CPU 1問 | CUDA 1問 | 倍率 | ピーク VRAM | CPU との判定一致（最大確率差） |
|---|---|---:|---:|---:|---:|---|
| 0.8B fp16 | text | 830 ms | 66 ms | 12.6 倍 | 3192 MiB | 3/3（5.8e-04） |
| 0.8B fp16 | image | 1277 ms | 80 ms | 16.0 倍 | 3722 MiB | 3/3（2.3e-03） |
| 4B fp16 | text | 6456 ms | 19172 ms | 0.3 倍（遅い） | 3820 MiB（頭打ち） | 3/3（5.4e-05） |
| 4B fp16 | image | 未測定（CPU 側が RAM 不足で中断） | 15260 ms | — | 3820 MiB | — |

**0.8B は GPU で 12〜16 倍速くなります。4B は 4 GB の GPU では逆に遅くなります**（VRAM 不足。VRAM の大きい GPU での測定はしていません。**未検証**）。

### 判定の正答率（%。出典: 同 bench の README。Commons 1268／synthetic 382／reasoning 225 問、CPU）
| モデル | Commons 英 | Commons 日 | synthetic 英 | synthetic 日 | reasoning 英 | reasoning 日 |
|---|---:|---:|---:|---:|---:|---:|
| OneJev 0.8B | 93.0 | 87.0 | 91.6 | 89.0 | 45.8 | 47.6 |
| OneJev 4B | 97.9 | 96.8 | 95.3 | 95.0 | 68.9 | 67.6 |

reasoning は選択肢が少なく当てずっぽうでも当たりやすい形式を含むため、偶然の正答率を意識して読んでください。

### メモリ
0.8B は fp16 でピーク RAM 約 3.9 GB、q4f16 で約 0.85 GB（作業時の測定）。4B は重みだけで約 10 GB で、bench のテキスト測定ではピーク RAM が 18.6 GB でした（測定中は他のプロセスも RAM を使っていました）。

速度の測定コマンド：`uv run verify/bench.py --model-size 0.8b`（1プロセス1測定。`--questions N`、`--image 224`、`--device cuda`）。

# Verification
```bash
uv run verify/run_all.py                                 # 実行時依存だけで通るゲート（0.8B）
uv run --group verify verify/run_all.py --full-fuzz      # tokenizers があればファズも本気で回す
uv run verify/run_all.py --model-size 4b                 # 4B（RAM を約 10 GB 使います）
uv run --group test pytest                               # モデル不要の単体テスト（GPU も不要）
```

| ゲート | 内容 | 必要なもの |
|---|---|---|
| check_prompt | プロンプト文字列と展開後 `input_ids` が参照実装と一致（9/9） | 実行時依存のみ |
| check_image | 前処理が processor の `pixel_values` と画素まで一致 | + pillow |
| verify | 選択肢の logit と型付き回答が PyTorch(fp32) と一致（17 問） | 実行時依存のみ |
| eval_decisions | 明確な 12 問（0.8B fp16 で 12/12） | 実行時依存のみ |
| check_bias | 選択肢を入れ替えても答えが変わらないか（4/4） | 実行時依存のみ |
| check_tokenizer_fuzz | Rust `tokenizers` との突合（`--full-fuzz` で 0.8B の `tokenizer.json` を 187,689 ケース、不一致 0） | + tokenizers |

`artifacts/golden.json`（0.8B）と `golden_4b.json`（4B）は、参照実装（`qev`、PyTorch fp32、CPU）を実際に動かして作った正解データです。同じ重みを両者が走らせるので、差が出たら移植のバグと断定できます。`artifacts/golden_pixels.npz` は、このリポジトリに入れた3枚の画像（`artifacts/images/`）について processor が作った `pixel_values` です。

`tests/` には、モデルも GPU も要らない単体テストがあります（プロバイダの選択ロジック、モデルの場所の決め方、プロンプトと回答の組み立て、トークナイザ、`download_model.py` と `release/make_release.py` のローカルでの往復）。

# Limitations
- 1問1プレフィル（共通部分の KV キャッシュ再利用は未実装）。質問が多いと、その分遅くなります
- 静止画のみ。動画（`<video:N>`）は未対応です
- 画像の URL 取得には対応しません（パスまたは data URI）
- `tokenizer.py` / `chat.py` はこのチェックポイント専用で、想定外の入力は例外で弾きます
- 4B は重く、4 GB の GPU では載りません。q4f16 は 0.8B のみです
- GPU では確率が CPU と完全には一致しません

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # Apache-2.0
pyproject.toml           # 依存定義（実行時は onnxruntime・numpy・pillow。グループ: cpu / test / verify / webcam / export、extra: gpu）
uv.lock
download_model.py        # モデル取得（GitHub Releases、SHA-256照合、0.8B / 4B の選択）
demo_inference_text.py   # テキストの推論デモ
demo_inference_image.py  # 画像の推論デモ
demo_inference_webcam.py # カメラで「人が映っているか」を判定（OpenCV）
onejev_decide/           # 推論実装（軽い依存のみ。export/ を import しない）
  runtime.py             #   ORT セッション、プレフィル、選択肢の文字での読み出し
  devices.py             #   --device / --require-gpu、実際に使われたプロバイダの検査
  paths.py               #   モデルの場所（引数 > 環境変数 > models/）
  tokenizer.py           #   Qwen byte-level BPE の純Python実装
  protocol.py            #   プロンプトの組み立てと回答の構築（参照実装の移植）
  chat.py                #   チャットテンプレート（jinja2 なし）と3次元 M-RoPE
  image.py               #   Pillow デコード + smart_resize + パッチ化
  _regex_translate.py    #   tokenizer.json の正規表現を Python 用に翻訳
  _uniprops.py           #   生成物: Unicode範囲表
export/                  # 重い依存（torch / transformers / onnx）
  prepare_model.py       #   0.8B: 公開 ONNX（onnx-community）を取得
  build_onnx_4b.py       #   4B: Qwen3.5-4B の公開 ONNX のグラフに OneJev-4B の重みを差し替える
  make_golden.py         #   artifacts/golden*.json を作る
  gen_uniprops.py        #   _uniprops.py の再生成
release/
  make_release.py        # models/ から Release 用のファイルを作る（分割・SHA-256・manifest）
  licenses/              # Release に同梱するライセンス文（LICENSE-ONEJEV、LICENSE-QWEN）
  notes.md               # Release の説明文
verify/                  # 検証ゲート（run_all.pyで一括）、bench.py
tests/                   # 単体テスト（モデル不要）
artifacts/               # golden*.json（検証の正解データ）、report.md、サンプル画像
models/                  # ONNX・トークナイザ・設定（gitignore）
```

# Reference
- [OmniJev/OneJev-0.8B](https://huggingface.co/OmniJev/OneJev-0.8B)、[OmniJev/OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B)（元のモデル。OmniJev Team）、[OmniJev/OneJev](https://github.com/OmniJev/OneJev)（参照実装 `qev`）
- [onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX)（0.8B の公開 ONNX）
- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)（元のモデルのベース。Alibaba Cloud）

# License
OneJev-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
モデルの重みは、元のモデル（[OmniJev/OneJev-0.8B](https://huggingface.co/OmniJev/OneJev-0.8B)、[OmniJev/OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B)。どちらもモデルカードに Apache-2.0 と記載。ベースの Qwen3.5 も Apache-2.0）のライセンスに従います。Release には、両方のライセンス文（`LICENSE-ONEJEV`＝OmniJev/OneJev の LICENSE、`LICENSE-QWEN`＝Qwen3.5 の LICENSE）を元のまま同梱しています。

**学習データについての注記（事実の記載であり、法的な判断ではありません）。** OneJev の学習データ [OmniJev/OneJev-Data](https://huggingface.co/datasets/OmniJev/OneJev-Data) のライセンスは「出典ごと」（`license: other`）で、研究用途のみ・非商用とされた出典（例：agent_reward_bench、aqa7、videofeedback の一部、konvid1k、fgvc_aircraft、flickr30k など。詳細は同データセットの [licenses.csv](https://huggingface.co/datasets/OmniJev/OneJev-Data/blob/main/licenses.csv)）を含みます。元のモデルの提供元はモデル自体を Apache-2.0 としていますが、これらのデータで学習した重みを商用に使えるかどうかは、このリポジトリでは判断していません。商用で使う場合は、ご自身で確認してください。

## 変換物であることについて
Release に含まれる ONNX ファイルは、OneJev の重みを ONNX 形式にしたものです。
- **0.8B**（`onnx/`）：[onnx-community/OneJev-0.8B-ONNX](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX)（Apache-2.0）で公開されている ONNX を、変更せずに再配布しています（`export/prepare_model.py` で同じファイルを直接取得できます）。同リポジトリは、[onnx-community/Qwen3.5-0.8B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT)（Apache-2.0）のグラフに OneJev-0.8B の重みを入れたものと説明しています。fp16 は元の実装（PyTorch fp32）との判定の一致を確認しています（17/17、最大ロジット差 0.010、確率差 0.002）。
- **4B**（`onnx-4b/`）：公開されている ONNX が無いため、`export/build_onnx_4b.py` で、Qwen3.5-4B の公開 ONNX のグラフに OneJev-4B の重みを差し替えて作りました（fp16。torch での書き出しはしていません）。差し替えの規則は、元の Qwen3.5-4B の公開 ONNX を再現できることを先に確認し、PyTorch fp32 との比較で 17/17 判定が一致しました（最大ロジット差 0.006）。**グラフの出所は [onnx-community/Qwen3.5-4B-ONNX](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX)（`-OPT` ではない版。出来上がったデコーダに `num_logits_to_keep` 入力が無く `GroupQueryAttention` を使っていることから判断）で、このリポジトリはライセンスを宣言していません**（ベースモデルの Qwen/Qwen3.5-4B は Apache-2.0 です）。0.8B が使う `-OPT` 系のグラフと違い、最後の位置だけ語彙射影する機能が無いので、4B は同じ条件の 0.8B より相対的に遅くなります（この点の速度への影響は測っていません）。
- 元のモデルの提供元、onnx-community、および、この変換物の動作を、提供元は保証していません。
