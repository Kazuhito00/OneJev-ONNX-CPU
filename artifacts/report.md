# OneJev の ONNX 推論を依存最小限で動かす — 検証結果

> **この文書について（このリポジトリへの移行時の注記）**: 2026-10-06 の作業時点（0.8B・CPU のみ）の検証記録をそのまま残している。
> その後の変更: 4B（`models/onnx-4b`、重み差し替え、検証 17/17 一致・最大ロジット差 0.006）と GPU(CUDA) 対応を追加し、
> 取得は GitHub Release（`download_model.py`）に変わった。§5 の速度は**他プロセスで 2.85 倍遅い状態**で測った参考値で、
> README には載せていない（README の速度は bench の測定を出典にしている）。§7 の「ONNX は 0.8B のみ」は公開 ONNX
> （onnx-community）についての記述で、このリポジトリの Release には 4B もある。§8 の `pip` / `python` は、
> このリポジトリでは `uv sync` / `uv run` に読み替える（README 参照）。

計測: 2026-10-06 / Windows 11 / Intel Core i7-12800H / Python 3.12 (uv) / CPU のみ
対象: [`OmniJev/OneJev-0.8B`](https://huggingface.co/OmniJev/OneJev-0.8B)
（Qwen3.5-0.8B の全パラメータ微調整、vision tower 固定、Apache-2.0）
ONNX: [`onnx-community/OneJev-0.8B-ONNX`](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX)

## 結論

**`onnxruntime` + `numpy` + `pillow` の 3 パッケージで、画像込みの型付き判定が動く。**
テキストだけなら 2 パッケージ。

- 参照実装（PyTorch / `qev`、fp32）との照合で、プロンプトとトークン列は **9/9 完全一致**、
  画像前処理は**画素まで一致**、ONNX fp16 の出力は **logit 差 ≤ 0.010、確率差 ≤ 0.002**
- **判定品質が高い。12 問セットで 12/12、選択肢の順序入れ替えにも 4/4 で頑健**。
  同じ 0.8B の OpenJev（汎用 Qwen3.5-0.8B）は 8/12 と 3/4、4B の汎用モデルでも 11/12 だった
- **OpenJev より速い**（同条件の交互測定で text 1.4 倍、image 1.9 倍）。
  公開 ONNX が `num_logits_to_keep` 入力を持つため
- ONNX は公開版をダウンロードするだけで使える（自前変換は不要）

## 1. 依存パッケージ

| 構成 | パッケージ数 | サイズ |
|---|---:|---:|
| 参照実装 `qev`（torch / transformers / pydantic / fastapi / jinja2 …） | 54 | 850 MB |
| **この実装 `.venv-min`** | **3** | **101 MB** |

`.venv-min` で不在を確認済み:

```
torch         ModuleNotFoundError: No module named 'torch'
tokenizers    ModuleNotFoundError: No module named 'tokenizers'
transformers  ModuleNotFoundError: No module named 'transformers'
jinja2        ModuleNotFoundError: No module named 'jinja2'
onnx          ModuleNotFoundError: No module named 'onnx'
regex         ModuleNotFoundError: No module named 'regex'
pydantic      ModuleNotFoundError: No module named 'pydantic'
```

`jinja2` はチャットテンプレート、`regex` は Qwen の分割正規表現の `\p{L}` 等のために
普通なら必要になるもので、どちらも自前実装で外した。`pydantic` は参照実装の
リクエスト/レスポンス型だが、ここでは素の dict で足りる。

## 2. プロトコル（参照実装の移植）

`qev/prompt.py`・`qev/answers.py`・`qev/media.py` を移植した。
チェックポイントは**この文字列そのもの**で学習されているので、定数はすべて
参照実装から取り、再導出していない。

```
<system> Apply the question to the state. Choose exactly one of the listed options. ...
<user>   <state>\n...state...\n</state>\n\nQuestion: ...\n\nOptions:\nA. ...\nB. ...\n\nAnswer with one letter: A, B.
<assistant> <think>\n\n</think>\n\n          ← 次のトークンが答えの文字
```

- 答えは最後の位置の logit を**選択肢の文字の token id** で読んで softmax
- `noul` は `["yes", "no"]` の順で、回答は p(yes)（スロット 0）
- `score` は `level 0..N`、回答は期待値と信頼度
- `choice` の信頼度は「一様から確実性への正規化」: `(max p − 1/k) / (1 − 1/k)`
- ラベルは `A..Z`、27 個以上は `AA`, `AB`, …（単一トークンにならない 25 通りを除く）
- 画像は状態の中から `<image:N>` で参照し、各画像を 1 回ずつ・順番どおりに
- 温度 1.0、順序の並べ替え平均なし（参照実装の既定。`calibration.json` は配布されていない）

## 3. 検証

### 3.1 参照実装の出力（ゲート）

`export/make_golden.py` が**参照実装（`qev`, PyTorch fp32, CPU）を実際に動かして**
`artifacts/golden.json` を作る。9 ケース・17 問（テキスト 5 / 画像 4。JSON 状態・日本語・
30 択・2 枚画像を含む）。質問ごとにプロンプト文字列・展開後 `input_ids`・
選択肢の生 logit・最終回答を記録する。

### 3.2 プロンプトと入力列（`check_prompt.py`）

```
ok  support_json       3 questions
ok  bug_text           2 questions
ok  incident           3 questions
ok  japanese           2 questions
ok  many_options       1 questions
ok  image_colour_red   1 questions  (1 image)
ok  image_colour_blue  1 questions  (1 image)
ok  image_shape        2 questions  (1 image)
ok  two_images         2 questions  (2 images)
failures: 0 / 9
```

プロンプト文字列と、画像ブロックを展開した後の `input_ids` の両方が一致。
2 枚画像（M-RoPE の位置計算）も含む。

### 3.3 画像前処理（`check_image.py`）

processor が実際に作った `pixel_values` と突き合わせ:

```
ok  image_colour_red   grid=[[1, 16, 16]]               shape=OK  max|dp|=0.0000
ok  image_shape        grid=[[1, 20, 14]]               shape=OK  max|dp|=0.0000
ok  two_images         grid=[[1, 16, 16], [1, 16, 16]]  shape=OK  max|dp|=0.0000
```

### 3.4 ONNX の出力（`verify.py`）

```
variant=fp16   decisions 17/17 agree (1 near-tie)   worst |dlogit| 0.010   worst |dp| 0.0021
variant=q4f16  decisions 17/17 agree (2 near-tie)   worst |dlogit| 1.286   worst |dp| 0.1969
```

- **fp16 は PyTorch fp32 とほぼ同じ**（logit 差 0.010）
- **q4f16 は数値上は明確に劣化する**（logit 差 1.29、確率差 0.20）。ただし判定は全部一致し、
  12 問セットも 11/12 で保たれる。境界付近のケースが 2 件ある
- 「near-tie」は、参照の上位 2 択の確率差が許容誤差内（例: p(yes)=0.498）で、
  ラベルが丸めで決まるケース。**黙って緩めず、引き分けと明示して数えている**
- fp32 の ONNX は未検証（デコーダが 3.0 GB の単一ファイルで 4.4 GB 級のため）。
  基準は PyTorch fp32 なので、fp16 の誤差が 0.010 に収まることで十分と判断した

### 3.5 トークナイザ（`check_tokenizer_fuzz.py`）

```
cases: 187689   mismatches: 0
```

#### 初回はバグがあった（ファズが捕まえた）

最初は OpenJev から `tokenizer.py` をコピーして使ったが、**5189 件中 54 件が不一致**だった:

```
'हिन्दी'  got: ['हि', 'न्', 'दी']     expect: ['ह', 'िन', '्', 'द', 'ी']
'+c️'     got: ['+c', '️']            expect: ['+', 'c', '️']
```

原因は分割正規表現の違い。OpenJev の `tokenizer.py` は Qwen3.5 の
`[\p{L}\p{M}]+`（結合文字を文字として扱う）を**ハードコード**していたが、
OneJev の `tokenizer.json` は Qwen3 系の `\p{L}+`（結合文字を含まない）だった。

| 系統 | 分割パターンの語クラス |
|---|---|
| OpenJev / Intern（Qwen3.5） | `[\p{L}\p{M}]+` |
| **OneJev**・kev（Qwen3） | `\p{L}+` |

放置すると、結合文字・絵文字の異体字セレクタ・インド系文字を含む入力で
**黙って誤ったトークン列がモデルに渡り続けた**（代表ケースでは通る）。

kev で作った「`tokenizer.json` が宣言するパターンをその場で Python 用に翻訳する」
汎用版（`_regex_translate.py`）に差し替えて解消した。

## 4. 判定品質

明確に答えが定まる 12 問と、選択肢の順序を入れ替える 4 件
（`eval_decisions.py` / `check_bias.py`）。他モデルと同じ問題セット。

| モデル | 12 問 | 順序に依存しない（かつ正解） |
|---|---:|---:|
| **OneJev-0.8B（fp16）** | **12/12** | **4/4** |
| OneJev-0.8B（q4f16） | 11/12 | 4/4 |
| OpenJev（Qwen3.5-4B, q4f16） | 11/12 | 4/4 |
| OpenJev（Qwen3.5-0.8B, fp16） | 8/12 | 3/4 |

**同じ 0.8B で 8/12 → 12/12。** OpenJev の 0.8B で目立った選択肢の位置バイアス
（先頭確率 0.504 などコイン投げに近い）が消え、確率が順序を入れ替えるとほぼ鏡像になる:

```
Thanks so much for the quick fix ...   A-first -> calm  | B-first -> calm     p(first) 0.026 ↔ 0.967
This is the third time this week ...   A-first -> angry | B-first -> angry    p(first) 0.881 ↔ 0.065
```

確信度も 0.53〜0.99 と分離している。**微調整の有無が品質を決めている**ことの裏付けで、
4B の汎用モデルより良い。

## 5. 性能

この PC は測定時、他プロセスの影響で**基準（GLiNER fp32 4 スレッド = 199 ms）の
2.85 倍遅い状態**で安定していた（562 / 562 / 568 ms）。
絶対値は 2.85 倍に膨らんでいるので、**同条件で交互に測った比**を使う。

### OneJev と OpenJev（Qwen3.5-0.8B）の比較（fp16, 8 スレッド, 3 回/2 回の交互測定）

| | OneJev | OpenJev | 比 |
|---|---:|---:|---:|
| text 1 問 | 1843 / 1880 / 1849 ms | 2544 / 2655 / 2671 ms | **1.4 倍速い** |
| image 1 問（448px） | 6420 / 5838 ms | 12452 / 11027 ms | **1.9 倍速い** |

静かな状態に換算（÷2.85）すると **text 1 問 約 650 ms、image 1 問 約 2 秒**。

### なぜ速いのか

公開 ONNX のデコーダが **`num_logits_to_keep` 入力**を持つ。

```
IN   num_logits_to_keep   tensor(int64)   []
OUT  logits               tensor(float16) [batch, num_logits_to_keep, 248320]
```

答えは最後の位置にしか無いので `1` を渡すと、語彙射影（1024 × 248320）が
最後の 1 位置だけで済む。OpenJev が使う公開グラフには無く、**全位置**の
`[S, 248320]` を計算している。プロンプトが長い（画像で約 700 トークン）ほど差が開く。

これは `onnx-community/Qwen3.5-0.8B-ONNX-OPT` 由来のグラフ
（融合 `LinearAttention` / `CausalConvWithState` 演算）。ORT 1.30 の CPU で
そのまま動いた。**状態入力としてゼロ長を渡すと「0 件」になってしまうので、
`num_logits_to_keep` だけは状態入力と区別して明示的に 1 を渡している。**

### その他

| 項目 | fp16 | q4f16 |
|---|---:|---:|
| text 1 問 | 1850 ms | 2047 ms |
| ピーク RAM | 3.9 GB | **0.85 GB** |
| セッションロード | 約 12 s | **4.3 s** |

- 8 スレッド 1850 ms に対し 4 スレッド 1749 ms で、0.8B は 4 スレッドで十分
  （P/E ハイブリッド構成では、これまでのモデルと同じ傾向）
- 質問は 1 問ごとに 1 プレフィルなので問い数に比例（3 問 = 5562 ms ≒ 1850 × 3）
- vision encoder は 1 リクエストにつき 1 回だけ実行して使い回す
  （最初の実装は質問ごとに再実行しており、3 問の画像で 11.1 s → 7.6 s 改善した）
- **q4f16 は CPU では fp16 より遅く、精度も少し落ちる**。12 問セットは 11/12
  （外したのは 0.487 の僅差の 1 問）、順序バイアスは 4/4 のままで、実用上の劣化は
  小さい。容量・RAM（0.85 GB）が主目的のときだけ選ぶ

## 6. 実装

`onejev_decide/` — 実行時依存は `onnxruntime` / `numpy`（画像時 + `pillow`）。

| ファイル | 役割 |
|---|---|
| `protocol.py` | 質問の描画・文字ラベル・型付き回答の構築（参照実装の移植） |
| `runtime.py` | 3 つの ORT セッション、プレフィル、選択肢の文字での読み出し |
| `tokenizer.py` | Qwen byte-level BPE（`tokenizer.json` の正規表現を翻訳して使う汎用版） |
| `_regex_translate.py` | Rust の `\p{L}` `\p{N}` `\p{M}` `\s` を Python の `re` 用に展開 |
| `chat.py` | チャットテンプレート（jinja2 なし）と 3 次元 M-RoPE |
| `image.py` | Pillow デコード + `smart_resize` + パッチ化 |
| `_uniprops.py` | 生成物: Unicode 範囲表 |

デコーダの state 入力（`past_conv.*` `past_recurrent.*` `past_key_values.*`）は、
形をグラフから読んで空で与える。OneJev の `past_conv` は
`[B, 6144, 3]`（OpenJev 用グラフの `[B, 6144, 4]` と違う）だが、
形を決め打ちしていないのでそのまま動いた。

## 7. 制約

- **ONNX が公開されているのは 0.8B のみ**。OneJev には 4B / 9B / 27B もあるが、
  PyTorch と GGUF だけ
- 1 問 1 プレフィル。参照実装のような共通部分（状態・画像）の KV キャッシュ再利用は
  未実装。結果は同じだが、質問が多いと遅い
- 静止画のみ。動画（`<video:N>`）は未対応
- fp32 は未検証。q4f16 は劣化が大きい
- 画像は URL 取得に対応しない（パスまたは data URI）
- `tokenizer.py` / `chat.py` はこのチェックポイント専用。想定外の入力は例外で弾く

## 8. 再現手順

```powershell
uv venv .venv-min --python 3.12
uv pip install --python .venv-min/Scripts/python.exe --no-deps onnxruntime numpy pillow

python export\prepare_model.py            # 公開 ONNX（fp16）、HTTPS のみ
python demo_inference_text.py
python demo_inference_image.py
python verify\run_all.py

# ゴールデンを作り直す場合（torch と参照実装が要る）
python export\prepare_model.py --checkpoint
git clone https://github.com/OmniJev/OneJev ref/OneJev
uv pip install --no-deps -e ref/OneJev    # + uv sync --group export
python export\make_golden.py
```
