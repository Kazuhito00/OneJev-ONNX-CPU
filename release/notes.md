ONNX files for [OneJev-ONNX-CPU](https://github.com/Kazuhito00/OneJev-ONNX-CPU): OneJev-0.8B and OneJev-4B (`OmniJev/OneJev-0.8B`, `OmniJev/OneJev-4B`, Apache-2.0) on onnxruntime + numpy.

Fetch them with `uv run download_model.py` (both sizes; `--model-size 0.8b` or `4b` for one of them). Every file is checked against the SHA-256 in `manifest-0.8b.json` / `manifest-4b.json`. Do not download the assets by hand: the asset names are the file paths with `/` replaced by `__` (for example `onnx__onnx__decoder_model_merged_fp16.onnx_data` is `models/onnx/onnx/decoder_model_merged_fp16.onnx_data`), and the 4B decoder weights come in 5 shards.

These are converted files (see "About the converted files" in the README):
- 0.8B (`onnx__*`): the public `onnx-community/OneJev-0.8B-ONNX` (Apache-2.0), unchanged. fp16 and q4f16.
- 4B (`onnx-4b__*`): the OneJev-4B weights swapped into the graphs of the public `onnx-community/Qwen3.5-4B-ONNX` by `export/build_onnx_4b.py` (fp16). That repository declares no licence; the base model Qwen/Qwen3.5-4B is Apache-2.0.
- Decisions agree with the PyTorch fp32 reference: 17/17 for both sizes (max logit difference 0.010 for 0.8B fp16, 0.006 for 4B).

Licences: `*__LICENSE-ONEJEV` (OmniJev/OneJev, Apache-2.0) and `*__LICENSE-QWEN` (Qwen3.5, Apache-2.0), unchanged. The training data of OneJev is licensed per source and includes research-only / non-commercial sources (see the README).

| Asset | Content |
|---|---|
| `onnx__onnx__{decoder_model_merged,embed_tokens,vision_encoder}_{fp16,q4f16}.onnx` and `.onnx_data` | 0.8B graphs and weights (fp16 about 2.1 GB, q4f16 about 0.65 GB) |
| `onnx-4b__onnx__decoder_model_merged_fp16.onnx`, `.onnx_data`, `.onnx_data_1` .. `_4` | 4B decoder (about 8 GB, 5 shards, each under 2 GiB) |
| `onnx-4b__onnx__{embed_tokens,vision_encoder}_fp16.onnx` and `.onnx_data` | 4B embedding (1.27 GB) and vision encoder (0.67 GB) |
| `onnx__tokenizer.json`, `onnx__config.json`, ... and the `onnx-4b__` equivalents | tokenizer and settings |
| `manifest-0.8b.json`, `manifest-4b.json`, `SHA256SUMS-0.8b`, `SHA256SUMS-4b` | file lists with sizes and SHA-256 |

Upload with: `gh release create v0.0.0 dist/release/* --title "v0.0.0" --notes-file release/notes.md` (after `uv run release/make_release.py`; no asset reaches 2 GiB, the largest is 2,095,788,032 bytes).
