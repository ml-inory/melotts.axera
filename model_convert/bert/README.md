# MeloTTS ZH BERT AXMODEL

This directory contains the reproducible conversion path for the ZH BERT hidden-state model used by the Python runtime.

The runtime uses BERT AXMODEL only. CPU BERT fallback is intentionally disabled: if an encoder ONNX exposes `bert` or `ja_bert`, pass `--bert ../models/bert-hidden-u16-zh.axmodel`.

## Export ONNX

```bash
cd model_convert/bert
python3 export_bert_onnx.py \
  --origin-dir ../../model_convert \
  --text-file ../test_text_zh.txt \
  --output-dir output \
  --max-len 256

python3 clean_bert_onnx.py \
  --input output/bert-hidden-zh.onnx \
  --output output/bert-hidden-zh-noand.onnx \
  --calib-dir output/calib_data
```

## Compile U16 AXMODEL

```bash
bash compile_bert_u16.sh output
```

The expected output is:

```text
output/bert_u16/bert-hidden-u16-zh.axmodel
```

Copy it to:

```text
models/bert-hidden-u16-zh.axmodel
```

Do not commit generated `.onnx`, `.onnx.data`, `.axmodel`, `.npy`, or `.tar.gz` artifacts.
