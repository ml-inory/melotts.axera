#!/usr/bin/env python3
import argparse
import json
import importlib.machinery
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


class BertHiddenWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-3]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def install_optional_dependency_stubs() -> None:
    if "torchaudio" not in sys.modules:
        module = types.ModuleType("torchaudio")
        module.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
        module.__version__ = "0.0"
        module.__path__ = []
        sys.modules["torchaudio"] = module

    if "torchvision" not in sys.modules:
        tv = types.ModuleType("torchvision")
        tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None, is_package=True)
        tv.__version__ = "0.0"
        tv.__path__ = []
        sys.modules["torchvision"] = tv
    if "torchvision.io" not in sys.modules:
        tv_io = types.ModuleType("torchvision.io")
        tv_io.__spec__ = importlib.machinery.ModuleSpec("torchvision.io", loader=None)

        class ImageReadMode:
            RGB = "RGB"

        def decode_image(*args, **kwargs):
            raise RuntimeError("torchvision is stubbed for text-only BERT export")

        tv_io.ImageReadMode = ImageReadMode
        tv_io.decode_image = decode_image
        sys.modules["torchvision.io"] = tv_io
    if "torchvision.transforms" not in sys.modules:
        tv_transforms = types.ModuleType("torchvision.transforms")
        tv_transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)

        class InterpolationMode:
            NEAREST = 0
            NEAREST_EXACT = 0
            BILINEAR = 2
            BICUBIC = 3
            BOX = 4
            HAMMING = 5
            LANCZOS = 1

        tv_transforms.InterpolationMode = InterpolationMode
        tv_transforms.__path__ = []
        sys.modules["torchvision.transforms"] = tv_transforms
    if "torchvision.transforms.functional" not in sys.modules:
        tv_functional = types.ModuleType("torchvision.transforms.functional")
        tv_functional.__spec__ = importlib.machinery.ModuleSpec(
            "torchvision.transforms.functional", loader=None
        )

        def pil_to_tensor(*args, **kwargs):
            raise RuntimeError("torchvision is stubbed for text-only BERT export")

        tv_functional.pil_to_tensor = pil_to_tensor
        sys.modules["torchvision.transforms.functional"] = tv_functional


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(af.size, bf.size)
    af = af[:n]
    bf = bf[:n]
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return float(np.dot(af, bf) / denom) if denom else 0.0


def metrics(a: np.ndarray, b: np.ndarray):
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(af.size, bf.size)
    af = af[:n]
    bf = bf[:n]
    diff = af - bf
    return {
        "shape_a": list(np.asarray(a).shape),
        "shape_b": list(np.asarray(b).shape),
        "compared_elements": int(n),
        "cosine": cosine(af, bf),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mse": float(np.mean(diff * diff)),
    }


def load_text(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            return line
    raise RuntimeError(f"empty text file: {path}")


def normalize_text(origin_dir: Path, text: str) -> str:
    sys.path.insert(0, str(origin_dir))
    sys.path.insert(0, str(origin_dir / "melo"))
    try:
        from melo.text import chinese

        return chinese.text_normalize(text)
    except ModuleNotFoundError:
        # The upstream test sentence has no digits, so this fallback is sufficient
        # for BERT export validation when optional MeloTTS text deps are absent.
        rep_map = {
            "：": ",",
            "；": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": ".",
            "、": ",",
        }
        for src, dst in rep_map.items():
            text = text.replace(src, dst)
        return text


def to_numpy_inputs(batch):
    result = {}
    for name in ["input_ids", "attention_mask", "token_type_ids"]:
        value = batch.get(name)
        if value is None:
            value = torch.zeros_like(batch["input_ids"])
        result[name] = value.cpu().numpy().astype(np.int64)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-dir", required=True, type=Path)
    parser.add_argument("--text-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--hf-home", type=Path, default=None)
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(args.hf_home / "hub")

    seed_everything(1234)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = args.output_dir / "calib_data"
    calib_dir.mkdir(parents=True, exist_ok=True)

    install_optional_dependency_stubs()
    from transformers import AutoTokenizer, BertModel

    raw_text = load_text(args.text_file)
    norm_text = normalize_text(args.origin_dir, raw_text)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    hf_model = BertModel.from_pretrained(args.model_id)
    hf_model.eval()
    wrapper = BertHiddenWrapper(hf_model).eval()

    batch = tokenizer(
        norm_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_len,
    )
    if "token_type_ids" not in batch:
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])

    with torch.no_grad():
        start = time.perf_counter()
        torch_output = wrapper(
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
        ).cpu().numpy().astype(np.float32)
        torch_ms = (time.perf_counter() - start) * 1000.0

    onnx_path = args.output_dir / "bert-hidden-zh.onnx"
    torch.onnx.export(
        wrapper,
        (
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
        ),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["hidden_states"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    np_inputs = to_numpy_inputs(batch)
    start = time.perf_counter()
    ort_output = sess.run(["hidden_states"], np_inputs)[0].astype(np.float32)
    ort_ms = (time.perf_counter() - start) * 1000.0

    for name, value in np_inputs.items():
        np.save(calib_dir / f"{name}.npy", value)
        value.tofile(calib_dir / f"{name}.bin")
    np.save(args.output_dir / "torch_hidden_states.npy", torch_output)
    np.save(args.output_dir / "onnx_hidden_states.npy", ort_output)

    meta = {
        "model_name": "melotts-zh-bert-hidden",
        "framework": "transformers",
        "source_model": args.model_id,
        "language": "ZH",
        "target_hardware": "AX650",
        "max_token_length": args.max_len,
        "opset": args.opset,
        "text": raw_text,
        "norm_text": norm_text,
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, args.max_len],
                "dtype": "int64",
                "layout": "BT",
                "preprocess": "AutoTokenizer padding=max_length truncation=True max_length=256",
            },
            {
                "name": "attention_mask",
                "shape": [1, args.max_len],
                "dtype": "int64",
                "layout": "BT",
                "preprocess": "AutoTokenizer attention mask",
            },
            {
                "name": "token_type_ids",
                "shape": [1, args.max_len],
                "dtype": "int64",
                "layout": "BT",
                "preprocess": "zeros when tokenizer does not provide segment ids",
            },
        ],
        "outputs": [
            {
                "name": "hidden_states",
                "shape": [1, args.max_len, 1024],
                "dtype": "float32",
                "semantic": "BERT hidden_states[-3], token level; CPU repeats token vectors by word2ph to produce MeloTTS phone-level BERT",
            }
        ],
        "tokenizer_path": args.model_id,
        "metrics": {
            "onnx_vs_torch": metrics(ort_output, torch_output),
            "torch_latency_ms": torch_ms,
            "onnxruntime_latency_ms": ort_ms,
        },
    }
    (args.output_dir / "bert_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# MeloTTS ZH BERT Export Report",
        "",
        f"- source model: `{args.model_id}`",
        f"- max token length: `{args.max_len}`",
        f"- ONNX: `{onnx_path}`",
        f"- text: `{raw_text}`",
        f"- normalized text: `{norm_text}`",
        "",
        "## Metrics",
        "",
        "| Comparison | Cosine | MAE | Max Abs Diff |",
        "|---|---:|---:|---:|",
        f"| ONNX vs Torch hidden_states[-3] | {meta['metrics']['onnx_vs_torch']['cosine']:.10f} | {meta['metrics']['onnx_vs_torch']['mae']:.10f} | {meta['metrics']['onnx_vs_torch']['max_abs_diff']:.10f} |",
        "",
        "The exported graph returns token-level `hidden_states[-3]`. MeloTTS phone-level BERT remains a CPU postprocess step that repeats token vectors according to `word2ph`.",
    ]
    (args.output_dir / "bert_export_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
