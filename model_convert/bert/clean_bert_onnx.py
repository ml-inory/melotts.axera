#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def metrics(a: np.ndarray, b: np.ndarray):
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(af.size, bf.size)
    af = af[:n]
    bf = bf[:n]
    diff = af - bf
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    return {
        "cosine": float(np.dot(af, bf) / denom) if denom else 0.0,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mse": float(np.mean(diff * diff)),
        "compared_elements": int(n),
    }


def replace_tensor_uses(model, old: str, new: str) -> None:
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == old:
                node.input[idx] = new
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calib-dir", required=True, type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.input), load_external_data=True)
    producers = {out: node for node in model.graph.node for out in node.output}
    remove = set()
    replaced = []
    for node in list(model.graph.node):
        if node.op_type != "Where":
            continue
        cond, true_value, false_value = node.input
        cond_node = producers.get(cond)
        if not cond_node or cond_node.op_type != "IsNaN":
            continue
        if cond_node.input[0] != false_value:
            continue
        replace_tensor_uses(model, node.output[0], false_value)
        remove.add(node.name)
        remove.add(cond_node.name)
        replaced.append({"where": node.name, "isnan": cond_node.name, "tensor": false_value})

    kept = [node for node in model.graph.node if node.name not in remove]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.output.name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )

    inputs = {
        name: np.load(args.calib_dir / f"{name}.npy")
        for name in ["input_ids", "attention_mask", "token_type_ids"]
    }
    old_sess = ort.InferenceSession(str(args.input), providers=["CPUExecutionProvider"])
    new_sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    old_out = old_sess.run(["hidden_states"], inputs)[0]
    new_out = new_sess.run(["hidden_states"], inputs)[0]
    result = {
        "removed_patterns": len(replaced),
        "patterns": replaced,
        "clean_vs_original": metrics(new_out, old_out),
    }
    report_path = args.output.with_suffix(".clean_report.json")
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
