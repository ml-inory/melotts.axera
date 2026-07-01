#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/output}"
ONNX_PATH="${2:-${OUT_DIR}/bert-hidden-zh-noand.onnx}"
AXMODEL_DIR="${3:-${OUT_DIR}/bert_u16}"
WORK_DIR="${AXMODEL_DIR}/bert_work"
CALIB_SRC="${OUT_DIR}/calib_data"
CALIB_DST="${AXMODEL_DIR}/calibration_dataset"
CONFIG_PATH="${AXMODEL_DIR}/pulsar2_config_bert_u16.json"

mkdir -p "${AXMODEL_DIR}" "${CALIB_DST}"

for name in input_ids attention_mask token_type_ids; do
  rm -rf "${CALIB_DST}/${name}" "${CALIB_DST}/${name}.tar.gz"
  mkdir -p "${CALIB_DST}/${name}"
  cp "${CALIB_SRC}/${name}.npy" "${CALIB_DST}/${name}/0.npy"
  tar -C "${CALIB_DST}" -czf "${CALIB_DST}/${name}.tar.gz" "${name}"
done

cat > "${CONFIG_PATH}" <<JSON
{
  "input": "${ONNX_PATH}",
  "output_dir": "${AXMODEL_DIR}",
  "output_name": "bert-hidden-u16-zh.axmodel",
  "work_dir": "${WORK_DIR}",
  "model_type": "ONNX",
  "target_hardware": "AX650",
  "npu_mode": "NPU3",
  "input_shapes": "input_ids:1x256;attention_mask:1x256;token_type_ids:1x256",
  "quant": {
    "calibration_method": "MinMax",
    "precision_analysis": false,
    "precision_analysis_method": "EndToEnd",
    "layer_configs": [
      {
        "start_tensor_names": ["DEFAULT"],
        "end_tensor_names": ["DEFAULT"],
        "data_type": "U16"
      },
      {
        "op_type": "Softmax",
        "data_type": "U16"
      },
      {
        "op_type": "LayerNormalization",
        "data_type": "U16"
      },
      {
        "op_type": "Erf",
        "data_type": "U16"
      }
    ],
    "transformer_opt_level": 1,
    "input_configs": [
      {
        "tensor_name": "input_ids",
        "calibration_dataset": "${CALIB_DST}/input_ids.tar.gz",
        "calibration_size": -1,
        "calibration_format": "Numpy"
      },
      {
        "tensor_name": "attention_mask",
        "calibration_dataset": "${CALIB_DST}/attention_mask.tar.gz",
        "calibration_size": -1,
        "calibration_format": "Numpy"
      },
      {
        "tensor_name": "token_type_ids",
        "calibration_dataset": "${CALIB_DST}/token_type_ids.tar.gz",
        "calibration_size": -1,
        "calibration_format": "Numpy"
      }
    ]
  },
  "input_processors": [
    {
      "tensor_name": "input_ids",
      "src_dtype": "S64"
    },
    {
      "tensor_name": "attention_mask",
      "src_dtype": "S64"
    },
    {
      "tensor_name": "token_type_ids",
      "src_dtype": "S64"
    }
  ],
  "compiler": {
    "check": 0,
    "npu_perf": true
  }
}
JSON

pulsar2 build --config "${CONFIG_PATH}"
