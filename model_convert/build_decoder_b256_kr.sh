#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

pulsar2 build \
    --config config_decoder_b256_kr_u16.json \
    --input decoder-b256-kr.onnx \
    --output_dir decoder-b256-kr \
    --output_name decoder-b256-kr.axmodel \
    --target_hardware AX620E \
    --npu_mode NPU1
