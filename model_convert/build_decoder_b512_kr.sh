#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

pulsar2 build \
    --config config_decoder_b512_kr_u16.json \
    --input decoder-b512-kr.onnx \
    --output_dir decoder-b512-kr \
    --output_name decoder-b512-kr.axmodel \
    --target_hardware AX620E \
    --npu_mode NPU1
