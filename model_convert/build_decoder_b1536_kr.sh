#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

pulsar2 build \
    --config config_decoder_b1536_kr_u16.json \
    --input decoder-b1536-kr.onnx \
    --output_dir decoder-b1536-kr \
    --output_name decoder-b1536-kr.axmodel \
    --target_hardware AX620E \
    --npu_mode NPU1
