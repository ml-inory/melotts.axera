#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_2stage_npu.py  —  MeloTTS 二阶段多桶导出脚本（sherpa-onnx 接口兼容）

【输入接口设计说明】
  输入不兼容 sherpa-onnx 是之前的设计选择，而非切图结构的限制。
  旧脚本继承了 AXERA convert.py 的 int32[phone_len] 接口，但完全可以将
  encoder 包装为接受 sherpa 风格的 7 个输入（int64[1,L] 等），并在
  wrapper 内部完成 int64→int32 转换、language 张量构建、emb_g(sid) 计算。
  本脚本即实现这一兼容性。

【设计目标】
  1. Encoder 输入与 sherpa-onnx 完全一致（7 输入，int64/float32）
  2. Decoder 固定形状多桶，最大化 NPU 利用率（flow+HiFiGAN 合并在 NPU）
  3. 运行时只需：选桶 + 一次右侧 pad + 一次尾部 trim，无 overlap/pronoun_lens

【切分方案】
  CPU: encoder-sherpa-{lang}.onnx
    输入 (7, 与 sherpa-onnx offline-tts-vits-model.cc 构建的输入完全对应):
      x             int64[1, L]    phone IDs（batch=1）
      x_lengths     int64[1]       序列长度（接口保留，内部由 x 直接推断）
      tones         int64[1, L]    tone IDs（batch=1）
      sid           int64[1]       说话人 ID
      noise_scale   float32[1]
      length_scale  float32[1]     1.0/speed（与 sherpa 语义一致）
      noise_scale_w float32[1]
    输出:
      z_p           float32[1, 192, T]  潜在表示（T 动态）
      g             float32[1, 256, 1]  说话人嵌入（传给 decoder）
    内部处理（对 C++ 透明）:
      x[0]: int64→int32，去掉 batch dim
      tones[0]: int64→int32，去掉 batch dim
      language: 根据语言固定 lang_id 常量构建
      g: emb_g(sid) 在图内部计算，无需外部 g.bin
      sdp_ratio: 固定 0.0（与 sherpa 行为一致）

  NPU: decoder-b{N}-{lang}.onnx  (多个固定桶)
    输入:  z_p float32[1,192,N], g float32[1,256,1]
    输出:  audio float32[1,1,N*upsample_factor]

【运行时 5 步逻辑（C++）】
  1. z_p, g = encoder.Run(x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w)
  2. z_len  = z_p.shape[2]
  3. B      = min(b for b in BUCKETS if b >= z_len)     ← 选最小可容纳桶
  4. z_p_B  = zero_pad_right(z_p, B)                    ← 右侧 pad
  5. audio  = decoder_bB.Run(z_p_B, g)[:, :, :z_len * UPSAMPLE]  ← trim

【与 sherpa C++ 前后处理的对齐】
  - 复用 MeloTtsLexicon::ConvertTextToTokenIds() → tokens/tones int64[1,L]
  - 复用 OfflineTtsImpl::AddBlank() 插入 blank token
  - speed → length_scale = 1.0/speed 保持 sherpa 语义
  - 后处理：读取音频张量并转 float vector，与 sherpa Process() 完全一致
  - C++ 改动最小：只替换 sess_->Run() 部分，增加 encoder+decoder 两个会话

【使用方法】
  conda activate cosyvoice
  cd /home/m5stack/Workspace/AXERA/melotts.axera/model_convert
  python /path/to/export_2stage_npu.py -l ZH --buckets 256,512,1024,1536 --out-dir ./output_2stage
"""

import argparse
import json
import os
import sys
from typing import List

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxsim
except Exception:
    print("Please install onnx and onnxsim first: pip install onnx onnxsim")
    raise

# 这里假定脚本放在 AXERA/melotts.axera/model_convert 下运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from melotts.download_utils import load_or_download_config, load_or_download_model
from melotts.tts import TTS

# ─── 语言 ID 映射 ─────────────────────────────────────────────────────────────
# ZH=3 由 AXERA melotts.cpp 中 `langids.assign(phones.size(), 3)` 确认
# 其他语言 ID 需根据实际训练配置验证，可通过 --lang-id 参数覆盖
LANG_ID_MAP = {
    "ZH": 3,
    "EN": 0,
    "JP": 1,
    "KR": 2,
    "FR": 4,
    "ES": 5,
}

SAMPLE_TEXTS = {
    "ZH": "爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。",
    "EN": "Did you ever hear a folk tale about a giant turtle?",
    "JP": "海の向こうには何があるの？",
    "KR": "한국 음식을 먹어보고 싶어요. 불고기랑 김치찌개가 제가 좋아하는 음식이에요.",
    "FR": "Les cafés animés résonnent de conversations passionnées.",
    "ES": "El susurro suave del viento atraviesa los campos de lavanda.",
}

# tone_start: 各语言声调 ID 的起始偏移（与 language_tone_start_map 对应）
# ZH=0 已知；其他语言需根据实际训练配置确认，可通过 melo.text.language_tone_start_map 查询
TONE_START_MAP = {
    "ZH": 0,
    "EN": 6,
    "JP": 0,
    "KR": 0,
    "FR": 0,
    "ES": 0,
}

LANGUAGE_DISPLAY_NAMES = {
    "ZH": "Chinese + English",
    "EN": "English",
    "JP": "Japanese",
    "KR": "Korean",
    "FR": "French",
    "ES": "Spanish",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

class SherpaCompatEncoder(nn.Module):
    """
    Encoder wrapper：输入接口与 sherpa-onnx MeloTTS model.onnx 完全一致（7 输入）。

    【兼容性实现说明】
    sherpa-onnx C++ 在 offline-tts-vits-model.cc Run() 中构建 7 个输入张量：
      x int64[1,L], x_lengths int64[1], tones int64[1,L], sid int64[1],
      noise_scale float32[1], length_scale float32[1], noise_scale_w float32[1]
    本 wrapper 接受完全相同的 7 个输入，在 ONNX 计算图内部完成：
      - x[0]:     int64[L] → int32[L]  （strip batch dim + cast）
      - tones[0]: int64[L] → int32[L]  （strip batch dim + cast）
      - language: int32[L] 由 lang_id 常量构建（per-language 固定值）
      - g:        emb_g(sid) 在图内部计算（无需外部 g.bin 文件）
      - sdp_ratio: 固定 0.0 buffer（与 sherpa 行为一致）
    对 C++ 推理层完全透明，无需修改任何前处理代码。

    Inputs (7, 与 sherpa-onnx 一致):
      x:             int64[1, L]   phone IDs
      x_lengths:     int64[1]      保留以匹配接口，内部不使用
      tones:         int64[1, L]   tone IDs
      sid:           int64[1]      speaker ID
      noise_scale:   float32[1]
      length_scale:  float32[1]    = 1.0 / speed
      noise_scale_w: float32[1]

    Outputs:
      z_p:  float32[1, 192, T]    潜在表示（T 动态）
      g:    float32[1, 256, 1]    说话人嵌入（直接传给 decoder，无需 g.bin）
    """

    def __init__(self, model, lang_id: int):
        super().__init__()
        self.model = model
        self.lang_id = lang_id
        # sdp_ratio 固定 0.0，用 register_buffer 确保 ONNX constant folding 正确
        self.register_buffer("_sdp_ratio", torch.zeros(1, dtype=torch.float32))

    def forward(self, x, x_lengths, tones, sid,
                noise_scale, length_scale, noise_scale_w):
        # ── 1. 类型转换 + 去掉 batch dim ─────────────────────────────────────
        phone = x[0].to(torch.int32)       # int64[1,L] → int32[L]
        tone  = tones[0].to(torch.int32)   # int64[1,L] → int32[L]
        # language: 与原始 ModelWrapper 一致——奇数位（实际音素位）赋 lang_id，
        # 偶数位（blank token）保持 0。对应原始: lang_id[:, 1::2] = self.lang_id
        language = torch.zeros_like(phone)
        language[1::2] = self.lang_id

        # ── 2. 说话人嵌入（在图内部，不需要外部 g.bin）───────────────────────
        # sid: int64[1] → emb_g → float32[1,256] → float32[1,256,1]
        g = self.model.emb_g(sid).unsqueeze(-1)

        # ── 3. enc_forward（忽略 pronoun_lens/audio_len）─────────────────────
        z_p, _, _ = self.model.enc_forward(
            phone, tone, language, g,
            noise_scale=noise_scale[0],
            noise_scale_w=noise_scale_w[0],
            length_scale=length_scale[0],
            sdp_ratio=self._sdp_ratio[0],
        )
        return z_p, g


class BucketedDecoder(nn.Module):
    """
    合并 flow + HiFiGAN Generator 的固定形状 decoder，适合 NPU 部署。

    Inputs:
      z_p:   float32[1, 192, B]   固定桶长度 B（不设 dynamic_axes → NPU 可编译）
      g:     float32[1, 256, 1]   说话人嵌入（来自 encoder 输出）

    Output:
      audio: float32[1, 1, B * upsample_factor]

    内部等价于：
      y_mask = ones(1, B)
      z = flow(z_p, y_mask, g, reverse=True)
      audio = dec(z * y_mask, g=g)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z_p, g):
        return self.model.flow_dec_forward(z_p, g)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def simplify_onnx(path: str) -> None:
    """ONNX 图优化并原地覆盖保存。"""
    model = onnx.load(path)
    model_sim, ok = onnxsim.simplify(model)
    if not ok:
        print(f"  Warning: onnxsim check failed for {os.path.basename(path)}, saving unsimplified")
        model_sim = model
    onnx.save(model_sim, path)


def add_meta_data(filename: str, meta_data: dict) -> None:
    """向 ONNX 模型写入 metadata（原地覆盖，与原始 export-onnx.py 实现一致）。"""
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)


def get_upsample_factor(model) -> int:
    """从 HiFiGAN Generator 的上采样卷积层中读取总上采样倍率。"""
    factor = 1
    for up in model.dec.ups:
        factor *= int(up.stride[0])
    return factor


def parse_buckets(s: str) -> List[int]:
    vals = sorted({int(v.strip()) for v in s.split(",") if v.strip()})
    if any(v <= 0 for v in vals):
        raise ValueError(f"All bucket lengths must be positive: {vals}")
    return vals


# ═══════════════════════════════════════════════════════════════════════════════
# Export functions
# ═══════════════════════════════════════════════════════════════════════════════

def export_encoder(tts: TTS, out_path: str, lang_id: int, lang: str,
                   speaker_id: int, L_dummy: int = 160,
                   skip_simplify: bool = False) -> None:
    """
    导出 sherpa-onnx 接口兼容的 Encoder ONNX。
    动态轴: x/tones dim=1 (L), z_p dim=2 (T)
    """
    wrapper = SherpaCompatEncoder(tts.model, lang_id=lang_id).eval()

    # dummy 输入与 sherpa C++ 构建的 7 个输入张量完全对应
    x             = torch.zeros(1, L_dummy, dtype=torch.int64)
    x_lengths     = torch.tensor([L_dummy], dtype=torch.int64)
    tones_t       = torch.zeros(1, L_dummy, dtype=torch.int64)
    sid           = torch.tensor([0], dtype=torch.int64)
    noise_scale   = torch.tensor([0.667], dtype=torch.float32)
    length_scale  = torch.tensor([1.0],   dtype=torch.float32)
    noise_scale_w = torch.tensor([0.8],   dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, x_lengths, tones_t, sid, noise_scale, length_scale, noise_scale_w),
            out_path,
            input_names=["x", "x_lengths", "tones", "sid",
                         "noise_scale", "length_scale", "noise_scale_w"],
            output_names=["z_p", "g"],
            dynamic_axes={
                "x":     {1: "L"},
                "tones": {1: "L"},
                "z_p":   {2: "T"},
            },
            opset_version=16,
            do_constant_folding=True,
            export_params=True,
        )

    if not skip_simplify:
        simplify_onnx(out_path)

    # Metadata（与原始 export-onnx.py add_meta_data 字段保持一致）
    add_meta_data(out_path, {
        "model_type": "melo-vits",
        "comment": "melo-axera-2stage-encoder",
        "version": 2,
        "language": LANGUAGE_DISPLAY_NAMES.get(lang, lang),
        "add_blank": int(tts.hps.data.add_blank),
        "n_speakers": len(tts.hps.data.spk2id),
        "jieba": 1 if lang == "ZH" else 0,
        "sample_rate": tts.hps.data.sampling_rate,
        "bert_dim": 1024,
        "ja_bert_dim": 768,
        "speaker_id": speaker_id,
        "lang_id": lang_id,
        "tone_start": TONE_START_MAP.get(lang, 0),
        "url": "https://github.com/myshell-ai/MeloTTS",
        "license": "MIT license",
        "description": "MeloTTS 2-stage encoder for AXERA NPU (sherpa-onnx compatible interface)",
    })

    mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  OK  {os.path.basename(out_path)} ({mb:.1f} MB)")
    print(f"      Input:  x int64[1,L], x_lengths int64[1], tones int64[1,L],")
    print(f"              sid int64[1], noise_scale/length_scale/noise_scale_w float32[1]")
    print(f"      Output: z_p float32[1,192,T], g float32[1,256,1]")


def export_decoder_bucket(tts: TTS, out_path: str, bucket_len: int,
                          upsample: int, lang: str,
                          skip_simplify: bool = False) -> None:
    """
    导出固定形状 Decoder ONNX（flow + HiFiGAN 合并），一桶一文件。
    不设 dynamic_axes → 完全固定形状 → NPU 可直接编译为 axmodel。
    """
    wrapper = BucketedDecoder(tts.model).eval()
    z_p = torch.rand(1, 192, bucket_len, dtype=torch.float32)
    g   = torch.rand(1, 256, 1,          dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (z_p, g),
            out_path,
            input_names=["z_p", "g"],
            output_names=["audio"],
            opset_version=16,
            do_constant_folding=True,
            export_params=True,
            # 不设 dynamic_axes：固定形状，NPU 友好
        )

    if not skip_simplify:
        simplify_onnx(out_path)

    add_meta_data(out_path, {
        "model_type": "melo-vits-decoder-bucket",
        "language": LANGUAGE_DISPLAY_NAMES.get(lang, lang),
        "bucket_len": bucket_len,
        "upsample_factor": upsample,
        "sample_rate": tts.hps.data.sampling_rate,
        "url": "https://github.com/myshell-ai/MeloTTS",
        "license": "MIT license",
        "description": f"MeloTTS 2-stage flow+generator decoder bucket B={bucket_len} for AXERA NPU",
    })

    mb = os.path.getsize(out_path) / 1024 / 1024
    audio_len = bucket_len * upsample
    print(f"  OK  {os.path.basename(out_path)} ({mb:.1f} MB)")
    print(f"      Input:  z_p float32[1,192,{bucket_len}], g float32[1,256,1]")
    print(f"      Output: audio float32[1,1,{audio_len}]")


def generate_calibration_data(tts: TTS, speaker_id: int, lang: str,
                               out_dir: str, buckets: List[int],
                               lang_id: int) -> None:
    """
    生成 decoder 量化校准数据（z_p 来自 encoder 真实输出，非随机数）。
    """
    import tarfile
    from melotts.tts import get_text_for_tts_infer

    text = SAMPLE_TEXTS[lang]
    print(f"  Calibration text: {text}")

    model = tts.model
    _, _, phones, tones_t, lang_ids = get_text_for_tts_infer(
        text, tts.language, tts.hps, "cpu", tts.symbol_to_id
    )

    with torch.no_grad():
        # 使用与 SherpaCompatEncoder 相同的路径计算 g
        g = model.emb_g(torch.LongTensor([speaker_id])).unsqueeze(-1)
        z_p, _, _ = model.enc_forward(
            phones.int(), tones_t.int(), lang_ids.int(), g,
            noise_scale=0.667, noise_scale_w=0.8,
            length_scale=1.0, sdp_ratio=0.0,
        )

        for b in buckets:
            zp_dir = os.path.join(out_dir, f"calib_b{b}_zp")
            g_dir  = os.path.join(out_dir, f"calib_b{b}_g")
            os.makedirs(zp_dir, exist_ok=True)
            os.makedirs(g_dir,  exist_ok=True)

            T, n = z_p.shape[2], 0
            for start in range(0, T, b):
                chunk = z_p[:, :, start:start + b]
                if chunk.shape[2] < b:
                    chunk = torch.nn.functional.pad(chunk, (0, b - chunk.shape[2]))
                np.save(os.path.join(zp_dir, f"{n:04d}.npy"), chunk.numpy())
                np.save(os.path.join(g_dir,  f"{n:04d}.npy"), g.numpy())
                n += 1

            for d, name in [(zp_dir, "z_p"), (g_dir, "g")]:
                tar = d + ".tar.gz"
                with tarfile.open(tar, "w:gz") as tf:
                    for fn in sorted(os.listdir(d)):
                        if fn.endswith(".npy"):
                            tf.add(os.path.join(d, fn), arcname=fn)
                print(f"  Packed {n} samples → {tar}")


def write_decoder_quant_config(out_path: str, zp_tar: str, g_tar: str,
                               npu_mode: str = "NPU1",
                               data_type: str = "U16") -> None:
    """
    生成与 config_decoder_u16.json 同结构的量化配置。
    """
    cfg = {
        "model_type": "ONNX",
        "npu_mode": npu_mode,
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "z_p",
                    "calibration_dataset": zp_tar,
                    "calibration_size": -1,
                    "calibration_format": "Numpy",
                },
                {
                    "tensor_name": "g",
                    "calibration_dataset": g_tar,
                    "calibration_size": -1,
                    "calibration_format": "Numpy",
                },
            ],
            "layer_configs": [
                {
                    "start_tensor_names": ["DEFAULT"],
                    "end_tensor_names": ["DEFAULT"],
                    "data_type": data_type,
                }
            ],
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
        },
        "input_processors": [
            {
                "tensor_name": "z_p",
                "src_dtype": "FP32",
            },
            {
                "tensor_name": "g",
                "src_dtype": "FP32",
            },
        ],
        "compiler": {
            "check": 2,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def write_bucket_quant_configs(out_dir: str, lang: str, buckets: List[int],
                               npu_mode: str = "NPU1",
                               data_type: str = "U16") -> None:
    """
    为每个 decoder-b{N}-{lang}.onnx 生成对应量化配置。
    """
    low_lang = lang.lower()
    index = {
        "language": lang,
        "npu_mode": npu_mode,
        "data_type": data_type,
        "configs": [],
    }

    for b in buckets:
        cfg_name = f"config_decoder_b{b}_{low_lang}_{data_type.lower()}.json"
        cfg_path = os.path.join(out_dir, cfg_name)
        zp_tar = os.path.join(out_dir, f"calib_b{b}_zp.tar.gz")
        g_tar = os.path.join(out_dir, f"calib_b{b}_g.tar.gz")

        if not os.path.exists(zp_tar) or not os.path.exists(g_tar):
            raise FileNotFoundError(
                f"Missing calibration tar for bucket {b}: {zp_tar} or {g_tar}. "
                "Run with --calib first (or make sure tar files already exist)."
            )

        write_decoder_quant_config(
            out_path=cfg_path,
            zp_tar=zp_tar,
            g_tar=g_tar,
            npu_mode=npu_mode,
            data_type=data_type,
        )
        index["configs"].append({
            "bucket": b,
            "onnx": f"decoder-b{b}-{low_lang}.onnx",
            "config": cfg_name,
            "zp_tar": os.path.basename(zp_tar),
            "g_tar": os.path.basename(g_tar),
        })
        print(f"  OK  {cfg_name}")

    index_path = os.path.join(out_dir, "quant_config_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"  OK  {os.path.basename(index_path)}")


def write_runtime_spec(out_path: str, lang: str, lang_id: int,
                        buckets: List[int], upsample: int) -> None:
    """写 C++ 加载用的 JSON 运行时规格。"""
    spec = {
        "language": lang,
        "lang_id": lang_id,
        "upsample_factor": upsample,
        "buckets": buckets,
        "encoder": f"encoder-sherpa-{lang.lower()}.onnx",
        "decoder_pattern": f"decoder-b{{N}}-{lang.lower()}.onnx",
        "runtime_steps": [
            "z_p, g = encoder.Run(x, x_lengths, tones, sid, noise_scale, length_scale, noise_scale_w)",
            "z_len = z_p.shape[2]",
            "B = min(b for b in buckets if b >= z_len)",
            "z_p_padded = zero_pad_right(z_p, B)  // right-pad on dim 2",
            f"audio = decoder_bB.Run(z_p_padded, g)[:, :, :z_len * {upsample}]  // trim",
        ],
        "note": "No overlap slicing. No pronoun_lens/audio_len dependency. Single pad + single trim.",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print(f"  OK  {os.path.basename(out_path)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="MeloTTS 2-stage bucketed export (sherpa-onnx compatible input interface)"
    )
    p.add_argument("-l", "--language", default="ZH",
                   choices=["ZH", "EN", "JP", "KR", "FR", "ES"])
    p.add_argument("--config",   default="config.json")
    p.add_argument("--ckpt",     default="checkpoint.pth")
    p.add_argument("--out-dir",  default="./output_2stage_bucket")
    p.add_argument("--buckets",  default="256,512,1024,1536",
                   help="z_p dim=2 桶长度，逗号分隔 (e.g. 256,512,1024,1536)")
    p.add_argument("--lang-id",  type=int, default=None,
                   help="per-token 语言 ID（默认从 LANG_ID_MAP 读取，ZH=3）")
    p.add_argument("--calib",    action="store_true",
                   help="为每个 decoder 桶生成 NPU 量化校准数据")
    p.add_argument("--quant-config", action="store_true",
                   help="基于校准数据为每个 decoder 桶自动生成 pulsar2 量化配置")
    p.add_argument("--quant-data-type", default="U16", choices=["U8", "U16"],
                   help="量化层默认数据类型（写入 layer_configs）")
    p.add_argument("--quant-npu-mode", default="NPU1", choices=["NPU1", "NPU2", "NPU3"],
                   help="写入量化配置的 npu_mode")
    p.add_argument("--skip-simplify", action="store_true",
                   help="跳过 onnxsim（更快但 ONNX 更大）")
    return p.parse_args()


def main():
    args    = get_args()
    lang    = args.language
    buckets = parse_buckets(args.buckets)
    lang_id = args.lang_id if args.lang_id is not None else LANG_ID_MAP.get(lang, 0)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nMeloTTS 2-stage bucketed export (sherpa-onnx interface)")
    print(f"  Language : {lang}  (lang_id={lang_id})")
    print(f"  Buckets  : {buckets}")
    print(f"  Out dir  : {args.out_dir}\n")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.config):
        print("Downloading config...")
        load_or_download_config(locale=lang)
    if not os.path.exists(args.ckpt):
        print("Downloading checkpoint...")
        load_or_download_model(locale=lang, device="cpu")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    spk_key    = "EN-US" if lang == "EN" else lang
    speaker_id = cfg["data"]["spk2id"][spk_key]
    print(f"  Speaker ID : {speaker_id}")

    tts = TTS(
        language=lang,
        dec_len=max(buckets),
        config_path=args.config,
        ckpt_path=args.ckpt,
        device="cpu",
    )
    upsample = get_upsample_factor(tts.model)
    low_lang = lang.lower()
    print(f"  Upsample   : {upsample}x  (decoder output = bucket × {upsample})\n")

    # ── 1. Encoder（sherpa-onnx 接口）────────────────────────────────────────
    enc_path = os.path.join(args.out_dir, f"encoder-sherpa-{low_lang}.onnx")
    print("[1/3] Encoder (sherpa-onnx compatible interface)")
    export_encoder(tts, enc_path, lang_id=lang_id, lang=lang,
                   speaker_id=speaker_id, skip_simplify=args.skip_simplify)

    # ── 2. Decoder 多桶（固定形状，NPU 友好）──────────────────────────────────
    print(f"\n[2/3] Decoder buckets ({len(buckets)} buckets, fixed shape)")
    for b in buckets:
        dec_path = os.path.join(args.out_dir, f"decoder-b{b}-{low_lang}.onnx")
        export_decoder_bucket(tts, dec_path, bucket_len=b, upsample=upsample,
                              lang=lang, skip_simplify=args.skip_simplify)

    # ── 3. 运行时规格 JSON ────────────────────────────────────────────────────
    print("\n[3/3] Runtime spec")
    spec_path = os.path.join(args.out_dir, "runtime_spec.json")
    write_runtime_spec(spec_path, lang, lang_id, buckets, upsample)

    # ── 可选：校准数据 ────────────────────────────────────────────────────────
    if args.calib:
        print("\n[+] Generating decoder calibration data...")
        generate_calibration_data(tts, speaker_id, lang, args.out_dir,
                                  buckets, lang_id)

    if args.quant_config:
        print("\n[+] Writing per-bucket quant configs...")
        write_bucket_quant_configs(
            out_dir=args.out_dir,
            lang=lang,
            buckets=buckets,
            npu_mode=args.quant_npu_mode,
            data_type=args.quant_data_type,
        )

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("Export complete")
    files = ([f"encoder-sherpa-{low_lang}.onnx"] +
             [f"decoder-b{b}-{low_lang}.onnx" for b in buckets] +
             ["runtime_spec.json"])
    for fn in files:
        fp = os.path.join(args.out_dir, fn)
        if os.path.exists(fp):
            mb = os.path.getsize(fp) / 1024 / 1024
            print(f"  {fn:<42}  {mb:>7.1f} MB")

    print(f"\nRuntime 5-step flow:")
    print(f"  z_p, g = encoder.Run(x, x_lengths, tones, sid, ...);")
    print(f"  z_len  = z_p.shape[2];")
    print(f"  B      = select_bucket(z_len, {buckets});")
    print(f"  audio  = decoder_bB.Run(pad_right(z_p, B), g);")
    print(f"  audio  = audio[:, :, :z_len * {upsample}];   // trim")
    print(f"  // No overlap. No pronoun_lens. No g.bin.")
    if args.quant_config:
        print("\nQuant config output:")
        print("  - config_decoder_b{N}_{lang}_u16.json (or u8)")
        print("  - quant_config_index.json")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()