import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import soundfile
import onnxruntime as ort
import axengine as axe
import argparse
import time
from split_utils import split_sentence
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from symbols import LANG_TO_SYMBOL_MAP
import re

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def get_text_for_tts_infer(text, language_str, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    phone = intersperse(phone, 0)
    tone = intersperse(tone, 0)
    language = intersperse(language, 0)

    phone = np.array(phone, dtype=np.int32)
    tone = np.array(tone, dtype=np.int32)
    language = np.array(language, dtype=np.int32)
    word2ph = np.array(word2ph, dtype=np.int32) * 2
    word2ph[0] += 1

    return phone, tone, language, norm_text, word2ph

def split_sentences_into_pieces(text, language, quiet=False):
    texts = split_sentence(text, language_str=language)
    if not quiet:
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
    return texts

def get_args():
    parser = argparse.ArgumentParser(
        prog="melotts",
        description="Run TTS on input sentence"
    )
    parser.add_argument("--sentence", "-s", type=str, required=False, default="爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用")
    parser.add_argument("--wav", "-w", type=str, required=False, default="output.wav")
    parser.add_argument("--encoder", "-e", type=str, required=False, default=None)
    parser.add_argument("--decoder", "-d", type=str, required=False, default=None)
    parser.add_argument("--dec_len", type=int, default=128)
    parser.add_argument("--sample_rate", "-sr", type=int, required=False, default=44100)
    parser.add_argument("--speed", type=float, required=False, default=0.8)
    parser.add_argument("--language", "-l", type=str, 
                        choices=["ZH", "ZH_MIX_EN", "JP", "EN", 'KR', "ES", "SP","FR"], required=False, default="ZH_MIX_EN")
    return parser.parse_args()


def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments


def merge_sub_audio(sub_audio_list, pad_size, audio_len):
    # Average pad part
    if pad_size > 0:
        for i in range(len(sub_audio_list) - 1):
            sub_audio_list[i][-pad_size:] += sub_audio_list[i+1][:pad_size]
            sub_audio_list[i][-pad_size:] /= 2
            if i > 0:
                sub_audio_list[i] = sub_audio_list[i][pad_size:]

    sub_audio = np.concatenate(sub_audio_list, axis=-1)
    return sub_audio[:audio_len]

# 计算每个词的发音时长
def calc_word2pronoun(word2ph, pronoun_lens):
    indice = [0]
    for ph in word2ph[:-1]:
        indice.append(indice[-1] + ph)
    word2pronoun = []
    for i, ph in zip(indice, word2ph):
        word2pronoun.append(np.sum(pronoun_lens[i : i + ph]))
    return word2pronoun

# 生成有overlap的slice，slice索引是对于zp的
def generate_slices(word2pronoun, dec_len):
    pn_start, pn_end = 0, 0
    zp_start, zp_end = 0, 0
    zp_len = 0
    pn_slices = []
    zp_slices = []
    while pn_end < len(word2pronoun):
        # 前一个slice长度大于2 且 加上现在这个字没有超过dec_len，则往前overlap两个字
        if pn_end - pn_start > 2 and np.sum(word2pronoun[pn_end - 2 : pn_end + 1]) <= dec_len:
            zp_len = np.sum(word2pronoun[pn_end - 2 : pn_end])
            zp_start = zp_end - zp_len
            pn_start = pn_end - 2
        else:
            zp_len = 0
            zp_start = zp_end
            pn_start = pn_end
            
        while pn_end < len(word2pronoun) and zp_len + word2pronoun[pn_end] <= dec_len:
            zp_len += word2pronoun[pn_end]
            pn_end += 1
        zp_end = zp_start + zp_len
        pn_slices.append(slice(pn_start, pn_end))
        zp_slices.append(slice(zp_start, zp_end))
    return pn_slices, zp_slices

def main():
    args = get_args()
    sentence = args.sentence
    sample_rate = args.sample_rate
    enc_model = args.encoder # default="../models/encoder.onnx"
    dec_model = args.decoder # default="../models/decoder.axmodel"
    language = args.language # default: ZH_MIX_EN
    dec_len = args.dec_len # default: 128

    if language == "ZH":
        language = "ZH_MIX_EN"

    if enc_model is None:
        if "ZH" in language:
            enc_model = "../models/encoder-zh.onnx"
        else:
            enc_model = f"../models/encoder-{language.lower()}.onnx"
        assert os.path.exists(enc_model), f"Encoder model ({enc_model}) not exist!"
    if dec_model is None:
        if "ZH" in language:
            dec_model = "../models/decoder-zh.axmodel"
        else:
            dec_model = f"../models/decoder-{language.lower()}.axmodel"
        assert os.path.exists(dec_model), f"Decoder model ({dec_model}) not exist!"

    print(f"sentence: {sentence}")
    print(f"sample_rate: {sample_rate}")
    print(f"encoder: {enc_model}")
    print(f"decoder: {dec_model}")
    print(f"language: {language}")

    _symbol_to_id = {s: i for i, s in enumerate(LANG_TO_SYMBOL_MAP[language])}

    # Split sentence
    start = time.time()
    sens = split_sentences_into_pieces(sentence, language, quiet=False)
    print(f"split_sentences_into_pieces take {1000 * (time.time() - start)}ms")

    # Load models
    start = time.time()
    sess_enc = ort.InferenceSession(enc_model, providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
    sess_dec = axe.InferenceSession(dec_model)
    print(f"load models take {1000 * (time.time() - start)}ms")

    # Load static input
    g = np.fromfile(f"../models/g-{language.lower()}.bin", dtype=np.float32).reshape(1, 256, 1)

    # Final wav
    audio_list = []

    # Iterate over splitted sentences
    for n, se in enumerate(sens):
        if language in ['EN', 'ZH_MIX_EN']:
            se = re.sub(r'([a-z])([A-Z])', r'\1 \2', se)
        print(f"\nSentence[{n}]: {se}")
        # Convert sentence to phones and tones
        phones, tones, lang_ids, norm_text, word2ph = get_text_for_tts_infer(se, language, symbol_to_id=_symbol_to_id)

        start = time.time()
        # Run encoder
        z_p, pronoun_lens, audio_len = sess_enc.run(None, input_feed={
                                    'phone': phones, 'g': g,
                                    'tone': tones, 'language': lang_ids, 
                                    'noise_scale': np.array([0], dtype=np.float32),
                                    'length_scale': np.array([1.0 / args.speed], dtype=np.float32),
                                    'noise_scale_w': np.array([0], dtype=np.float32),
                                    'sdp_ratio': np.array([0], dtype=np.float32)})
        print(f"encoder run take {1000 * (time.time() - start):.2f}ms")

        # 计算每个词的发音长度
        word2pronoun = calc_word2pronoun(word2ph, pronoun_lens)
        # 生成word2pronoun和zp的切片
        pn_slices, zp_slices = generate_slices(word2pronoun, dec_len)

        audio_len = audio_len[0]
        sub_audio_list = []
        for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
            zp_slice = z_p[..., zs]

            # Padding前zp的长度
            sub_dec_len = zp_slice.shape[-1]
            # Padding前输出音频的长度
            sub_audio_len = 512 * sub_dec_len

            # Padding到dec_len
            if zp_slice.shape[-1] < dec_len:
                zp_slice = np.concatenate((zp_slice, np.zeros((*zp_slice.shape[:-1], dec_len - zp_slice.shape[-1]), dtype=np.float32)), axis=-1)

            start = time.time()
            audio = sess_dec.run(None, input_feed={"z_p": zp_slice,
                                "g": g
                                })[0].flatten()
            
            # 处理overlap
            audio_start = 0
            if len(sub_audio_list) > 0:
                if pn_slices[i - 1].stop > ps.start:
                    # 去掉第一个字
                    audio_start = 512 * word2pronoun[ps.start]
    
            audio_end = sub_audio_len
            if i < len(pn_slices) - 1:
                if ps.stop > pn_slices[i + 1].start:
                    # 去掉最后一个字
                    audio_end = sub_audio_len - 512 * word2pronoun[ps.stop - 1]

            audio = audio[audio_start:audio_end]
            print(f"Decode slice[{i}]: decoder run take {1000 * (time.time() - start):.2f}ms")
            sub_audio_list.append(audio)
        sub_audio = merge_sub_audio(sub_audio_list, 0, audio_len)
        audio_list.append(sub_audio)
    audio = audio_numpy_concat(audio_list, sr=sample_rate, speed=args.speed)
    soundfile.write(args.wav, audio, sample_rate)
    print(f"Save to {args.wav}")

if __name__ == "__main__":
    main()
