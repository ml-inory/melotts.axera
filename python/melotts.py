import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import onnxruntime as ort
import axengine as axe
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
    word2ph = [i * 2 for i in word2ph]
    word2ph[0] += 1

    phone = np.array(phone, dtype=np.int32)
    tone = np.array(tone, dtype=np.int32)
    language = np.array(language, dtype=np.int32)
    word2ph = np.array(word2ph, dtype=np.int32)

    return phone, tone, language, norm_text, word2ph

def split_sentences_into_pieces(text, language, quiet=False):
    texts = split_sentence(text, language_str=language)
    if not quiet:
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
    return texts


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


class AxBertFeatureExtractor:
    def __init__(self, model_path, max_token_len=256, model_id="hfl/chinese-roberta-wwm-ext-large"):
        self.max_token_len = max_token_len
        self.sess = axe.InferenceSession(model_path)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BERT AXMODEL only replaces BERT inference. Tokenization still requires "
                "`transformers`; install python/requirements.txt."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _tokenize(self, norm_text):
        batch = self.tokenizer(
            norm_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_token_len,
        )
        input_ids = np.asarray([batch["input_ids"]], dtype=np.int32)
        attention_mask = np.asarray([batch["attention_mask"]], dtype=np.int32)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros((self.max_token_len,), dtype=np.int32)
        token_type_ids = np.asarray([token_type_ids], dtype=np.int32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def __call__(self, norm_text, word2ph, phone_len):
        if len(word2ph) > self.max_token_len:
            raise RuntimeError(
                f"BERT token length {len(word2ph)} exceeds AXMODEL max_token_len={self.max_token_len}"
            )

        start = time.time()
        hidden = self.sess.run(None, input_feed=self._tokenize(norm_text))[0].astype(np.float32)
        print(f"bert axmodel run take {1000 * (time.time() - start):.2f}ms")

        hidden = hidden.reshape(1, self.max_token_len, 1024)[0]
        phone_features = [
            np.repeat(hidden[i : i + 1], int(repeat), axis=0)
            for i, repeat in enumerate(word2ph)
        ]
        bert = np.concatenate(phone_features, axis=0).T
        if bert.shape[-1] != phone_len:
            raise RuntimeError(f"BERT seq len {bert.shape[-1]} != phone len {phone_len}")
        return bert.reshape(1, 1024, phone_len).astype(np.float32)

class MeloTTS:
    def __init__(self, enc_model, dec_model, language, dec_len, bert_model=None, bert_model_id="hfl/chinese-roberta-wwm-ext-large"):
        self.language = language
        if self.language == "ZH":
            self.language = "ZH_MIX_EN"
        self.dec_len = dec_len
        self._symbol_to_id = {s: i for i, s in enumerate(LANG_TO_SYMBOL_MAP[language])}

        self.sess_enc = ort.InferenceSession(enc_model, providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
        self.enc_input_names = {inp.name for inp in self.sess_enc.get_inputs()}
        self.needs_bert = bool({"bert", "ja_bert"} & self.enc_input_names)
        if self.needs_bert and bert_model is None:
            raise RuntimeError(
                "Encoder requires `bert/ja_bert` inputs, but no BERT AXMODEL was provided. "
                "Pass `--bert ../models/bert-hidden-u16-zh.axmodel`. CPU BERT fallback is disabled."
            )
        self.sess_bert = None
        if self.needs_bert:
            if self.language != "ZH_MIX_EN":
                raise RuntimeError("BERT AXMODEL path currently supports Chinese/ZH_MIX_EN only.")
            self.sess_bert = AxBertFeatureExtractor(bert_model, model_id=bert_model_id)
        self.sess_dec = axe.InferenceSession(dec_model)
        
        
        self.g = np.fromfile(f"../models/g-{language.lower()}.bin", dtype=np.float32).reshape(1, 256, 1)

    
    def run(self, text, speed=0.8, sample_rate=44100):
        audio_list = []
        sens = split_sentences_into_pieces(text, self.language, quiet=False)
        
        for n, se in enumerate(sens):
            if self.language in ['EN', 'ZH_MIX_EN']:
                se = re.sub(r'([a-z])([A-Z])', r'\1 \2', se)
            print(f"\nSentence[{n}]: {se}")
            # Convert sentence to phones and tones
            phones, tones, lang_ids, norm_text, word2ph = get_text_for_tts_infer(
                se, self.language, symbol_to_id=self._symbol_to_id
            )

            start = time.time()
            # Run encoder
            enc_inputs = {
                                        'phone': phones, 'g': self.g,
                                        'tone': tones, 'language': lang_ids,
                                        'noise_scale': np.array([0], dtype=np.float32),
                                        'length_scale': np.array([1.0 / speed], dtype=np.float32),
                                        'noise_scale_w': np.array([0], dtype=np.float32),
                                        'sdp_ratio': np.array([0], dtype=np.float32)}
            if "bert" in self.enc_input_names:
                enc_inputs["bert"] = self.sess_bert(norm_text, word2ph, len(phones))
            if "ja_bert" in self.enc_input_names:
                enc_inputs["ja_bert"] = np.zeros((1, 768, len(phones)), dtype=np.float32)
            z_p, pronoun_lens, audio_len = self.sess_enc.run(None, input_feed=enc_inputs)
            print(f"encoder run take {1000 * (time.time() - start):.2f}ms")

            required_zp_len = int(np.sum(pronoun_lens))
            if required_zp_len > z_p.shape[-1]:
                raise RuntimeError(
                    f"encoder z_p length {z_p.shape[-1]} is shorter than duration-expanded "
                    f"length {required_zp_len}; re-export encoder with a larger --enc_max_len"
                )

            # 计算每个词的发音长度
            word2pronoun = calc_word2pronoun(word2ph, pronoun_lens)
            # 生成word2pronoun和zp的切片
            pn_slices, zp_slices = generate_slices(word2pronoun, self.dec_len)

            audio_len = audio_len[0]
            sub_audio_list = []
            for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
                zp_slice = z_p[..., zs]

                # Padding前zp的长度
                sub_dec_len = zp_slice.shape[-1]
                # Padding前输出音频的长度
                sub_audio_len = 512 * sub_dec_len

                # Padding到dec_len
                if zp_slice.shape[-1] < self.dec_len:
                    zp_slice = np.concatenate((zp_slice, np.zeros((*zp_slice.shape[:-1], self.dec_len - zp_slice.shape[-1]), dtype=np.float32)), axis=-1)

                start = time.time()
                audio = self.sess_dec.run(None, input_feed={"z_p": zp_slice,
                                    "g": self.g
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
        audio = audio_numpy_concat(audio_list, sr=sample_rate, speed=speed)
        return audio
