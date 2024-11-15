import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.nn.functional as F

from .models import SynthesizerTrn
from .split_utils import split_sentence
from .download_utils import load_or_download_config, load_or_download_model
from .text import cleaned_text_to_sequence
from .text.cleaner import clean_text
from .commons import intersperse
import tarfile

def get_text_for_tts_infer(text, language_str, hps, device, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    if hps.data.add_blank:
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert = torch.zeros(1024, len(phone))
    ja_bert = torch.zeros(768, len(phone))
    # if getattr(hps.data, "disable_bert", False):
    #     bert = torch.zeros(1024, len(phone))
    #     ja_bert = torch.zeros(768, len(phone))
    # else:
    #     bert = get_bert(norm_text, word2ph, language_str, device)
    #     del word2ph
    #     assert bert.shape[-1] == len(phone), phone

    #     if language_str == "ZH":
    #         bert = bert
    #         ja_bert = torch.zeros(768, len(phone))
    #     elif language_str in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
    #         ja_bert = bert
    #         bert = torch.zeros(1024, len(phone))
    #     else:
    #         raise NotImplementedError()

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


class TTS(nn.Module):
    def __init__(self, 
                language,
                dec_len=128,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            dec_len=dec_len,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts
    
    def generate_data(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False):
        language = self.language
        # texts = self.split_sentences_into_pieces(text, language, quiet)
        texts = [text]
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)

        input_names = ['z_p', 'g', 'z']
        calib_dataset_root = "calibration_dataset"
        tar_files = {}
        tar_filenames = {}
        for name in input_names:
            os.makedirs(f"{calib_dataset_root}/{name}", exist_ok=True)
            tar_files[name] = tarfile.open(f"{calib_dataset_root}/{name}.tar.gz", "w:gz")
            tar_filenames[name] = f"{calib_dataset_root}/{name}.tar.gz"

        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            
            with torch.no_grad():
                g_src = self.model.emb_g(torch.IntTensor([speaker_id])).unsqueeze(-1)
                g_src.numpy().astype(np.float32).tofile("../models/g_src.bin")
                g_dst = torch.from_numpy(np.fromfile("../models/g_dst.bin", dtype=np.float32).reshape((1, 256, 1)))

                z_p, pronoun_lens, audio_len = self.model.enc_forward(
                    phones,
                    tones,
                    lang_ids,
                    g_src
                )
                audio_len = audio_len.item()

                dec_len = self.model.dec_len
                sub_audio_len = 0
                dec_slice_num = int(np.ceil(z_p.size(-1) / dec_len))

                for n in tqdm(range(dec_slice_num)):
                    z_p_slice = z_p[..., n * dec_len : (n+1) * dec_len]
                    if z_p_slice.size(-1) < dec_len:
                        z_p_slice = F.pad(z_p_slice, (0, dec_len - z_p_slice.size(-1)), mode="constant", value=0)

                    z = self.model.flow_forward(z_p_slice, g_src)

                    audio = self.model.flow_dec_forward(z, g_dst)
                    audio = audio[0, 0].data.cpu().float().numpy()
                    sub_audio_len += audio.shape[-1]

                    if sub_audio_len > audio_len:
                        audio_list.append(audio[: audio_len - audio_len // audio.shape[-1] * audio.shape[-1]])
                    else:
                        audio_list.append(audio)
                    
                    np.save(f"{calib_dataset_root}/z_p/{n}.npy", z_p_slice.numpy().astype(np.float32))
                    np.save(f"{calib_dataset_root}/g/{n}.npy", g.numpy().astype(np.float32))
                    np.save(f"{calib_dataset_root}/z/{n}.npy", z.numpy().astype(np.float32))

                    tar_files["z_p"].add(f"{calib_dataset_root}/z_p/{n}.npy")
                    tar_files["g"].add(f"{calib_dataset_root}/g/{n}.npy")
                    tar_files["z"].add(f"{calib_dataset_root}/z/{n}.npy")
                    
        for name in input_names:
            print(f"Save {tar_filenames[name]}")
            tar_files[name].close()

        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)


    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False):
        language = self.language
        # texts = self.split_sentences_into_pieces(text, language, quiet)
        texts = [text]
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)

        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            
            with torch.no_grad():
                g_src = self.model.emb_g(torch.IntTensor([speaker_id])).unsqueeze(-1)
                g_src.numpy().astype(np.float32).tofile("../models/g_src.bin")
                g_dst = torch.from_numpy(np.fromfile("../models/g_dst.bin", dtype=np.float32).reshape((1, 256, 1)))

                z_p, _, audio_len = self.model.enc_forward(
                    phones,
                    tones,
                    lang_ids,
                    g_src
                )
                audio_len = audio_len.item()

                dec_len = self.model.dec_len
                sub_audio_len = 0
                dec_slice_num = int(np.ceil(z_p.size(-1) / dec_len))

                for n in tqdm(range(dec_slice_num)):
                    z_p_slice = z_p[..., n * dec_len : (n+1) * dec_len]
                    if z_p_slice.size(-1) < dec_len:
                        z_p_slice = F.pad(z_p_slice, (0, dec_len - z_p_slice.size(-1)), mode="constant", value=0)

                    z = self.model.flow_forward(z_p_slice, g_src)

                    audio = self.model.flow_dec_forward(z, g_dst)
                    audio = audio[0, 0].data.cpu().float().numpy()
                    sub_audio_len += audio.shape[-1]

                    if sub_audio_len > audio_len:
                        audio_list.append(audio[: audio_len - audio_len // audio.shape[-1] * audio.shape[-1]])
                    else:
                        audio_list.append(audio)

        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
