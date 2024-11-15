import numpy as np
import soundfile
from axengine import InferenceSession
import onnxruntime as ort
import argparse
import time
from utils import *



def get_args():
    parser = argparse.ArgumentParser(
        prog="melotts",
        description="Run TTS on input sentence"
    )
    parser.add_argument("--sentence", "-s", type=str, required=False, default="爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用")
    parser.add_argument("--wav", "-w", type=str, required=False, default="output.wav")
    parser.add_argument("--encoder", "-e", type=str, required=False, default="../models/encoder.onnx")
    parser.add_argument("--flow", "-f", type=str, required=False, default="../models/flow.axmodel")
    parser.add_argument("--decoder", "-d", type=str, required=False, default="../models/decoder.axmodel")
    parser.add_argument("--sample_rate", "-sr", type=int, required=False, default=44100)
    parser.add_argument("--speed", type=float, required=False, default=0.8)
    parser.add_argument("--lexicon", type=str, required=False, default="../models/lexicon.txt")
    parser.add_argument("--token", type=str, required=False, default="../models/tokens.txt")
    parser.add_argument("--g_src", type=str, required=False, default="../models/g_src.bin")
    parser.add_argument("--g_dst", type=str, required=False, default="../models/g_dst.bin")
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


def generate_pronounce_slice(yinjie_num):
    # 按照每个词的音节数产生发音切片
    pron_slice = []
    start = 0
    end = 0
    for i, n in enumerate(yinjie_num):
        end = start + n
        pron_slice.append(slice(start, end))
        start = end
    return pron_slice


def generate_word_pron_num(pron_lens, pron_slices):
    # 求每个词语的发音长度
    pron_num = []
    for s in pron_slices:
        pron_num.append(pron_lens[s].sum())
    return pron_num


def decode_long_word(sess_flow, sess_dec, z_p, g_src, g_dst, dec_len):
    z_p_len = z_p.shape[-1]
    slice_num = int(np.ceil(z_p_len / dec_len))
    sub_audio_list = []

    for i in range(slice_num):
        z_p_slice = z_p[..., i * dec_len : (i + 1) * dec_len]
        sub_dec_len = z_p_slice.shape[-1]
        sub_audio_len = 512 * sub_dec_len

        if z_p_slice.shape[-1] < dec_len:
            z_p_slice = np.concatenate((z_p_slice, np.zeros((*z_p_slice.shape[:-1], dec_len - z_p_slice.shape[-1]), dtype=np.float32)), axis=-1)

        start = time.time()
        z = sess_flow.run(input_feed={"z_p": z_p_slice,
                            "g": g_src
                            })["z"].flatten()
        print(f"Long word slice[{i}]: flow run take {1000 * (time.time() - start):.2f}ms")

        start = time.time()
        audio = sess_dec.run(input_feed={"z_p": z,
                            "g": g_dst
                            })["audio"].flatten()
        audio = audio[:sub_audio_len]
        print(f"Long word slice[{i}]: decoder run take {1000 * (time.time() - start):.2f}ms")
        sub_audio_list.append(audio)
    return sub_audio_list


def generate_decode_slices(pron_num, dec_len):
    pron_slices = []
    zp_slices = []
    strip_flags = [] # head tail
    pron_lens = []
    is_long = []

    start = end = 0
    zp_start, zp_end = 0, 0
    while (end < len(pron_num)):
        if len(pron_lens) > 0 and pron_lens[-1] > 2 and pron_num[end] < dec_len:
            prev_end = pron_slices[-1][1]
            pad_size = pron_num[prev_end - 1] + pron_num[prev_end - 2]
            start = end - 2
            zp_start = zp_end - pad_size
            strip_head, strip_tail = True, False
            if is_long[-1]:
                strip_flags[-1][1] = False
            else:
                strip_flags[-1][1] = True
        else:
            pad_size = 0
            start = end
            zp_start = zp_end
            strip_head, strip_tail = False, False
            if len(strip_flags) > 0:
                strip_flags[-1][1] = False

        sub_dec_len = abs(zp_end - zp_start)
        while (end < len(pron_num) and sub_dec_len < dec_len):
            sub_dec_len += pron_num[end]
            end += 1

        if end - start == 1 and sub_dec_len > dec_len:
            # Long word
            is_long.append(True)
            if len(strip_flags) > 0:
                strip_flags[-1][1] = False
        else:
            is_long.append(False)
            if sub_dec_len > dec_len:
                sub_dec_len -= pron_num[end - 1]
                end -= 1

        zp_end = zp_start + sub_dec_len

        pron_slices.append([start, end])
        zp_slices.append([zp_start, zp_end])
        strip_flags.append([strip_head, strip_tail])
        pron_lens.append(end - start)

    return pron_slices, zp_slices, strip_flags, pron_lens, is_long


def main():
    args = get_args()
    sentence = args.sentence
    sample_rate = args.sample_rate
    lexicon_filename = args.lexicon
    token_filename = args.token
    enc_model = args.encoder # default="../models/encoder.onnx"
    flow_model = args.flow   # default="../models/flow.axmodel"
    dec_model = args.decoder # default="../models/decoder.axmodel"
    g_src_filename = args.g_src
    g_dst_filename = args.g_dst

    print(f"sentence: {sentence}")
    print(f"sample_rate: {sample_rate}")
    print(f"lexicon: {lexicon_filename}")
    print(f"token: {token_filename}")
    print(f"encoder: {enc_model}")
    print(f"flow: {flow_model}")
    print(f"decoder: {dec_model}")
    print(f"g_src: {g_src_filename}")
    print(f"g_dst: {g_dst_filename}")

    # Split sentence
    sens = split_sentences_zh(sentence)

    # Load lexicon
    lexicon = Lexicon(lexicon_filename, token_filename)

    # Load models
    start = time.time()
    sess_enc = ort.InferenceSession(enc_model, providers=["CPUExecutionProvider"], sess_options=ort.SessionOptions())
    sess_flow = InferenceSession.load_from_model(flow_model)
    sess_dec = InferenceSession.load_from_model(dec_model)
    dec_len = sess_dec.get_output_shapes()[0][-1] // 512
    print(f"load models take {1000 * (time.time() - start)}ms")

    # Load static input
    g_src = np.fromfile(g_src_filename, dtype=np.float32).reshape(1, 256, 1)
    g_dst = np.fromfile(g_dst_filename, dtype=np.float32).reshape(1, 256, 1)

    # Final wav
    audio_list = []

    # Iterate over splitted sentences
    for n, se in enumerate(sens):
        print(f"\nSentence[{n}]: {se}")
        # Convert sentence to phones and tones
        phone_str, yinjie_num, phones, tones = lexicon.convert(se)

        # Add blank between words
        phone_str = intersperse(phone_str, 0)
        phones = np.array(intersperse(phones, 0), dtype=np.int32)
        tones = np.array(intersperse(tones, 0), dtype=np.int32)
        yinjie_num = np.array(yinjie_num, dtype=np.int32) * 2
        yinjie_num[0] += 1
        assert (yinjie_num.sum() == phones.shape[0])
        pron_slices = generate_pronounce_slice(yinjie_num)

        # for s in pron_slices:
        #     print(phone_str[s])
        
        phone_len = phones.shape[-1]

        language = np.array([3] * phone_len, dtype=np.int32)

        start = time.time()
        z_p, pronoun_lens, audio_len = sess_enc.run(None, input_feed={
                                    'phone': phones, 'g': g_src,
                                    'tone': tones, 'language': language, 
                                    'noise_scale': np.array([0], dtype=np.float32),
                                    'length_scale': np.array([1.0 / args.speed], dtype=np.float32),
                                    'noise_scale_w': np.array([0], dtype=np.float32),
                                    'sdp_ratio': np.array([0], dtype=np.float32)})
        print(f"encoder run take {1000 * (time.time() - start):.2f}ms")
        
        audio_len = audio_len[0]
        actual_size = z_p.shape[-1]
        dec_slice_num = int(np.ceil(actual_size / dec_len))
        # print(f"origin z_p.shape: {z_p.shape}")
        z_p = np.pad(z_p, pad_width=((0,0),(0,0),(0, dec_slice_num * dec_len - actual_size)), mode="constant", constant_values=0)

        # print(f"phone_len: {phone_len}")
        # print(f"z_p.shape: {z_p.shape}")
        # print(f"dec_slice_num: {dec_slice_num}")
        # print(f"audio_len: {audio_len}")

        # 生成每个词的发音数量
        pron_num = generate_word_pron_num(pronoun_lens, pron_slices)
        # assert (sum(pron_num) == pronoun_lens.sum())

        sub_audio_list = []
        pron_num_slices, zp_slices, strip_flags, pron_lens, is_long = \
            generate_decode_slices(pron_num, dec_len)
        
        for i in range(len(pron_num_slices)):
            pron_start, pron_end = pron_num_slices[i]
            zp_start, zp_end = zp_slices[i]

            phone_strs = []
            for n in range(pron_start, pron_end):
                phone_strs.extend(phone_str[pron_slices[n]])
            # print(f"phone str: {phone_strs}")

            sub_dec_len = zp_end - zp_start
            sub_audio_len = 512 * sub_dec_len

            zp_slice = z_p[..., zp_start : zp_end]
            
            if is_long[i]:
                sub_audio_list.extend(decode_long_word(sess_flow, sess_dec, zp_slice, g_src, g_dst, dec_len))
            else:
                if zp_slice.shape[-1] < dec_len:
                    zp_slice = np.concatenate((zp_slice, np.zeros((*zp_slice.shape[:-1], dec_len - zp_slice.shape[-1]), dtype=np.float32)), axis=-1)

                start = time.time()
                z = sess_flow.run(input_feed={"z_p": zp_slice,
                                    "g": g_src
                                    })["z"].flatten()
                print(f"Sentence[{n}] Slice[{i}]: flow run take {1000 * (time.time() - start):.2f}ms")

                start = time.time()
                audio = sess_dec.run(input_feed={"z_p": z,
                                    "g": g_dst
                                    })["audio"].flatten()
                audio = audio[:sub_audio_len]
                print(f"Sentence[{n}] Slice[{i}]: decoder run take {1000 * (time.time() - start):.2f}ms")

                # start = time.time()
                # audio = sess_dec.run(input_feed={"z_p": zp_slice,
                #                     "g": g_src
                #                     })["audio"].flatten()
                # audio = audio[:sub_audio_len]
                # print(f"Sentence[{n}] Slice[{i}]: decoder run take {1000 * (time.time() - start):.2f}ms")

                if strip_flags[i][0]:
                    # strip head
                    head = 512 * pron_num[pron_start]
                    # print(f"Strip head: {phone_str[pron_slices[pron_start]]}")
                    audio = audio[head : ]
                if strip_flags[i][1]:
                    # strip tail
                    tail = 512 * pron_num[pron_end - 1]
                    # print(f"Strip tail: {phone_str[pron_slices[pron_end - 1]]}")
                    audio = audio[: -tail]

                sub_audio_list.append(audio)

        sub_audio = merge_sub_audio(sub_audio_list, 0, audio_len)
        audio_list.append(sub_audio)

    audio = audio_numpy_concat(audio_list, sr=sample_rate, speed=args.speed)
    soundfile.write(args.wav, audio, sample_rate)
    print(f"Save to {args.wav}")

if __name__ == "__main__":
    main()
