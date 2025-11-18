import argparse
import os
from melotts import MeloTTS
import soundfile

def main():
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
    
    args = parser.parse_args()
    
    
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

    melotts = MeloTTS(enc_model, dec_model, language, dec_len)

    audio = melotts.run(sentence, speed=args.speed, sample_rate=sample_rate)
    soundfile.write(args.wav, audio, sample_rate)
    print(f"Save to {args.wav}")

if __name__ == "__main__":
    main()
