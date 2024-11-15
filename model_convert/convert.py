import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from melotts.download_utils import load_or_download_config, load_or_download_model
from melotts.tts import TTS
import torch
import onnx, onnxsim
import json

TEXT = {
    "ZH": "爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用",
    "JP": "海の向こうには何があるの？"
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="ZH",
        choices=["EN", "FR", "JP", "ES", "ZH", "KR"],
        help="target language for TTS",
    )
    parser.add_argument(
        "-d",
        "--dec_len",
        type=int,
        default=128,
        help="decoder input length",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    language = args.language
    config_path = "config.json"
    ckpt_path = "checkpoint.pth"
    device = "cpu"
    if not os.path.exists(config_path):
        load_or_download_config(locale=language)
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
            support_langs = config["data"]["spk2id"].keys()
        if language not in support_langs:
            # Force redownload
            load_or_download_config(locale=language)
            load_or_download_model(locale=language, device=device)    

    if not os.path.exists(ckpt_path):
        load_or_download_model(locale=language, device=device)

    with open(config_path, "r") as f:
        config = json.load(f)
        speaker_id = config["data"]["spk2id"][language]

        print(f"speaker_id: {speaker_id}")

    tts = TTS(language=language, dec_len=args.dec_len, config_path=config_path, ckpt_path=ckpt_path, device=device)

    print(f"Generating calibration dataset...")
    text = TEXT[language]
    print(f"text: {text}")
    tts.generate_data(text=text, speaker_id=speaker_id)

    with torch.no_grad():
        phone_len = 256
        phones = torch.zeros(phone_len, dtype=torch.int32)
        tones = torch.randint(1, 5, size=(phone_len,), dtype=torch.int32)
        g = torch.rand(1, 256, 1)
        lang_ids = torch.zeros(phone_len, dtype=torch.int32) + 3
        noise_scale = torch.FloatTensor([0.667])
        noise_scale_w = torch.FloatTensor([0.8])
        length_scale = torch.FloatTensor([1])
        sdp_ratio = torch.FloatTensor([0])

        inputs = (
            phones, tones, lang_ids, g, noise_scale, noise_scale_w, length_scale, sdp_ratio
        )
        input_names = ['phone','tone', 'language', 'g', 'noise_scale', 'noise_scale_w', 'length_scale', 'sdp_ratio']
        dynamic_axes = {
            "phone": {0: "phone_len"},
            "tone": {0: "phone_len"},
            "language": {0: "phone_len"},
        }
        # Export encoder
        encoder_name = "encoder.onnx"
        tts.model.forward = tts.model.enc_forward
        torch.onnx.export(tts.model,               # model being run
                        inputs,                    # model input (or a tuple for multiple inputs)
                        encoder_name,              # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        dynamic_axes=dynamic_axes,
                        input_names = input_names, # the model's input names
                        output_names = ['z_p', 'pronoun_lens', 'audio_len'], # the model's output names
                        )
        sim_model,_ = onnxsim.simplify(encoder_name)
        onnx.save(sim_model, encoder_name)
        print(f"Export encoder to {encoder_name}")

        # Export flow
        dec_len = tts.model.dec_len
        inputs = (
            torch.rand(1, 192, dec_len), g
        )
        model_name = "flow.onnx"
        tts.model.forward = tts.model.flow_forward
        torch.onnx.export(tts.model,               # model being run
                        inputs,                    # model input (or a tuple for multiple inputs)
                        model_name,              # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['z_p', 'g'],# the model's input names
                        output_names = ['z'],  # the model's output names
                        )
        sim_model,_ = onnxsim.simplify(model_name)
        onnx.save(sim_model, model_name)
        print(f"Export flow to {model_name}")

        # Export decoder
        dec_len = tts.model.dec_len
        inputs = (
            torch.rand(1, 192, dec_len), g
        )
        decoder_name = "decoder.onnx"
        tts.model.forward = tts.model.flow_dec_forward
        torch.onnx.export(tts.model,               # model being run
                        inputs,                    # model input (or a tuple for multiple inputs)
                        decoder_name,              # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['z_p', 'g'],# the model's input names
                        output_names = ['audio'],  # the model's output names
                        )
        sim_model,_ = onnxsim.simplify(decoder_name)
        onnx.save(sim_model, decoder_name)
        print(f"Export decoder to {decoder_name}")


if __name__ == "__main__":
    main()