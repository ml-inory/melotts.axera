import argparse
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import soundfile as sf
from melotts import MeloTTS


# 简单的模型缓存，避免重复加载
_tts_cache = {}


def resolve_models(language: str, enc_model: str | None, dec_model: str | None):
    """
    按照你原来的 demo 逻辑，自动推断 encoder/decoder 的路径。
    """
    lang = language
    if lang == "ZH":
        lang = "ZH_MIX_EN"

    # 处理 encoder
    if enc_model is None:
        if "ZH" in lang:
            enc_model = "../models/encoder-zh.onnx"
        else:
            enc_model = f"../models/encoder-{lang.lower()}.onnx"
    if not os.path.exists(enc_model):
        raise FileNotFoundError(f"Encoder model ({enc_model}) not exist!")

    # 处理 decoder
    if dec_model is None:
        if "ZH" in lang:
            dec_model = "../models/decoder-zh.axmodel"
        else:
            dec_model = f"../models/decoder-{lang.lower()}.axmodel"
    if not os.path.exists(dec_model):
        raise FileNotFoundError(f"Decoder model ({dec_model}) not exist!")

    return lang, enc_model, dec_model


def get_tts(language: str, enc_model: str | None, dec_model: str | None, dec_len: int):
    """
    获取 MeloTTS 实例（带缓存）
    key 只跟 enc/dec/lang/dec_len 有关，speed 和 sample_rate 是调用时决定的
    """
    lang, enc_model_resolved, dec_model_resolved = resolve_models(language, enc_model, dec_model)
    key = (lang, enc_model_resolved, dec_model_resolved, dec_len)

    if key not in _tts_cache:
        print(f"Loading MeloTTS: lang={lang}, encoder={enc_model_resolved}, decoder={dec_model_resolved}, dec_len={dec_len}")
        _tts_cache[key] = MeloTTS(enc_model_resolved, dec_model_resolved, lang, dec_len)
    return _tts_cache[key]


class TTSServerHandler(BaseHTTPRequestHandler):

    def _send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/tts":
            self._send_json({"error": "not found"}, 404)
            return

        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)

        # 支持两种：x-www-form-urlencoded / json
        params = {}
        try:
            if "application/json" in content_type:
                params = json.loads(body.decode("utf-8"))
            elif "application/x-www-form-urlencoded" in content_type:
                qs = parse_qs(body.decode("utf-8"))
                params = {k: v[0] for k, v in qs.items()}
            else:
                self._send_json({"error": "Unsupported Content-Type"}, 400)
                return
        except Exception as e:
            self._send_json({"error": f"Failed to parse body: {e}"}, 400)
            return

        # 必填：sentence
        sentence = params.get("sentence")
        if not sentence:
            self._send_json({"error": "Field 'sentence' is required"}, 400)
            return
        print(f"Request: sentence={sentence}")
        # 可选参数
        language = params.get("language", "ZH")
        enc_model = params.get("encoder", "../models/encoder-onnx/encoder-zh.onnx")  # 可以为 None
        dec_model = params.get("decoder", "../models/decoder-ax650/decoder-zh.axmodel")  # 可以为 None

        try:
            sample_rate = int(params.get("sample_rate", 44100))
        except ValueError:
            self._send_json({"error": "sample_rate must be int"}, 400)
            return

        try:
            speed = float(params.get("speed", 0.8))
        except ValueError:
            self._send_json({"error": "speed must be float"}, 400)
            return

        try:
            dec_len = int(params.get("dec_len", 128))
        except ValueError:
            self._send_json({"error": "dec_len must be int"}, 400)
            return

        # 生成音频
        try:
            tts = get_tts(language, enc_model, dec_model, dec_len)
            audio = tts.run(sentence, speed=speed, sample_rate=sample_rate)
        except Exception as e:
            self._send_json({"error": f"TTS failed: {e}"}, 500)
            return

        # 使用 soundfile 写入内存中的 wav
        try:
            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, format="WAV")
            wav_bytes = buf.getvalue()
        except Exception as e:
            self._send_json({"error": f"Failed to encode WAV: {e}"}, 500)
            return

        # 返回二进制 wav
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        # 方便浏览器下载，也可以改为 inline
        self.send_header("Content-Disposition", 'attachment; filename="tts.wav"')
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.end_headers()
        self.wfile.write(wav_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    host = "0.0.0.0"
    port = args.port
    server = HTTPServer((host, port), TTSServerHandler)
    print(f"TTS Server started at http://{host}:{port}")
    server.serve_forever()
