import requests

url = "http://127.0.0.1:8001/tts"
data = {
    "sentence": "你好，我是一个文本转语音的测试。",
    "language": "ZH",
    "speed": "0.8",
    "sample_rate": "44100",
}

resp = requests.post(url, data=data)
if resp.status_code == 200:
    with open("tts_output.wav", "wb") as f:
        f.write(resp.content)
    print("Saved to tts_output.wav")
else:
    print("Error:", resp.status_code, resp.text)
