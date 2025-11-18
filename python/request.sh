curl -X POST "http://127.0.0.1:8001/tts" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "sentence=爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。" \
  --output tts.wav
