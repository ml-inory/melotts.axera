# 模型转换

## 创建虚拟环境

```
conda create -n melotts python=3.9  
conda activate melotts
```

## 安装依赖

```
git clone https://github.com/ml-inory/melotts.axera.git
cd model_convert 
sudo apt install libmecab-dev
pip install unidic-lite fugashi
pip install -r requirements.txt
```  

## 转换模型(PyTorch -> ONNX)

```
python convert.py
```  
运行参数说明：
| 参数名称 | 说明 | 默认值 |
| --- | --- | --- |
| -l/--language | 转换哪个语言的模型，目前支持EN, FR, JP, ES, ZH, KR，分别对应英语、法语、日语、西班牙语、中文、韩语 | ZH |
| --dec_len | decoder输入长度 | 128 |
| --enc_max_len | encoder 输出的 duration-expanded `z_p` 最大帧长 | 1024 |


## 转换模型(ONNX -> Axera)

```
pulsar2 build --input decoder.onnx --config config_decoder_u16.json --output_dir decoder --output_name decoder.axmodel --target_hardware AX650 --npu_mode NPU3 --compiler.check 0
```  

转换完成后会生成 decoder/decoder.axmodel
