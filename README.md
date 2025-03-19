# melotts.axera

MeloTTS DEMO on Axera AX650

- 目前模型分成了 encoder、decoder 两部分，encoder 部分尚未转成 axmodel（目前通过 onnxruntime 运行） 

## 模型转换

预转换好的模型（中文、英语、日语）可通过脚本download_models.sh下载  
如需自行转换请参考：[模型转换](./model_convert/README.md)

## 上板部署

- AX650N 的设备已预装 Ubuntu22.04
- 以 root 权限登陆 AX650N 的板卡设备
- 链接互联网，确保 AX650N 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备：AX650N DEMO Board、爱芯派Pro

### 添加中文输入支持

执行以下命令，正确安装中文输入法之后，重启终端登录

```
locale-gen C.utf8
update-locale LANG=C.utf8
```

### Python API 运行

#### Requirements

```
cp -rf nltk_data /root
apt-get install libsndfile1-dev libmecab-dev
mkdir /opt/site-packages 
pip3 install -r requirements.txt --prefix=/opt/site-packages
```

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后，重新连接终端或者执行 `source ~/.bashrc`

```
export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

登陆开发板后

```
git clone https://github.com/ml-inory/melotts.axera.git
cd python  
python3 melotts.py -s 要生成语音的句子
```  

输入命令

```
python3 melotts.py -s 爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用
```

输出音频

https://github.com/user-attachments/assets/eda5c10c-7d30-46e5-a56a-f6edcf7813af


详细的运行参数：  
| 参数名称 | 说明 | 默认值 |
| --- | --- | --- |
| -s/--sentence | 输入句子 | |
| -w/--wav | 输出音频路径，wav格式 | output.wav |
| -e/--encoder | encoder模型路径 | ../models/encoder.onnx |
| -d/--decoder | decoder模型路径 | ../models/decoder.axmodel |
| -sr/--sample_rate | 采样率 | 44100 |
| --speed | 语速，越大表示越快 | 0.8 |
| --language | 从"ZH", "ZH_MIX_EN", "JP", "EN", 'KR', "SP", "FR"选择，分别对应中文、中英混合、日语、英语、韩语、西班牙语，法语 | ZH_MIX_EN

### CPP API 运行

#### 交叉编译

下载BSP

```
bash download_bsp.sh
```

编译

```
bash build.sh
```

#### 运行

```
./install/bin/melotts -l ../models/melo_lexicon_zh.txt -t ../models/melo_tokens.txt -e ../models/enc-sim.onnx -f ../models/flow.axmodel -d ../models/decoder.axmodel --g ../models/g.bin -w test_cn.wav -s 爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用
```

## 技术讨论

- Github issues
- QQ 群: 139953715
