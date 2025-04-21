/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <numeric>

#include "cmdline.hpp"
#include "OnnxWrapper.hpp"
#include <ax_sys_api.h>
#include "EngineWrapper.hpp"
#include "AudioFile.h"
#include "Lexicon.hpp"
#include "split_utils.hpp"

using namespace std;

static std::vector<int> intersperse(const std::vector<int>& lst, int item) {
    std::vector<int> result(lst.size() * 2 + 1, item);
    for (size_t i = 1; i < result.size(); i+=2) {
        result[i] = lst[i / 2];
    }
    return result;
}

static int calc_product(const std::vector<int64_t>& dims) {
    int64_t result = 1;
    for (auto i : dims)
        result *= i;
    return result;
}

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}


// 计算每个词的发音时长
vector<int> calc_word2pronoun(const vector<int>& word2ph, const vector<int>& pronoun_lens) {
    vector<int> indice = {0};
    for (size_t i = 0; i < word2ph.size() - 1; ++i) {
        indice.push_back(indice.back() + word2ph[i]);
    }
    
    vector<int> word2pronoun;
    for (size_t i = 0; i < word2ph.size(); ++i) {
        int start = indice[i];
        int end = start + word2ph[i];
        int sum = accumulate(pronoun_lens.begin() + start, pronoun_lens.begin() + end, 0);
        word2pronoun.push_back(sum);
    }
    return word2pronoun;
}

struct Slice {
    int start;
    int end;
    Slice(int s, int e) : start(s), end(e) {}
};

// 生成有overlap的slice，slice索引是对于zp的
pair<vector<Slice>, vector<Slice>> generate_slices(const vector<int>& word2pronoun, int dec_len) {
    int pn_start = 0, pn_end = 0;
    int zp_start = 0, zp_end = 0;
    int zp_len = 0;
    vector<Slice> pn_slices;
    vector<Slice> zp_slices;
    
    while (pn_end < static_cast<int>(word2pronoun.size())) {
        // 检查是否可以向前overlap两个字
        if (pn_end - pn_start > 2 && 
            accumulate(word2pronoun.begin() + pn_end - 2, word2pronoun.begin() + pn_end + 1, 0) <= dec_len) {
            zp_len = accumulate(word2pronoun.begin() + pn_end - 2, word2pronoun.begin() + pn_end, 0);
            zp_start = zp_end - zp_len;
            pn_start = pn_end - 2;
        } else {
            zp_len = 0;
            zp_start = zp_end;
            pn_start = pn_end;
        }
        
        while (pn_end < static_cast<int>(word2pronoun.size()) && 
               zp_len + word2pronoun[pn_end] <= dec_len) {
            zp_len += word2pronoun[pn_end];
            pn_end++;
        }
        
        zp_end = zp_start + zp_len;
        pn_slices.emplace_back(pn_start, pn_end);
        zp_slices.emplace_back(zp_start, zp_end);
    }
    
    return make_pair(pn_slices, zp_slices);
}

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("encoder", 'e', "encoder onnx", false, "");
    cmd.add<std::string>("decoder", 'd', "decoder axmodel", false, "");
    cmd.add<std::string>("lexicon", 'l', "lexicon.txt", false, "../models/lexicon.txt");
    cmd.add<std::string>("token", 't', "tokens.txt", false, "../models/tokens.txt");
    cmd.add<std::string>("g", 0, "g.bin", false, "");
    cmd.add<std::string>("language", 0, "language, choose from ZH, EN, JP", false, "ZH");

    cmd.add<std::string>("sentence", 's', "input sentence", false, "爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。服务智慧城市、智能驾驶、机器人的海量普惠的应用");
    cmd.add<std::string>("wav", 'w', "wav file", false, "output.wav");

    cmd.add<float>("speed", 0, "speak speed", false, 0.8f);
    cmd.add<int>("sample_rate", 0, "sample rate", false, 44100);
    cmd.parse_check(argc, argv);

    auto encoder_file   = cmd.get<std::string>("encoder");
    auto decoder_file   = cmd.get<std::string>("decoder");
    auto lexicon_file   = cmd.get<std::string>("lexicon");
    auto token_file     = cmd.get<std::string>("token");
    auto g_file         = cmd.get<std::string>("g");
    auto language       = cmd.get<std::string>("language");

    auto sentence       = cmd.get<std::string>("sentence");
    auto wav_file       = cmd.get<std::string>("wav");

    auto speed          = cmd.get<float>("speed");
    auto sample_rate    = cmd.get<int>("sample_rate");

    std::string lower_lang = language;
    std::transform(language.begin(), language.end(), lower_lang.begin(),
        [](unsigned char c){ return std::tolower(c); });
    if (encoder_file.empty()) {
        encoder_file = "../models/encoder-" + lower_lang + ".onnx";
    }
    if (decoder_file.empty()) {
        decoder_file = "../models/decoder-" + lower_lang + ".axmodel";
    }
    if (g_file.empty()) {
        if (lower_lang == "zh") {
            g_file = "../models/g-zh_mix_en.bin";
        } else {
            g_file = "../models/g-" + lower_lang + ".bin";
        }
    }

    printf("encoder: %s\n", encoder_file.c_str());
    printf("decoder: %s\n", decoder_file.c_str());
    printf("lexicon: %s\n", lexicon_file.c_str());
    printf("token: %s\n", token_file.c_str());
    printf("g: %s\n", g_file.c_str());
    printf("language: %s\n", language.c_str());
    printf("sentence: %s\n", sentence.c_str());
    printf("wav: %s\n", wav_file.c_str());
    printf("speed: %f\n", speed);
    printf("sample_rate: %d\n", sample_rate);

    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }

    // Load lexicon
    Lexicon lexicon(lexicon_file, token_file);

    // Read g.bin
    std::vector<float> g(256, 0);
    FILE* fp = fopen(g_file.c_str(), "rb");
    if (!fp) {
        printf("Open %s failed!\n", g_file.c_str());
        return -1;
    }
    fread(g.data(), sizeof(float), g.size(), fp);
    fclose(fp);
	
    double start, end;

    start = get_current_time();
    OnnxWrapper encoder;
    if (0 != encoder.Init(encoder_file)) {
        printf("encoder init failed!\n");
        return -1;
    }
    end = get_current_time();
    printf("Load encoder take %.2f ms\n", (end - start));	
	
    start = get_current_time();	
    EngineWrapper decoder_model;
    if (0 != decoder_model.Init(decoder_file.c_str())) {
        printf("Init decoder model failed!\n");
        return -1;
    }
    end = get_current_time();
    printf("Load decoder take %.2f ms\n", (end - start));	

    float noise_scale   = 0.3f;
    float length_scale  = 1.0 / speed;
    float noise_scale_w = 0.6f;
    float sdp_ratio     = 0.2f;

    // Split sentences
    auto sens = split_sentence(sentence, 10, language);
    std::vector<float> wavlist;

    for (auto& se : sens) {
        printf("\nSplit sentence: %s\n", se.c_str());
        // Convert sentence to phones and tones
        std::vector<int> phones_bef, tones_bef, word2ph;
        lexicon.convert(se, phones_bef, tones_bef, word2ph);

        // Add blank between words
        auto phones = intersperse(phones_bef, 0);
        auto tones = intersperse(tones_bef, 0);
        for (int& i : word2ph) {
            i *= 2;
        }
        if (!word2ph.empty())
            word2ph[0] += 1;

        int phone_len = phones.size();

        std::vector<int> langids(phone_len, 3);
        
        // Run encoder
        start = get_current_time();
        auto encoder_output = encoder.Run(phones, tones, langids, g, noise_scale, noise_scale_w, length_scale, sdp_ratio);
        float* zp_data = encoder_output.at(0).GetTensorMutableData<float>();
        int* pronoun_lens_data = encoder_output.at(1).GetTensorMutableData<int>();
        int audio_len = encoder_output.at(2).GetTensorMutableData<int>()[0];
        auto zp_info = encoder_output.at(0).GetTensorTypeAndShapeInfo();
        auto zp_shape = zp_info.GetShape();
        std::vector<int> pronoun_lens(pronoun_lens_data, pronoun_lens_data + phone_len);
        end = get_current_time();
        printf("Encoder run take %.2f ms\n", (end - start));

        int zp_size = decoder_model.GetInputSize(0) / sizeof(float);
        int dec_len = zp_size / zp_shape[1];
        int audio_slice_len = decoder_model.GetOutputSize(0) / sizeof(float);
        std::vector<float> decoder_output(audio_slice_len);

        // Generate pronoun slices for better effect
        auto word2pronoun = calc_word2pronoun(word2ph, pronoun_lens);
        auto dec_slices = generate_slices(word2pronoun, dec_len);

        // int dec_slice_num = int(std::ceil(zp_shape[2] * 1.0 / dec_len));
        size_t dec_slice_num = dec_slices.first.size();
        
        // Iteratively run decoder
        start = get_current_time();

        for (size_t i = 0; i < dec_slice_num; i++) {
            const Slice& ps = dec_slices.first[i];
            const Slice& zs = dec_slices.second[i];

            std::vector<float> zp_slice(zp_size, 0);
            int actual_size = std::min(zs.end - zs.start, dec_len);
            for (int n = 0; n < zp_shape[1]; n++) {
                memcpy(zp_slice.data() + n * dec_len, zp_data + n * zp_shape[2] + zs.start, sizeof(float) * actual_size);
            }

            // 输出音频的长度
            int sub_audio_len = 512 * actual_size;

            decoder_model.SetInput(zp_slice.data(), 0);
            decoder_model.SetInput(g.data(), 1);
            if (0 != decoder_model.RunSync()) {
                printf("Run decoder model failed!\n");
                return -1;
            }
            decoder_model.GetOutput(decoder_output.data(), 0);

            // 处理overlap
            int audio_start = 0;
            if (!wavlist.empty())
                if (dec_slices.first[i - 1].end > ps.start)
                    // 去掉第一个字
                    audio_start = 512 * word2pronoun[ps.start];

            int audio_end = sub_audio_len;
            if (i < dec_slices.first.size() - 1)
                if (ps.end > dec_slices.first[i + 1].start)
                    // 去掉最后一个字
                    audio_end = sub_audio_len - 512 * word2pronoun[ps.end - 1];

            wavlist.insert(wavlist.end(), decoder_output.begin() + audio_start, decoder_output.begin() + audio_end);
        }

        // for (int i = 0; i < dec_slice_num; i++) {
        //     std::vector<float> zp(zp_size, 0);
        //     int actual_size = (i + 1) * dec_len < zp_shape[2] ? dec_len : zp_shape[2] - i * dec_len;
        //     for (int n = 0; n < zp_shape[1]; n++) {
        //         memcpy(zp.data() + n * dec_len, zp_data + n * zp_shape[2] + i * dec_len, sizeof(float) * actual_size);
        //     }

        //     decoder_model.SetInput(zp.data(), 0);
        //     decoder_model.SetInput(g.data(), 1);
        //     if (0 != decoder_model.RunSync()) {
        //         printf("Run decoder model failed!\n");
        //         return -1;
        //     }
        //     decoder_model.GetOutput(decoder_output.data(), 0);
            
        //     actual_size = (i + 1) * audio_slice_len < audio_len ? audio_slice_len : audio_len - i * audio_slice_len;
        //     wavlist.insert(wavlist.end(), decoder_output.begin(), decoder_output.begin() + actual_size);
        // }

        end = get_current_time();
        printf("Decoder run %d times take %.2f ms\n", (end - start), dec_slice_num);	
    }

    
    AudioFile<float> audio_file;
    std::vector<std::vector<float> > audio_samples{wavlist};
    audio_file.setAudioBuffer(audio_samples);
    audio_file.setSampleRate(sample_rate);
    if (!audio_file.save(wav_file)) {
        printf("Save audio file failed!\n");
        return -1;
    }

    printf("Saved audio to %s\n", wav_file.c_str());

    return 0;
}
