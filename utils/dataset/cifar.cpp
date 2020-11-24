//
// Created by 田地 on 2020/11/9.
//

#include "cifar.h"
#include <fstream>
#include <iostream>

void cifar::CifarLoader::load(string fpath, bool isTest, uint64_t size) {
    std::ifstream is(fpath, std::ifstream::binary);
    if(is.fail()) {
        throw std::invalid_argument("open file failed: " + fpath);
    }
    try {
        for(int i=0; i<size; i++) {
            ten::Tensor data({1, 3, 32, 32}, ten::f32);
            char label;

            is.get(label);
            for(int j=0; j<3072; j++) {
                 ((float*)data.MutableData().data())[j] = (float)is.get();
            }

            if(isTest) {
                testData.push_back(data);
                testLabel.push_back(label);
            } else {
                trainData.push_back(data);
                trainLabel.push_back(label);
            }
        }
        is.close();
    } catch (std::ifstream::failure& e) {
        std::cerr << "Exception opening/reading/closing file\n" <<e.what() <<endl;
    }
}

void cifar::CifarLoader::Load() {
    for(int i=1; i<6; i++) {
        load(path+"/data_batch_" + to_string(i) + ".bin", false, 10000);
    }
    load(path+"/test_batch.bin", true, 10000);
}

void cifar::CifarLoader::LoadTest(uint64_t size=10000) {
    load(path+"/test_batch.bin", true, size);
}