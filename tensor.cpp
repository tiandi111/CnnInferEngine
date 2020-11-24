//
// Created by 田地 on 2020/9/26.
//

#include "tensor.h"
#include <string>
#include <stdexcept>
#include <iostream>
#include "math.h"

void ten::Tensor::computeBytes() {
    uint64_t b = Len();
    switch (dtype) {
        case f32:
            bytes = sizeof(float) * b;
            return;
        case i64:
            bytes = sizeof(int64_t) * b;
            return;
        case i8:
            bytes = sizeof(int8_t) * b;
            return;
        case ui8:
            bytes = sizeof(uint8_t) * b;
            return;
        default:
            throw std::invalid_argument("unsupported data type");
    }
}

void ten::Tensor::checkBuffer() {
    if(data.size() != bytes) {
        throw std::invalid_argument("buffer size does not match dimsions");
    }
}


ten::Tensor::Tensor(vector<int64_t> dims, DataType t) : dims(dims), dtype(t) {
    computeBytes();
    data = vector<char>(bytes);
}

ten::Tensor::Tensor(vector<int64_t> dims, DataType t, vector<char>& data): dims(dims), dtype(t), data(data) {
    computeBytes();
    checkBuffer();
}

void ten::Tensor::Write(void* handle) {
    char* p = (char*) handle;
    data.assign(p, p+bytes);
}


const vector<char>& ten::Tensor::Data() const {
    return data;
}

const ten::DataType ten::Tensor::Type() const {
    return dtype;
}

const vector<int64_t>& ten::Tensor::Dims() const {
    return dims;
}

uint64_t ten::Tensor::ArgMax1D() {
    int count = 0;
    for(uint64_t dim : dims) {
        count += dim != 1? 1 : 0;
    }
    if(dims.size() != 1 && count != 1) {
        throw invalid_argument("ArgMax1D only applied to 1-D Tensor");
    }
    switch (dtype) {
        case f32: {
            float max = 0;
            uint64_t maxIdx = -1;
            float *arr = (float*)data.data();
            uint64_t size = Len();
            for(int i=0; i<size; i++) {
                if(i==0 || arr[i] > max) {
                    max = arr[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
        default:
            throw invalid_argument("only float32 is supported now");
    }
}

void ten::Tensor::GlobalNormalization() {
    uint64_t b = Len();
    switch (dtype) {
        case f32: {
            float *arr = (float *) data.data();
            float sum = 0;
            for (int i = 0; i < b; i++) {
                sum += arr[i];
            }
            float mean = sum / b;
            float varSum = 0;
            for (int i = 0; i < b; i++) {
                float a = arr[i] - mean;
                varSum += a * a;
            }
            float std = sqrt(varSum / b);
            for (int i = 0; i < b; i++) {
                arr[i] = (arr[i] - mean) / (std + 1e-8);
            }
            return;
        }
        default:
            throw invalid_argument("only float32 is supported now");
    }
}

uint64_t ten::Tensor::Len() {
    uint64_t b = 1;
    for(int64_t d : dims) {b *= d;}
    return b;
}