//
// Created by 田地 on 2020/11/9.
//

#ifndef SERVER_CIFAR_H
#define SERVER_CIFAR_H

#include <string>
#include "tensor.h"

using namespace std;
using namespace ten;

namespace cifar {
    class CifarLoader {
    private:
        string path;
        vector<Tensor> trainData;
        vector<char> trainLabel;
        vector<Tensor> testData;
        vector<char> testLabel;

        void load(string fpath, bool isTest, uint64_t size);
    public:
        CifarLoader(string path) : path(path) {}
        void Load();
        void LoadTest(uint64_t size);
        const vector<Tensor>& GetTestData() const { return testData;}
        const vector<char>& GetTrainLabel() const { return trainLabel;}
        const vector<Tensor>& GetTrainData() const { return trainData;}
        const vector<char>& GetTestLabel() const { return testLabel;}
    };
}

#endif //SERVER_CIFAR_H
