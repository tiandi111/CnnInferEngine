//
// Created by 田地 on 2020/11/11.
//

#include "utils_test.h"
#include "./utils/dataset/cifar.h"
#include <iostream>

void utilsTest::TestCifarLoaderLoad() {
    cifar::CifarLoader cf( "/Users/tiandi03/Desktop/dataset/cifar-10-batches-bin");
    cf.Load();
    assert(cf.GetTrainData().size() == 50000);
    assert(cf.GetTrainLabel().size() == 50000);
    assert(cf.GetTestData().size() == 10000);
    assert(cf.GetTestLabel().size() == 10000);
}