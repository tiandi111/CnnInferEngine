//
// Created by 田地 on 2020/10/26.
//

#include "tensor_test.h"
#include "tensor.h"
#include "test.h"

void tensorTest::TestTensor() {
    auto ten = ten::Tensor({1, 2, 3}, ten::f32);
    AssertEqual(int(ten.Data().size()), 24, "TestTensor, case 1");

    vector<float> x = {1, 2 ,3};
    auto ten1 = ten::Tensor({1, 3}, ten::f32, (char*)x.data(), 12);
    AssertEqual(int(ten1.Data().size()), 12, "TestTensor, case 2");
}

void tensorTest::TestWrite() {
    auto ten = ten::Tensor({1, 2, 3}, ten::f32);
    vector<float> x = {1, 1, 1, 1, 1, 1};
    ten.Write(x.data());
    vector<float> y((float*)ten.Data().data(), ((float*)ten.Data().data())+6);
    AssertEqual(6, int(y.size()), "TestWrite, case 1");
    for(float e : y) {
        AssertEqual(e, float(1), "TestWrite, case 1");
    }
}

void tensorTest::TestGlobalNormalization() {
    auto ten = ten::Tensor({1, 10}, ten::f32);
    vector<float> x(10, 1);
    ten.Write(x.data());
    ten.GlobalNormalization();
    for(int i=0; i<10; i++) {
        assert(*((float*)ten.Data().data()[i]) == 0);
    }
}
