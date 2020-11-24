//
// Created by 田地 on 2020/10/24.
//

#include "util_test.h"
#include "test.h"
#include <vector>
#include <iostream>

void utilTest::TestGather() {
    vector<float> src = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int64_t> srcDims = {3,3};
    vector<int64_t> indices = {0, 1, 2, 2};
    vector<int64_t> indicesDims = {2, 2};
    float dst[12];
    vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 8, 9};
    utils::gather(src.data(), dst, indices.data(), srcDims, indicesDims, 0);
    AssertEqual(vector<float>(dst, dst+12), expected, "TestGather, case1");

    vector<int> src1 = {1, 2, 3, 4};
    vector<int64_t> srcDims1 = {4};
    vector<int64_t> indices1 = {2};
    vector<int64_t> indicesDims1 = {1};
    int dst1[1];
    vector<int> expected1 = {3};
    utils::gather(src1.data(), dst1, indices1.data(), srcDims1, indicesDims1, 0);
    AssertEqual(vector<int>(dst1, dst1+1), expected1, "TestGather, case2");
}

void utilTest::TestGatherDims() {
    int axis = 0;
    vector<int64_t> srcDims = {3,3};
    vector<int64_t> indicesDims = {2, 2};
    vector<int64_t> expected = {2, 2, 3};
    auto gDims = utils::gatherDims(srcDims, indicesDims, axis);
    AssertEqual(gDims, expected, "TestGatherDims, case1");
}

void utilTest::TestMul1D() {
    int a[3] = {1, 1, 1};
    int b[3] = {1, 2, 3};
    int adim[1] = {3};
    int bdim[1] = {3};
    int dst[3];
    vector<int> expected = {1, 2, 3};
    utils::mul1d(a, b, dst, adim, bdim);
    AssertEqual(vector<int>(dst, dst+3), expected, "TestMul1D, case1");

    int a1[3] = {1, 1, 1};
    int b1[1] = {2};
    int adim1[1] = {3};
    int bdim1[1] = {1};
    int dst1[3];
    vector<int> expected1 = {2, 2, 2};
    utils::mul1d(a1, b1, dst1, adim1, bdim1);
    AssertEqual(vector<int>(dst1, dst1+3), expected1, "TestMul1D, case2");

    int a2[1] = {1};
    int b2[1] = {2};
    int adim2[1] = {1};
    int bdim2[1] = {1};
    int dst2[1];
    vector<int> expected2 = {2};
    utils::mul1d(a2, b2, dst2, adim2, bdim2);
    AssertEqual(vector<int>(dst2, dst2+3), expected2, "TestMul1D, case3");
}


void utilTest::TestMul2D() {
    int a[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int b[3] = {1, 2, 3};
    int adim[2] = {3, 3};
    int bdim[2] = {3, 1};
    int dst[9];
    vector<int> expected = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    utils::mul2d(a, b, dst, adim, bdim);
    AssertEqual(vector<int>(dst, dst+9), expected, "TestMul2D, case1");

    int a1[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int b1[3] = {1, 2, 3};
    int adim1[2] = {3, 3};
    int bdim1[2] = {1, 3};
    int dst1[9];
    vector<int> expected1 = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    utils::mul2d(a1, b1, dst1, adim1, bdim1);
    AssertEqual(vector<int>(dst1, dst1+9), expected1, "TestMul2D, case2");

    int a2[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    int b2[3] = {2};
    int adim2[2] = {3, 3};
    int bdim2[2] = {1, 1};
    int dst2[9];
    vector<int> expected2 = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    utils::mul2d(a2, b2, dst2, adim2, bdim2);
    AssertEqual(vector<int>(dst2, dst2+9), expected2, "TestMul2D, case3");

    utils::mul2d(b2, a2, dst2, bdim2, adim2);
    AssertEqual(vector<int>(dst2, dst2+9), expected2, "TestMul2D, case4");
}

void utilTest::TestMul3D() {
    int a[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int b[4] = {1, 2, 3, 4};
    int adim[3] = {2, 2, 2};
    int bdim[3] = {2, 2, 1};
    int dst[8];
    vector<int> expected = {1, 1, 2, 2, 3, 3, 4, 4};
    utils::mul3d(a, b, dst, adim, bdim);
    AssertEqual(vector<int>(dst, dst+8), expected, "TestMul3D, case1");

    int a1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int b1[4] = {1, 2, 3, 4};
    int adim1[3] = {2, 2, 2};
    int bdim1[3] = {2, 1, 2};
    int dst1[8];
    vector<int> expected1 = {1, 2, 1, 2, 3, 4, 3, 4};
    utils::mul3d(a1, b1, dst1, adim1, bdim1);
    AssertEqual(vector<int>(dst1, dst1+8), expected1, "TestMul3D, case2");

    float a2[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    float b2[2] = {1, 2};
    int adim2[3] = {2, 2, 2};
    int bdim2[3] = {1, 1, 2};
    float dst2[8];
    vector<float> expected2 = {1, 2, 1, 2, 1, 2, 1, 2};
    utils::mul3d(a2, b2, dst2, adim2, bdim2);
    AssertEqual(vector<float>(dst2, dst2+8), expected2, "TestMul3D, case3");
}
