//
// Created by 田地 on 2020/10/26.
//

#include "loader.h"
#include "context.h"
#include "engine.h"
#include "engine_test.h"
#include "tensor.h"
#include "test.h"
#include <fstream>

void engineTest::TestExecGemmNode() {
    ifstream in("/Users/tiandi03/road-to-dl/d2l/server/test/models/gemm.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);

    vector<float> data = {2}; vector<int64_t> dims = {1, 1};
    ten::Tensor src(dims, ten::f32, (char*)data.data(), 4);
    ctx::InputContext inCtx({{"input", src}});
    ctx::OutputContext ouCtx({{"10", ten::Tensor({1}, ten::f32)}});

    mklEngine.Execute(inCtx,ouCtx, g);

    auto got = ouCtx.Outputs().find("10");
    AssertFalse(got==ouCtx.Outputs().end(), "TestExecGemmNode, case1, not exist");
    AssertEqual(*((float*)got->second.Data().data()), float(3), "TestExecGemmNode, case1, wrong output");
}

void engineTest::TestBNNode()  {
    ifstream in("/Users/tiandi03/road-to-dl/d2l/server/test/models/bn.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    g.Fuse();
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);

    vector<float> data = {
            0,0,0,
            0,0,0,
            0,0,0,
            0,0,0,
            0,0,0,
            0,0,0,
            0,0,0,
            0,0,0,
            0,-1,-1,
    };
    vector<int64_t> dims = {1, 3, 3, 3};
    ten::Tensor src(dims, ten::f32, (char*)data.data(), 108);

    ctx::InputContext inCtx({{"input.1", src}});
    ctx::OutputContext ouCtx({{"9", ten::Tensor({1, 3, 3, 3}, ten::f32)}});

    mklEngine.Execute(inCtx,ouCtx, g);

    auto got = ouCtx.Outputs().find("9");
    AssertFalse(got==ouCtx.Outputs().end(), "TestBNNode, case1, not exist");
    for(int i=0; i<27; i++) {
        float out = *((float*)got->second.Data().data() + i);
        AssertEqual(out, float(0), "TestExecGemmNode, case1, wrong output");
    }
}

void engineTest::TestAddNode()  {
    ifstream in("/Users/tiandi03/road-to-dl/d2l/server/test/models/add.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    g.Fuse();
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);

    vector<float> data(27, 1);
    vector<int64_t> dims = {1, 3, 3, 3};
    ten::Tensor src(dims, ten::f32, (char*)data.data(), 108);

    ctx::InputContext inCtx({{"0", src}});
    ctx::OutputContext ouCtx({{"8", ten::Tensor({1, 3, 3, 3}, ten::f32)}});

    mklEngine.Execute(inCtx,ouCtx, g);

    auto got = ouCtx.Outputs().find("8");
    AssertFalse(got==ouCtx.Outputs().end(), "TestAddNode, case1, not exist");
    for(int i=0; i<27; i++) {
        float out = *((float*)got->second.Data().data() + i);
        AssertEqual(out, float(2), "TestAddNode, case1, wrong output");
    }
}

void engineTest::TestGlobalAvgPoolingNode2D() {
    ifstream in("/Users/tiandi03/road-to-dl/d2l/server/test/models/gavgpool.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    g.Fuse();
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);

    vector<float> data(27, 1);
    vector<int64_t> dims = {1, 3, 3, 3};
    ten::Tensor src(dims, ten::f32, (char*)data.data(), 108);

    ctx::InputContext inCtx({{"x", src}});
    ctx::OutputContext ouCtx({{"8", ten::Tensor({1, 3, 1, 1}, ten::f32)}});

    mklEngine.Execute(inCtx,ouCtx, g);

    auto got = ouCtx.Outputs().find("8");
    AssertFalse(got==ouCtx.Outputs().end(), "TestGlobalAvgPoolingNode2D, case1, not exist");
    for(int i=0; i<3; i++) {
        float out = *((float*)got->second.Data().data() + i);
        AssertEqual(out, float(1), "TestGlobalAvgPoolingNode2D, case1, wrong output");
    }
}

void engineTest::TestConvBNNode() {
    ifstream in("/Users/tiandi03/road-to-dl/d2l/server/test/models/conv_bn.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    g.Fuse();
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);

    vector<float> data(27, 1);
    vector<int64_t> dims = {1, 3, 3, 3};
    ten::Tensor src(dims, ten::f32, (char*)data.data(), 108);

    ctx::InputContext inCtx({{"input.1", src}});
    ctx::OutputContext ouCtx({{"12", ten::Tensor({1, 3, 3, 3}, ten::f32)}});

    mklEngine.Execute(inCtx,ouCtx, g);

    auto got = ouCtx.Outputs().find("12");
    AssertFalse(got==ouCtx.Outputs().end(), "TestConvBNNode, case1, not exist");

    float expected[27] = {
            -0, 0.253054, 0.356329,
            -0, 0.199778, 0.208745,
            -0, 0.117498, 0.189091,
            0.266686, -0, -0,
            -0, -0, -0,
            -0, -0, -0,
            0.240964, 0.141764, 0.244372,
            -0, -0, -0,
            -0, -0, -0,
    };

    for(int i=0; i<27; i++) {
        float out = *((float*)got->second.Data().data() + i);
        cout<< out <<endl;
//        AssertEqual(out, expected[i], "TestConvBNNode, case1, wrong output");
    }
}