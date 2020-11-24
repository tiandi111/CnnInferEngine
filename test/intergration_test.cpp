//
// Created by 田地 on 2020/10/27.
//

#include "intergration_test.h"
#include "loader.h"
#include "context.h"
#include "engine.h"
#include "engine_test.h"
#include "tensor.h"
#include "test.h"
#include <fstream>

void interTest::TestInference() {
    ifstream in("/Users/tiandi03/road-to-dl/d2l/server/test/models/resnet10.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    g.Fuse();
//    for(auto node : g.GetNodes()) {
//        cout<< node->Type() <<endl;
//    }
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);

    vector<float> data(3072, 1);
    vector<int64_t> dims = {1, 3, 32, 32};
    ten::Tensor src(dims, ten::f32, (char*)data.data(), 12288);
    ctx::InputContext inCtx({{"input.1", src}});
    string outputName = "140";
    ctx::OutputContext ouCtx({{outputName, ten::Tensor({1, 10}, ten::f32)}});

    mklEngine.Execute(inCtx,ouCtx, g);

    auto got = ouCtx.Outputs().find(outputName);
    AssertFalse(got==ouCtx.Outputs().end(), "TestInference, case1, not exist");
    for(int i=0; i<10; i++) {
        float out = *((float*)got->second.Data().data() + i);
        cout<< out <<endl;
// expected output:
//    -0.176996
//    0.13497
//    -0.261156
//    0.019311
//    0.069262
//    0.0610936
//    0.33649
//    0.300155
//    -0.0916125
//    0.0081279
    }
}