//
// Created by 田地 on 2020/11/11.
//

#include "benchmark.h"
#include "loader.h"
#include "engine.h"
#include "utils/dataset/cifar.h"
#include <fstream>
#include <ctime>
#include <torch/script.h>

using namespace std;

void bench::CifarResNetInfer() {
    float N = 1;
    // load dataset
    cifar::CifarLoader cf( "/Users/tiandi03/Desktop/dataset/cifar-10-batches-bin");
    cf.LoadTest(N);
    // load model
    ifstream in("../resource/model/res_cifar_1606001241.onnx", ios_base::binary);
    auto g = load::LoadOnnx(&in);
    g.Fuse();
    // init engine
    eng::MKLEngine mklEngine("cpu", eng::DeviceType::cpu);
    // inference
    auto testData = cf.GetTestData();
    auto testLabel = cf.GetTestLabel();
    float correct = 0;
    string outputName = "516";
    clock_t start = clock();
    for(int i=0; i<N; i++) {
        clock_t iter_st = clock();
        // globally normalizes the image
        testData[i].GlobalNormalization();
        // prepare input and output context
        ctx::InputContext inCtx({{"input.1", testData[i]}});
        ctx::OutputContext ouCtx({{outputName, ten::Tensor({1, 10}, ten::f32)}});
        // predict
        mklEngine.Execute(inCtx,ouCtx, g);
        // get result
        auto got = ouCtx.Outputs().find(outputName);
        if(got == ouCtx.Outputs().end()) {
            throw runtime_error("output not found");
        }
        uint64_t pred = got->second.ArgMax1D();
        uint64_t label = testLabel[i];
        cout<< "pred: " << pred << ", label: " << label <<endl;
        // compare the result
        correct += pred == label? 1 : 0;
        std::cout<< (clock()-iter_st) * 1.0 / CLOCKS_PER_SEC * 1000 << " milliseconds for this iter\n";
    }
    std::cout<< (clock()-start) * 1.0 / CLOCKS_PER_SEC * 1000 << " milliseconds in total\n";
    cout<< "score: " << correct / N <<endl;
}

void bench::TorchCifarResNetInfer() {
    float N = 1000;
    // load dataset
    cifar::CifarLoader cf( "/Users/tiandi03/Desktop/dataset/cifar-10-batches-bin");
    cf.LoadTest(N);
    // load model
    torch::jit::script::Module module;
    module = torch::jit::load("/Users/tiandi03/road-to-dl/Project636/res_cifar_1606044772.jit");
    module.eval();
    //
    auto data = cf.GetTestData();
    auto testLabel = cf.GetTestLabel();
    uint64_t correct = 0;
    clock_t start = clock();

    for(int i=0; i<N; i++) {
        clock_t iter_st = clock();
        data[i].GlobalNormalization();
        std::vector<torch::jit::IValue> input({torch::from_blob(data[i].GetDataHandle(),
                                                                {1, 3, 32, 32},
                                                                at::TensorOptions().dtype(at::kFloat))});
        at::Tensor output = module.forward(input).toTensor();
        uint64_t pred = output.argmax().item<int>();
        uint64_t label = testLabel[i];
        cout<< "pred: " << pred << ", label: " << label <<endl;
        correct += pred == label? 1 : 0;
        std::cout<< (clock()-iter_st) * 1.0 / CLOCKS_PER_SEC * 1000 << " milliseconds for this iter\n";
    }
    std::cout<< (clock()-start) * 1.0 / CLOCKS_PER_SEC * 1000 << " milliseconds in total\n";
    cout<< "score: " << correct / N <<endl;
}
