//
// Created by 田地 on 2020/9/25.
//

#ifndef SERVER_CONTEXT_H
#define SERVER_CONTEXT_H

#include <unordered_map>
#include <string>
#include "tensor.h"

using namespace std;

namespace ctx {
    class InputContext {
    private:
        unordered_map<string, ten::Tensor> inputs;
    public:
        ~InputContext() = default;
        InputContext(unordered_map<string, ten::Tensor> inputs) : inputs(inputs) {};
        unordered_map<string, ten::Tensor>& Inputs() {return inputs;}
    };

    class OutputContext {
    private:
        unordered_map<string, ten::Tensor> outputs;
    public:
        ~OutputContext() = default;
        OutputContext(unordered_map<string, ten::Tensor> outputs) : outputs(outputs) {};
        unordered_map<string, ten::Tensor>& Outputs() {return outputs;}
    };
}


#endif //SERVER_CONTEXT_H
