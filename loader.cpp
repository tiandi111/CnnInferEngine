//
// Created by 田地 on 2020/9/24.
//

#include "loader.h"
#include "engine.h"
#include "context.h"
#include <fstream>
#include <stdexcept>

// parse input dimension info
map<string, vector<int>> load::ParseInputInfos(onnx::ModelProto oModel) {
    map<string, vector<int>> inputDimInfos;
    for (int i = 0; i < oModel.graph().input_size(); i++) {
        vector<int> inDims;
        auto oInDims = oModel.graph().input(i).type().tensor_type().shape().dim();
        for (int j = 0; j < oInDims.size(); j++) {
            inDims.push_back(oInDims[j].dim_value());
        }
        inputDimInfos[oModel.graph().input(i).name()] = inDims;
    }
    return inputDimInfos;
}

// parse output dimension info
map<string, vector<int>> load::ParseOutputInfos(onnx::ModelProto oModel) {
    map<string, vector<int>> outputDimInfos;
    for(int i=0; i<oModel.graph().output_size(); i++) {
        vector<int> outDims;
        auto oOutDims = oModel.graph().output(i).type().tensor_type().shape().dim();
        for(int j=0; j<oOutDims.size(); j++) {
            outDims.push_back(oOutDims[j].dim_value());
        }
        outputDimInfos[oModel.graph().output(i).name()] = outDims;
    }
    return outputDimInfos;
}

node::OpType load::OnnxType2OpType(string t) {
    if(t == "Conv") {
        return node::conv;
    }
    if(t == "BatchNormalization") {
       return node::bn;
    }
    if(t == "Relu") {
        return node::relu;
    }
    if(t == "Shape") {
        return node::shape;
    }
    if(t == "Gather") {
        return node::gather;
    }
    if(t == "Mul") {
        return node::mul;
    }
    if(t == "Unsqueeze") {
        return node::unsqueeze;
    }
    if(t == "Concat") {
        return node::concat;
    }
    if(t == "Reshape") {
        return node::reshape;
    }
    if(t == "Gemm") {
        return node::gemm;
    }
    if(t == "Flatten") {
        return node::flatten;
    }
    if(t == "Pad") {
        return node::pad;
    }
    if(t == "AveragePool") {
        return node::avgpool;
    }
    if(t == "GlobalAveragePool") {
        return node::gavgpool;
    }
    if(t == "Add") {
        return node::add;
    }
    // constant node is not supported now!
//    if(t == "Constant") {
//        return node::constant;
//    }
    return node::unknown;
}

ten::DataType load::OnnxDataType2TenDataType(int odtype) {
    switch(odtype) {
        case onnx::TensorProto::FLOAT :
            return ten::DataType::f32;
        case onnx::TensorProto::INT64 :
            return ten::DataType::i64;
        default:
            throw "unsupported data type";
            return ten::DataType::unknown;
    }
}

std::shared_ptr<node::Node> load::ParseNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    if(oNode.op_type() == "Conv") {
        return ParseConvNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "BatchNormalization") {
        return ParseBnNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Shape") {
        return ParseShapeNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Gather") {
        return ParseGatherNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Mul") {
        return ParseMulNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Unsqueeze") {
        return ParseUnsqueezeNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Concat") {
        return ParseConcatNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Reshape") {
        return ParseReshapeNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Flatten") {
        return ParseFlattenNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Gemm") {
        return ParseGemmNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Pad") {
        return ParsePadNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "AveragePool") {
        return ParseAvgPoolingNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "GlobalAveragePool") {
        return ParseGlobalAvgPoolingNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Add") {
        return ParseAddNode(oNode, id, inputs, outputs, g);
    }
    if(oNode.op_type() == "Constant") {
        return ParseConstantNode(oNode, id, inputs, outputs, g);
    }
    node::Node node = node::Node(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g);
    return std::make_shared<node::Node>(node);
}

std::shared_ptr<node::ConvNode> load::ParseConvNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    int group;
    vector<int> dilations;
    vector<int> kernelShape;
    vector<int> pads; // l, r, u, d
    vector<int> strides;
    vector<int> weightDims;
    vector<int> biasDims;
    string srcName, weightName, biasName;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if(attr.name() == "dilations") {
            dilations.assign(attr.ints().begin(), attr.ints().end());
        }
        if(attr.name() == "kernel_shape") {
            kernelShape.assign(attr.ints().begin(), attr.ints().end());
        }
        if(attr.name() == "pads") {
            pads.assign(attr.ints().begin(), attr.ints().end());
        }
        if(attr.name() == "strides") {
            strides.assign(attr.ints().begin(), attr.ints().end());
        }
        if(attr.name() == "group") {
            group = attr.i();
        }
    }
    for(const auto & in : inputs) {
        auto name = in;
        auto got = g.GetWeights().find(name);
        if(name.rfind("weight") != string::npos) {
            if(got == g.GetWeights().end()) {
                throw "conv layer weight not found, the model may have been broken";
            }
            weightDims.assign(got->second.Dims().begin(), got->second.Dims().end());
            weightName = name;
        } else if(name.rfind("bias") != string::npos) {
            if(got == g.GetWeights().end()) {
                throw "conv layer bias not found, the model may have been broken";
            }
            biasDims.assign(got->second.Dims().begin(), got->second.Dims().end());
            biasName = name;
        } else {
            srcName = name;
        }
    }
    node::ConvNode node = node::ConvNode(
            load::OnnxType2OpType(oNode.op_type()),
            id,
            inputs,
            outputs,
            g,
            group,
            dilations,
            kernelShape,
            pads,
            strides,
            weightDims,
            biasDims,
            srcName,
            weightName,
            biasName);
    return std::make_shared<node::ConvNode>(node);
}

std::shared_ptr<node::BatchNormNode> load::ParseBnNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    float epsilon;
    float momentum; // factor used to compute running mean and standard
    vector<int> dim;
    string weightName;
    string biasName;
    string meanName;
    string varName;
    string srcInputName;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if(attr.name() == "epsilon") {
            epsilon = attr.f();
        }
        if(attr.name() == "kernel_shape") {
            momentum = attr.f();
        }
    }
    for(const auto & in : inputs) {
        if(in.rfind("weight") != string::npos) {
            weightName = in;
        } else if(in.rfind("bias") != string::npos) {
            biasName = in;
        } else if(in.rfind("running_mean") != string::npos) {
            meanName = in;
        } else if(in.rfind("running_var") != string::npos) {
            varName = in;
        } else {
            srcInputName = in;
        }
    }
    auto got = g.GetWeights().find(weightName);
    if(got == g.GetWeights().end()) {
        throw std::invalid_argument("weight " + weightName + " not found for batch norm node" + std::to_string(id));
    }
    dim.assign(got->second.Dims().begin(), got->second.Dims().end());
    node::BatchNormNode node = node::BatchNormNode(
            load::OnnxType2OpType(oNode.op_type()),
            id,
            inputs,
            outputs,
            g,
            epsilon,
            momentum,
            dim,
            weightName,
            biasName,
            meanName,
            varName,
            srcInputName);
    return std::make_shared<node::BatchNormNode>(node);
}

std::shared_ptr<node::ShapeNode> load::ParseShapeNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    return std::make_shared<node::ShapeNode>(
            node::ShapeNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g));
}

std::shared_ptr<node::GatherNode> load::ParseGatherNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    int axis;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if (attr.name() == "axis") {
            axis = attr.i();
        }
    }
    return std::make_shared<node::GatherNode>(
            node::GatherNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g, axis));
}

std::shared_ptr<node::MulNode> load::ParseMulNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    return std::make_shared<node::MulNode>(
            node::MulNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g));
}

std::shared_ptr<node::UnsqueezeNode> load::ParseUnsqueezeNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    return std::make_shared<node::UnsqueezeNode>(
            node::UnsqueezeNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g));
}

std::shared_ptr<node::ConcatNode> load::ParseConcatNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    int axis;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if (attr.name() == "axis") {
            axis = attr.i();
        }
    }
    return std::make_shared<node::ConcatNode>(
            node::ConcatNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g, axis));
}

std::shared_ptr<node::FlattenNode> load::ParseFlattenNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    int axis;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if (attr.name() == "axis") {
            axis = attr.i();
        }
    }
    return std::make_shared<node::FlattenNode>(
            node::FlattenNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g, axis));
}

std::shared_ptr<node::ReshapeNode> load::ParseReshapeNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    return std::make_shared<node::ReshapeNode>(
            node::ReshapeNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g));
}

std::shared_ptr<node::GemmNode> load::ParseGemmNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    float alpha, beta; bool transA, transB;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if (attr.name() == "alpha") {
            alpha = attr.f();
        }
        if (attr.name() == "beta") {
            beta = attr.f();
        }
        if (attr.name() == "transA") {
            transA = attr.i();
        }
        if (attr.name() == "transB") {
            transB = attr.i();
        }
    }
    return std::make_shared<node::GemmNode>(
            node::GemmNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g, alpha, beta,
                    transA, transB, true));
}

std::shared_ptr<node::PadNode> load::ParsePadNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    string mode;
    vector<int> pads;
    float value;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if (attr.name() == "mode") {
            mode = attr.s();
        }
        if (attr.name() == "pads") {
            pads.assign(attr.ints().begin(), attr.ints().end());
        }
        if (attr.name() == "value") {
            value = attr.f();
        }
    }
    return std::make_shared<node::PadNode>(
            node::PadNode(load::OnnxType2OpType(oNode.op_type()),
                          id, inputs, outputs, g, mode, pads, value));
}

std::shared_ptr<node::AvgPoolingNode> load::ParseAvgPoolingNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    vector<int> kernelShape, pads, strides;
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if (attr.name() == "kernel_shape") {
            kernelShape.assign(attr.ints().begin(), attr.ints().end());
        }
        if (attr.name() == "pads") {
            pads.assign(attr.ints().begin(), attr.ints().end());
        }
        if (attr.name() == "strides") {
            strides.assign(attr.ints().begin(), attr.ints().end());
        }
    }
    return std::make_shared<node::AvgPoolingNode>(
            node::AvgPoolingNode(load::OnnxType2OpType(oNode.op_type()),
                          id, inputs, outputs, g, kernelShape, pads, strides));
}

std::shared_ptr<node::GlobalAvgPoolingNode> load::ParseGlobalAvgPoolingNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    return std::make_shared<node::GlobalAvgPoolingNode>(
            node::GlobalAvgPoolingNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g));
}

std::shared_ptr<node::AddNode> load::ParseAddNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    return std::make_shared<node::AddNode>(
            node::AddNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g));
}

std::shared_ptr<node::ConstantNode> load::ParseConstantNode(
        onnx::NodeProto& oNode,
        int id,
        const vector<string>& inputs,
        const vector<string>& outputs,
        grp::Graph& g) {
    for(int i=0; i<oNode.attribute_size(); i++) {
        auto attr = oNode.attribute(i);
        if(attr.name() == "value") {
            auto container = attr.t();
            vector<int64_t> dims(container.dims().begin(), container.dims().end());
            vector<char> data(container.raw_data().begin(), container.raw_data().end());
            g.AddWeight(outputs[0], ten::Tensor(
                    dims,
                    load::OnnxDataType2TenDataType(container.data_type()),
                    data));
        } else {
            throw runtime_error("attribute " + attr.name() + " is not supported now");
        }
    }
    return std::make_shared<node::ConstantNode>(
            node::ConstantNode(load::OnnxType2OpType(oNode.op_type()), id, inputs, outputs, g));
}

unordered_map<string, ten::Tensor> load::ReadWeights(onnx::GraphProto oGraph) {
    unordered_map<string, ten::Tensor> weights;
    for(int i=0; i<oGraph.initializer_size(); i++) {
        auto initial = oGraph.initializer(i);
        vector<int64_t> dims(initial.dims().begin(), initial.dims().end());
        vector<char> data(initial.raw_data().begin(), initial.raw_data().end());
        weights[initial.name()] = ten::Tensor(
                dims,
                load::OnnxDataType2TenDataType(initial.data_type()),
                data);
    }
    return weights;
}


grp::Graph load::LoadOnnx(istream *is) {

    onnx::ModelProto oModel = onnx::ModelProto();
    oModel.ParseFromIstream(is);
    cout << "load onnx model successfully" << endl;

    grp::Graph graph = grp::Graph(ParseInputInfos(oModel), ParseOutputInfos(oModel), load::ReadWeights(oModel.graph()));

    // parse node info
    for (int i = 0; i < oModel.graph().node_size(); i++) {
        auto oNode = oModel.graph().node(i);

        const vector<string> inputs(oNode.input().begin(), oNode.input().end());
        const vector<string> outputs(oNode.output().begin(), oNode.output().end());
        auto node = load::ParseNode(oNode, i, inputs, outputs, graph);

        graph.AddNode(node);

    }

    return graph;
}