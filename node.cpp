//
// Created by 田地 on 2020/9/22.
//

#include "node.h"
#include "graph.h"
#include <vector>
#include "math.h"
#include <iostream>

using namespace std;

bool node::Node::Absorb(Node another) {
    throw std::runtime_error("node type " + to_string(this->Type()) + " does not support fuse");
    return false;
}

node::ConvNode::ConvNode(
        node::OpType t,
        int id,
        const vector<string> inputs,
        const vector<string> outputs,
        grp::Graph& g,
        int group,
        vector<int> dilations,
        vector<int> kernelShape,
        vector<int> pads,
        vector<int> strides,
        vector<int> weightDims,
        vector<int> biasDims,
        string srcName,
        string weightName,
        string biasName) : Node(t, id, inputs, outputs, g) {
    this->group = group;
    this->dilations = dilations;
    this->kernelShape = kernelShape;
    this->pads = pads;
    this->strides = strides;
    this->weightDims = weightDims;
    this->biasDims = biasDims;
    this->srcInputName = srcName;
    this->weightName = weightName;
    this->biasName = biasName;
}

const vector<int>& node::ConvNode::KernelShape() const {
    return this->kernelShape;
}
const vector<int>& node::ConvNode::Pads() const {
    return this->pads;
}
const vector<int>& node::ConvNode::Strides() const {
    return this->strides;
}
const vector<int>& node::ConvNode::WeightDims() const {
    return this->weightDims;
}
const vector<int>& node::ConvNode::BiasDims() const {
    return this->biasDims;
}
const string node::ConvNode::SrcInputName() const {
    return this->srcInputName;
}
const string node::ConvNode::WeightName() const {
    return this->weightName;
}
const string node::ConvNode::BiasName() const {
    return this->biasName;
}
const string node::ConvNode::OutputName() const {
    return this->Outputs()[0];
}
bool node::ConvNode::Absorb(std::shared_ptr<Node> another) {
    // todo: different data type
    if( !relu && another->Type() == node::bn) {
        std::shared_ptr<node::BatchNormNode> bnNode = std::dynamic_pointer_cast<node::BatchNormNode>(another);
        auto & weights = GetGraph().GetMutableWeights();
        auto & convWeight = weights[weightName];
        auto & convBias = weights[biasName];
        auto & mean = weights[bnNode->MeanName()];
        auto & var = weights[bnNode->VarName()];
        auto & bnWeight = weights[bnNode->WightName()];
        auto & bnBias = weights[bnNode->BiasName()];

        size_t OC = var.Data().size() / sizeof(float);
        float * wei  = (float *) bnWeight.Data().data();
        float * bias  = (float *) bnBias.Data().data();
        float * bvar = (float *) var.Data().data();
        float * bmean = (float *) mean.Data().data();
        float scalar[OC];
        float shifter[OC];
        for(int i=0; i<OC; i++) {
            scalar[i] = wei[i] / sqrt(bvar[i]);
            shifter[i] = bias[i] - scalar[i] * bmean[i];
        }

        // scale weight
        size_t perOCSize = (convWeight.Data().size() / sizeof(float)) / OC;
        float * cw = (float *) convWeight.Data().data();
        for(size_t i=0; i<OC; i++) {
            size_t from = i * perOCSize;
            size_t to = from + perOCSize;
            for(; from<to; from++) {
                cw[from] *= scalar[i];
            }
        }

        // scale and shift bias
        float * cb = (float *) convBias.Data().data();
        for(size_t i=0; i<OC; i++) {
            cb[i] = scalar[i] * cb[i] + shifter[i];
        }

        vector<string> newOutputs = {another->Outputs()[0]};
        setOutputs(newOutputs);
        return true;
    }
    if(another->Type() == node::relu) {
        EnablePostRelu();
        setOutputs(another->Outputs());
        return true;
    }
    if(!PostSum() && (another->Type() == node::add)) {
        for(const auto& in : another->Inputs()) {
            if(in != OutputName()) {
                auto anotherOperandNode = GetGraph().GetNodeByOutputName(in);
                if(!anotherOperandNode) throw runtime_error("add node only has 1 operand");
                // Only the toplogically later node can absorb 'ADD' node
                if(ID() <= anotherOperandNode->ID()) {
                    return false;
                } else {
                    auto allSuccNode = GetGraph().GetDependRelations().find(in);
                    if(allSuccNode == GetGraph().GetDependRelations().end()) throw runtime_error("graph info is corrupted");
                    for(auto& succNode : allSuccNode->second) {
                        // if any successor node of the anotherOperandNode is toplogically later than this node
                        // we cannot absorb 'ADD' node
                        if((succNode->ID() != another->ID()) && (succNode->ID() > ID())) {
                            return false;
                        }
                    }
                    EnablePostSum(in);
                    setOutputs(another->Outputs());
                    return true;
                }
            }
        }
    }
    return false;
}
void node::ConvNode::EnablePostRelu() {
    this->relu = true;
}

node::BatchNormNode::BatchNormNode(
        node::OpType t,
        int id,
        const vector<string> inputs,
        const vector<string> outputs,
        grp::Graph& g,
        float epsilon,
        float momentum,
        vector<int> dim,
        string wightName,
        string biasName,
        string meanName,
        string varName,
        string srcInputName) :
        Node(t, id, inputs, outputs, g),
        epsilon(epsilon),
        momentum(momentum),
        dim(dim),
        wightName(wightName),
        biasName(biasName),
        meanName(meanName),
        varName(varName),
        srcInputName(srcInputName) {}

float node::BatchNormNode::Epsilon() const {
    return this->epsilon;
}
float node::BatchNormNode::Momentum() const {
    return this->momentum;
}
const vector<int>& node::BatchNormNode::Dim() const {
    return this->dim;
}
string node::BatchNormNode::WightName() const {
    return this->wightName;
}
string node::BatchNormNode::BiasName() const {
    return this->biasName;
}
string node::BatchNormNode::MeanName() const {
    return this->meanName;
}
string node::BatchNormNode::VarName() const {
    return this->varName;
}
string node::BatchNormNode::SrcInputName() const {
    return this->srcInputName;
}
string node::BatchNormNode::OutputName() const {
    return this->Outputs()[0];
}
bool node::BatchNormNode::Absorb(std::shared_ptr<Node> another) {
    // remember, if this has post-relu, it cannot absorb another bn or conv
    // also, note the difference of bn absorb conv and conv absorb bn
    if(another->Type() == node::relu) {
        EnablePostRelu();
        setOutputs(vector<string>(another->Outputs()));
        return true;
    }
    return false;
}