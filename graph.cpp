//
// Created by 田地 on 2020/9/22.
//

#include "graph.h"
#include "node.h"
#include <iostream>
using namespace grp;

void Graph::AddNode(std::shared_ptr<node::Node> node) {
    nodes.push_back(node);
    for(const auto& in : node->Inputs()) {
        inputDepends[in].push_back(node);
    }
    for(const auto& out : node->Outputs()) {
        outputSrcNode[out] = node;
    }
}

const vector<std::shared_ptr<node::Node>>& Graph::GetNodes() const {
    return this->nodes;
}

const unordered_map<string, ten::Tensor>& Graph::GetWeights() const {
    return this->weights;
}

unordered_map<string, ten::Tensor>& Graph::GetMutableWeights() {
    return weights;
}

const void * Graph::GetWeightHandle(string name) const {
    auto got = weights.find(name);
    if (got == weights.end()) {
        return nullptr;
    }
    return static_cast<const void * >(got->second.Data().data());
}

const ten::Tensor& Graph::GetWeightTensor(string name) const {
    auto got = weights.find(name);
    if(got == weights.end()) {
        throw std::runtime_error("weight " + name + " not found");
    }
    return got->second;
}

void Graph::Fuse() {
    for(auto it = nodes.begin(); it<nodes.end(); it++) {
        if(it+1 == nodes.end()) {
            return;
        }
        auto node = *it;
        if(node->Type() == node::OpType::conv || node->Type() == node::OpType::bn) {
            string outputName = node->Outputs()[0];
            // a node with multiple out edges can not absorb any of its child
            // (except all children are exactly the same, but we ignore this situation for now)
            auto got = inputDepends.find(outputName);
            if(got == inputDepends.end() || got->second.size()>1) {
                continue;
            }
            auto succNode = got->second[0];
            // if succNode is absorbed
            if((node->Type() == node::OpType::conv && std::dynamic_pointer_cast<node::ConvNode>(node)->Absorb(succNode)) ||
            (node->Type() == node::OpType::bn && std::dynamic_pointer_cast<node::BatchNormNode>(node)->Absorb(succNode))) {
                inputDepends.erase(outputName);
                for(auto it1=it; it1<nodes.end(); it1++) {
                    if((*it1)->ID() == succNode->ID()) {
                        nodes.erase(it1);
                        break;
                    }
                }
                it--;
            }
        }
    }
}

