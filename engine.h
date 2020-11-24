//
// Created by 田地 on 2020/9/25.
//

#ifndef SERVER_ENGINE_H
#define SERVER_ENGINE_H

#include "dnnl.hpp"
#include "graph.h"
#include "mkl.h"
#include "context.h"
#include "tensor.h"
#include <string>
#include <vector>
#include <iostream>

using namespace std;
using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

namespace eng {
    enum DeviceType {
        cpu,
        gpu
    };

    class Engine {
    private:
        string name;
        DeviceType dtype;

    public:
        Engine();
        Engine(string name, DeviceType t);
        virtual ~Engine() = default;
//        Execute(ictx::InputContext ctx, grp::Graph g);
    };


    class DnnlMemCache {
    private:
        unordered_map<string, memory> cache;
    public:
        bool Exists(const string& key) {
            return cache.find(key) != cache.end();
        }
        void Put(const string& key, const memory& mem) {
            cache.insert({key, mem});
        }
        // must call this function when InMemCache() returns true
        memory Get(const string& key) {
            return cache.find(key)->second;
        }
    };

    class MKLEngine : Engine {
    private:
        std::unordered_map<string, dnnl::primitive> prims;
        dnnl::engine eng;
        DnnlMemCache memCache;
    public:
        MKLEngine(string name, DeviceType t);
        virtual ~MKLEngine() = default;
        void Execute(ctx::InputContext& ictx, ctx::OutputContext& octx, grp::Graph& g);
        DnnlMemCache& GetDnnlMemCache() {return memCache;}
    };

    // todo: handle shape, gather and others that may not need or not support by mkl prims
    //     define a executable node to proxy mkl primitive and native node operation
    //     before shape, we need original shape, not the shape converted by mkl
    //     before gather, we need to reorder mkl prim shape to the original one
    //     => we need the control of layout transformation, for joint optimization of graph-level and op-level
    //          make it a node!
    //          at this stage, don't do any layout trans, the first goal is to run the lenet!

    class ExecutableNode {
    private:
    public:
        virtual ~ExecutableNode() = default;
        virtual void Execute() = 0;
        static dnnl::memory FindInputMemUtil (
                unordered_map<string, dnnl::memory>& inputs,
                const unordered_map<string, ten::Tensor>& weights,
                const dnnl::engine& eng,
                const string& key) {
            auto got = inputs.find(key);
            if( got != inputs.end() ) {
                return got->second;
            }
            // if not in inputs and memCache, then must in weight, otherwise will throw exception
            else {
                auto got = weights.find(key);
                if(got == weights.end()) {
                    throw std::runtime_error("weight " + key + " not found");
                }
                auto& tensor = got->second;
                return dnnl::memory(
                        {tensor.Dims(),
                         mkl::TensorDtypeToMKLType(tensor.Type()),
                         mkl::ChosseDefaultTag(tensor.Dims().size())}, eng, (void*)tensor.Data().data());
            }
        }
        static dnnl::memory FindInputMemUtilWithCache (
                unordered_map<string, dnnl::memory>& inputs,
                DnnlMemCache& memCache,
                const unordered_map<string, ten::Tensor>& weights,
                const dnnl::engine& eng,
                const string& key) {
            if(memCache.Exists(key)) {
                return memCache.Get(key);
            }
            return FindInputMemUtil(inputs, weights, eng, key);
        }
    };

    // todo: maybe we should let all these nodes to init by themselves, think about it
    // nodes that executed natively
    class ExecShapeNode : public ExecutableNode {
    private:
        dnnl::memory data;
        dnnl::memory dst;
    public:
        ExecShapeNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::ShapeNode> node,
                dnnl::engine eng);
        inline void Execute() override {
            mkl::WriteToDnnlMemory(data.get_desc().dims().data(), dst);
        }
    };

    class ExecGatherNode : public ExecutableNode {
    private:
        int axis;
        dnnl::memory data;
        dnnl::memory indices;
        dnnl::memory dst;
    public:
        ExecGatherNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::GatherNode> node,
                dnnl::engine eng);
        void Execute() override ;
    };

    class ExecMulNode : public ExecutableNode {
    private:
        dnnl::memory a;
        dnnl::memory b;
        dnnl::memory dst;
    public:
        ExecMulNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::MulNode> node,
                dnnl::engine eng);
        void Execute() override ;
    };

    class ExecUnsqueezeNode : public ExecutableNode {
    private:
        dnnl::memory dst;
    public:
        // will create a new dnnl memomry object with same data handle, different shape
        ExecUnsqueezeNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::UnsqueezeNode> node,
                dnnl::engine eng);
        // no execution needed
        inline void Execute() override {};
    };

    class ExecReshapeNode : public ExecutableNode {
    private:
        dnnl::memory dst;
    public:
        // will create a new dnnl memomry object with same data handle, different shape
        ExecReshapeNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::ReshapeNode> node,
                dnnl::engine eng);
        // no execution needed
        inline void Execute() override {};
    };

    class ExecFlattenNode : public ExecutableNode {
    private:
        dnnl::memory dst;
    public:
        // will create a new dnnl memomry object with same data handle, different shape
        ExecFlattenNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::FlattenNode> node,
                dnnl::engine eng,
                dnnl::stream stream);
        // no execution needed
        inline void Execute() override {};
    };

    class ExecPadNode : public ExecutableNode {
    private:
        dnnl::memory dst;
    public:
        ExecPadNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::PadNode> node,
                dnnl::engine eng);
        inline void Execute() override {};
    };

    class ExecConstantNode : public ExecutableNode {
    private:
        dnnl::memory dst;
    public:
        ExecConstantNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::ConstantNode> node,
                dnnl::engine eng) {};
        inline void Execute() override {};
    };

    // a class of node that executes with the help of mkl
    // todo: this does not look good, we should leave the initialization for nodes themselves
    class ExecNodeMKL : public ExecutableNode {
    protected:
        dnnl::stream stream;
        dnnl::primitive prim;
        unordered_map<int, dnnl::memory> args;
    public:
        ExecNodeMKL(dnnl::stream stream) : stream(stream) {};
        ExecNodeMKL(dnnl::stream stream,
                    dnnl::primitive& prim,
                    unordered_map<int,dnnl::memory>& args) :
                    stream(stream), prim(prim), args(args) {};
        ~ExecNodeMKL() override = default;
        inline void Execute() override {
            prim.execute(stream, args);
        }
    };

    class ExecConvNode : public ExecNodeMKL {
    public:
        ExecConvNode(
                unordered_map<string, dnnl::memory>& inputs,
                DnnlMemCache& memCache,
                grp::Graph& g,
                shared_ptr<node::ConvNode> node,
                dnnl::engine& eng,
                dnnl::stream& stream);
    };

    class ExecBNNode : public ExecNodeMKL {
    public:
        ExecBNNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::BatchNormNode> node,
                dnnl::engine eng,
                dnnl::stream stream);
    };

    class ExecConcatNode : public ExecNodeMKL {
    public:
        ExecConcatNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::ConcatNode> node,
                dnnl::engine eng,
                dnnl::stream stream);
    };

    class ExecGemmNode : public ExecNodeMKL {
    public:
        ExecGemmNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::GemmNode> node,
                dnnl::engine eng,
                dnnl::stream stream);
    };

    class ExecAvgPoolingNode2D : public ExecNodeMKL {
    public:
        ExecAvgPoolingNode2D(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::AvgPoolingNode> node,
                dnnl::engine eng,
                dnnl::stream stream);
    };

    class ExecGlobalAvgPoolingNode2D : public ExecNodeMKL {
    public:
        ExecGlobalAvgPoolingNode2D(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::GlobalAvgPoolingNode> node,
                dnnl::engine eng,
                dnnl::stream stream);
    };

    class ExecAddNode : public ExecNodeMKL {
    public:
        ExecAddNode(
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                shared_ptr<node::AddNode> node,
                dnnl::engine eng,
                dnnl::stream stream);
    };


    class MKLExecutionContext {
    private:
        dnnl::stream& stream;
        dnnl::engine& eng;
        unordered_map<string, dnnl::memory>& inputs;
        grp::Graph& g;
        MKLEngine& mklEng;
    public:
        MKLExecutionContext(
                dnnl::stream& stream,
                dnnl::engine& eng,
                unordered_map<string, dnnl::memory>& inputs,
                grp::Graph& g,
                MKLEngine& mklEng) : inputs(inputs), g(g), eng(eng), stream(stream), mklEng(mklEng) {}
        void Execute();
    private:
        void InitNode(const std::shared_ptr<node::Node>& n);
    };
}

#endif //SERVER_ENGINE_H
