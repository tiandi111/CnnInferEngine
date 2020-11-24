//
// Created by 田地 on 2020/9/25.
//

#include "engine.h"
#include "mkl.h"
#include "utils.h"
#include <memory>
#include <stdexcept>
#include <algorithm>

eng::Engine::Engine() {}

eng::Engine::Engine(string name, DeviceType t) {
    this->name = name;
    this->dtype = t;

}

eng::MKLEngine::MKLEngine(string name, DeviceType t) : Engine(name, t){
    // todo: engine index?
    this->eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
}

void eng::MKLEngine::Execute(ctx::InputContext& ictx, ctx::OutputContext& octx, grp::Graph& g) {
    dnnl::stream stream(this->eng);
    // inputs and weights, for quick getting only
    unordered_map<string, dnnl::memory> inputs;
    // create memory object for inputs
    for(auto& it : ictx.Inputs()) {
        dnnl::memory::dims dims(it.second.Dims().begin(), it.second.Dims().end());
        auto inMemory = dnnl::memory(
                {dims, dt::f32, mkl::ChosseDefaultTag(dims.size())},
                this->eng,
                it.second.GetDataHandle()); // todo: data type and tag
        inputs.insert({it.first, inMemory});
    }
    auto execCtx = MKLExecutionContext(
            stream,
            eng,
            inputs,
            g,
            *this);
    execCtx.Execute();
    for(auto& ot : octx.Outputs()) {
        auto got = inputs.find(ot.first);
        if(got == inputs.end()) {
            throw std::runtime_error("output " + ot.first + " not found");
        }
        ot.second.Write(got->second.get_data_handle());
    }
}

void eng::MKLExecutionContext::Execute() {
    auto& nodes = g.GetNodes();
    for(auto& node : nodes) {
        InitNode(node);
    }
    this->stream.wait();
}

void eng::MKLExecutionContext::InitNode(const std::shared_ptr<node::Node>& n) {
    switch(n->Type()) {
        case node::OpType::conv : {
            std::shared_ptr<node::ConvNode> node = std::dynamic_pointer_cast<node::ConvNode>(n);
            if(node->KernelShape().size() != 2) {
                throw runtime_error(to_string(node->KernelShape().size()-2) + "d convolution is not supported now");
            }
            ExecConvNode execNode(inputs, mklEng.GetDnnlMemCache(), g, node, eng, stream);
            execNode.Execute();
            break;
        }
        case node::OpType::bn : {
            std::shared_ptr<node::BatchNormNode> node = std::dynamic_pointer_cast<node::BatchNormNode>(n);
            ExecBNNode execNode(inputs, g, node, eng, stream);
            execNode.Execute();
            break;
        }
        case node::OpType::shape : {
            std::shared_ptr<node::ShapeNode> node = std::dynamic_pointer_cast<node::ShapeNode>(n);
            ExecShapeNode execNode(inputs, g, node, eng);
            execNode.Execute();
            break;
        }
        case node::OpType::gather : {
            std::shared_ptr<node::GatherNode> node = std::dynamic_pointer_cast<node::GatherNode>(n);
            ExecGatherNode execNode(inputs, g, node, eng);
            execNode.Execute();
            break;
        }
        case node::OpType::mul : {
            std::shared_ptr<node::MulNode> node = std::dynamic_pointer_cast<node::MulNode>(n);
            ExecMulNode execNode(inputs, g, node, eng);
            execNode.Execute();
            break;
        }
        case node::OpType::unsqueeze : {
            std::shared_ptr<node::UnsqueezeNode> node = std::dynamic_pointer_cast<node::UnsqueezeNode>(n);
            ExecUnsqueezeNode execNode(inputs, g, node, eng);
            execNode.Execute();
            break;
        }
        case node::OpType::concat : {
            std::shared_ptr<node::ConcatNode> node = std::dynamic_pointer_cast<node::ConcatNode>(n);
            ExecConcatNode execNode(inputs, g, node, eng, stream);
            execNode.Execute();
            break;
        }
        case node::OpType::reshape : {
            std::shared_ptr<node::ReshapeNode> node = std::dynamic_pointer_cast<node::ReshapeNode>(n);
            ExecReshapeNode execNode(inputs, g, node, eng);
            execNode.Execute();
            break;
        }
        case node::OpType::flatten : {
            std::shared_ptr<node::FlattenNode> node = std::dynamic_pointer_cast<node::FlattenNode>(n);
            ExecFlattenNode execNode(inputs, g, node, eng, stream);
            execNode.Execute();
            break;
        }
        case node::OpType::constant : {
            std::shared_ptr<node::ConstantNode> node = std::dynamic_pointer_cast<node::ConstantNode>(n);
            ExecConstantNode execNode(inputs, g, node, eng);
            execNode.Execute();
            break;
        }
        case node::OpType::gemm : {
            std::shared_ptr<node::GemmNode> node = std::dynamic_pointer_cast<node::GemmNode>(n);
            ExecGemmNode execNode(inputs, g, node, eng, stream);
            execNode.Execute();
            break;
        }
        case node::OpType::pad : {
            std::shared_ptr<node::PadNode> node = std::dynamic_pointer_cast<node::PadNode>(n);
            ExecPadNode execNode(inputs, g, node, eng);
            execNode.Execute();
            break;
        }
        case node::OpType::avgpool : {
            std::shared_ptr<node::AvgPoolingNode> node = std::dynamic_pointer_cast<node::AvgPoolingNode>(n);
            if(node->KernelShape().size() != 4) {
                throw runtime_error(to_string(node->KernelShape().size()-2) + "d pooling is not supported now");
            }
            ExecAvgPoolingNode2D execNode(inputs, g, node, eng, stream);
            execNode.Execute();
            break;
        }
        case node::OpType::gavgpool : {
            std::shared_ptr<node::GlobalAvgPoolingNode> node = std::dynamic_pointer_cast<node::GlobalAvgPoolingNode>(n);
            ExecGlobalAvgPoolingNode2D execNode(inputs, g, node, eng, stream);
            execNode.Execute();
            break;
        }
        case node::OpType::add : {
            std::shared_ptr<node::AddNode> node = std::dynamic_pointer_cast<node::AddNode>(n);
            ExecAddNode execNode(inputs, g, node, eng, stream);
            execNode.Execute();
            break;
        }
        default:
            throw std::runtime_error("invalid op type" + to_string(n->Type()));
    }
}

eng::ExecConvNode::ExecConvNode(
        unordered_map<string, dnnl::memory>& inputs,
        DnnlMemCache& memCache,
        grp::Graph& g,
        shared_ptr<node::ConvNode> node,
        dnnl::engine& eng,
        dnnl::stream& stream) : ExecNodeMKL(stream){
    auto wMemory = FindInputMemUtilWithCache(inputs, memCache, g.GetWeights(), eng, node->WeightName());
    auto bMemory = FindInputMemUtilWithCache(inputs, memCache, g.GetWeights(), eng, node->BiasName());
    auto srcMemory = FindInputMemUtilWithCache(inputs,  memCache, g.GetWeights(), eng, node->SrcInputName());

    auto srcDims = srcMemory.get_desc().dims();
    memory::dims wDims(node->WeightDims().begin(), node->WeightDims().end());
    memory::dims bDims(node->BiasDims().begin(), node->BiasDims().end());
    auto dstDims = utils::ComputeConvOutputDims(
            srcDims[0], srcDims[2], srcDims[3], node->KernelShape()[0], node->KernelShape()[1],
            node->Pads()[0], node->Pads()[1], node->Pads()[2], node->Pads()[3], node->Strides()[0], node->Strides()[1],
            node->WeightDims()[0]);
    auto dstMemory = dnnl::memory({dstDims, dt::f32, tag::nchw}, eng);
    // todo: if bias not exist?

    memory::dims strides_dims(node->Strides().begin(), node->Strides().end());
    auto pads = node->Pads();
    memory::dims paddingDimsLeft(pads.begin(), pads.begin()+pads.size()/2);
    memory::dims paddingDimsRight(pads.begin()+pads.size()/2, pads.end());
    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto convSrcMd = memory::desc(srcDims, dt::f32, tag::any);
    auto convWeightsMd = memory::desc(wDims, dt::f32, tag::any);
    auto convDstMd = memory::desc(dstDims, dt::f32, tag::any);
    // Create memory descriptor and memory object for input bias.
    auto userBiasMd = memory::desc(bDims, dt::f32, tag::a);
    // Create operation descriptor.
    auto convDesc = convolution_forward::desc(prop_kind::forward_inference,
                                               algorithm::convolution_direct, convSrcMd, convWeightsMd,
                                               userBiasMd, convDstMd, strides_dims, paddingDimsLeft,
                                               paddingDimsRight);

    // set post-relu
    post_ops postOps;
    primitive_attr convAttr;
    if(node->PostRelu()) {
        postOps.append_eltwise(1.f, algorithm::eltwise_relu, 0, 0);
    }
    if(node->PostSum()) {
        postOps.append_sum(1.f, srcMemory.get_desc().data_type());
        dstMemory = FindInputMemUtil(inputs, g.GetWeights(), eng, node->PostSumOutputName());
    }
    convAttr.set_post_ops(postOps);

    // Create primitive descriptor.
    auto convPrimDesc = convolution_forward::primitive_desc(convDesc, convAttr, eng);
    // Create the primitive.
    // In case dnnl creates a different format from the user defined one, we need to reorder them
    auto convSrcMem = mkl::PossibleReorder(srcMemory, convPrimDesc.src_desc(), stream, eng, to_string(node->ID()) + "_src_" + node->SrcInputName());
    auto convWeightsMem = mkl::PossibleReorder(wMemory, convPrimDesc.weights_desc(), stream, eng, to_string(node->ID()) + "_wei_" + node->WeightName());
//    dstMemory = mkl::PossibleReorder(dstMemory, convPrimDesc.dst_desc(), stream, eng, to_string(node->ID()) + "_dst_" + node->OutputName());
    if(!node->PostSum()) {
        dstMemory = dnnl::memory(convPrimDesc.dst_desc(), eng);
    }

    if(!memCache.Exists(node->WeightName())) {
        memCache.Put(node->WeightName(), convWeightsMem);
    }

    prim = convolution_forward(convPrimDesc);
    args.insert({
            {DNNL_ARG_SRC, convSrcMem},
            {DNNL_ARG_WEIGHTS, convWeightsMem},
            {DNNL_ARG_BIAS, bMemory},
            {DNNL_ARG_DST, dstMemory}
    });
    inputs.insert({node->Outputs()[0], dstMemory});
}

eng::ExecBNNode::ExecBNNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::BatchNormNode> node,
        dnnl::engine eng,
        dnnl::stream stream) : ExecNodeMKL(stream) {
    // retrieve srouce memory
    auto srcMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->SrcInputName());
    // scale and shift memory
    auto scaleShiftMd = memory::desc({2, node->Dim()[0]}, dt::f32, tag::nc);
    auto scaleShiftMem = memory(scaleShiftMd, eng);
    size_t scaleShiftSize = scaleShiftMd.get_size();
    mkl::WriteToDnnlMemoryFromTo(g.GetWeightHandle(node->WightName()), scaleShiftMem,
            0, 0, scaleShiftSize/2);
    mkl::WriteToDnnlMemoryFromTo(g.GetWeightHandle(node->BiasName()), scaleShiftMem,
            0, scaleShiftSize/2, scaleShiftSize/2);
    // set post-relu
    primitive_attr bnormAttr;
    normalization_flags flags = normalization_flags::use_scale_shift | normalization_flags::use_global_stats;
    if(node->PostRelu()) {
        post_ops postOps;
        postOps.append_eltwise(1, algorithm::eltwise_relu, 0, 0);
        bnormAttr.set_post_ops(postOps);
        flags |= normalization_flags::fuse_norm_relu;
    }
    // Create operation descriptor.
    auto bnormDesc = batch_normalization_forward::desc(
            prop_kind::forward_inference, srcMem.get_desc(), node->Epsilon(), flags);
    // Create primitive descriptor.
    auto bnormPrimDesc = batch_normalization_forward::primitive_desc(bnormDesc, bnormAttr, eng);
    auto dstMem = dnnl::memory(bnormPrimDesc.dst_desc(), eng);
    // Create memory objects using memory descriptors created by the primitive
    // descriptor: mean, variance, workspace.
    // NOTE: Here, the ReLU post-ops require a workspace for later usage in
    // backward propagation mode.
    auto meanMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->MeanName());
    auto varianceMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->VarName());
    // Create the primitive.
    prim = batch_normalization_forward(bnormPrimDesc);
    // possible reorder
//    auto BnSrcMem = mkl::PossibleReorder(srcMem, bnormPrimDesc.src_desc(), stream, eng);
//    auto BnMeanMem = mkl::PossibleReorder(meanMem, bnormPrimDesc.mean_desc(), stream, eng);
//    auto BnVarMem = mkl::PossibleReorder(varianceMem, bnormPrimDesc.variance_desc(), stream, eng);
    auto BnSrcMem = srcMem;
    auto BnMeanMem = meanMem;
    auto BnVarMem = varianceMem;
    // todo under some cases, we can not do in-place bn, so uncomment below line
    //    inputs.insert({node->Outputs()[0], dstMem});
    //      e.g, in -> BN ->+-> out
    //            |          |
    //            |          |
    //            ·----------·
    args = {
            {DNNL_ARG_SRC, BnSrcMem},
            {DNNL_ARG_MEAN, BnMeanMem},
            {DNNL_ARG_VARIANCE, BnVarMem},
            {DNNL_ARG_SCALE_SHIFT, scaleShiftMem},
            {DNNL_ARG_DST, dstMem}
    };
    inputs.insert({node->OutputName(), dstMem});
}

eng::ExecShapeNode::ExecShapeNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::ShapeNode> node,
        dnnl::engine eng) {
    data = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputName());
    dst = dnnl::memory({{int64_t(data.get_desc().dims().size())}, dt::s32, tag::a}, eng);
    inputs.insert({node->OutputName(), dst});
}

eng::ExecGatherNode::ExecGatherNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::GatherNode> node,
        dnnl::engine eng) {
    axis = node->Axis();
    data = FindInputMemUtil(inputs, g.GetWeights(), eng, node->DataName());
    indices = FindInputMemUtil(inputs, g.GetWeights(), eng, node->IndicesName());
    auto gDims = utils::gatherDims(data.get_desc().dims(), indices.get_desc().dims(), axis);
    dst = dnnl::memory({gDims, data.get_desc().data_type(), mkl::ChosseDefaultTag(gDims.size())}, eng);
    inputs.insert({node->OutputName(), dst});
}

void eng::ExecGatherNode::Execute() {
    auto dtype = data.get_desc().data_type();
    switch (dtype) {
        case dt::s32 : {
            utils::gather((int32_t *)data.get_data_handle(),
                   (int32_t*)dst.get_data_handle(),
                   (int64_t*)(indices.get_data_handle()),
                   data.get_desc().dims(),
                   indices.get_desc().dims(),
                   axis);
        }
        case dt::f32 : {
            utils::gather((float*)data.get_data_handle(),
                   (float*)dst.get_data_handle(),
                   (int64_t*)(indices.get_data_handle()),
                   data.get_desc().dims(),
                   indices.get_desc().dims(),
                   axis);
        }
        default:
            throw std::invalid_argument("data type not supported: " + to_string(static_cast<int>(dtype)));
    }
}

eng::ExecMulNode::ExecMulNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::MulNode> node,
        dnnl::engine eng) {
    a = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputAName());
    b = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputBName());
    dnnl::memory::dims dims;
    auto aDims = a.get_desc().dims();
    auto bDims = b.get_desc().dims();
    int aSize = aDims.size();
    int bSize = bDims.size();
    for(int i=0; i<aSize && i<bSize; i++) {
        int adim = aSize-1-i;
        int bdim = bSize-1-i;
        if ( adim < 0 ) {
            dims.push_back(bDims[bdim]);
        } else if ( bdim < 0 ) {
            dims.push_back(aDims[adim]);
        } else {
            if(aDims[adim] != bDims[bdim] && aDims[adim] != 1 && bDims[bdim] != 1) {
                throw std::runtime_error("unbroadcastable dims");
            }
            dims.push_back(max(aDims[adim], bDims[bdim]));
        }
    }
    reverse(dims.begin(), dims.end());
    dst = dnnl::memory({dims, a.get_desc().data_type(), mkl::ChosseDefaultTag(dims.size())}, eng);
    inputs.insert({node->OutputName(), dst});
}

// todo: a and b's tags, should we change them to default?
void eng::ExecMulNode::Execute() {
    auto aDims = a.get_desc().dims();
    auto bDims = b.get_desc().dims();
    if(aDims.empty() || bDims.empty()) {
        throw std::invalid_argument("dim size cannot be 0");
    }
    int aSize = aDims.size();
    int bSize = bDims.size();
    int dims = max(aSize, bSize);
    int aCurr[dims];
    int bCurr[dims];
    int aBroadDims[dims];
    int bBroadDims[dims];
    // prepare index
    for(int i=0; i<dims; i++) {
        aCurr[i] = 0;
        bCurr[i] = 0;
        // broadcast dims
        int broadDim = dims-i-1;
        int adim = aSize-1-i;
        int bdim = bSize-1-i;
        if ( adim < 0 ) {
            aBroadDims[broadDim] = 1;
        } else if ( bdim < 0 ) {
            bBroadDims[broadDim] = 1;
        } else {
            if(aDims[adim] != bDims[bdim] && aDims[adim] != 1 && bDims[bdim] != 1) {
                throw std::runtime_error("unbroadcastable dims");
            }
            aBroadDims[broadDim] = aDims[adim];
            bBroadDims[broadDim] = bDims[bdim];
        }
    }
    if(a.get_desc().data_type() == dt::f32) {
        switch (dims) {
            case 1 :
                return utils::mul1d((float*)a.get_data_handle(), (float*)b.get_data_handle(), (float*)dst.get_data_handle(), aBroadDims, bBroadDims);
            case 2 :
                return utils::mul2d((float*)a.get_data_handle(), (float*)b.get_data_handle(), (float*)dst.get_data_handle(), aBroadDims, bBroadDims);
            case 3 :
                return utils::mul3d((float*)a.get_data_handle(), (float*)b.get_data_handle(), (float*)dst.get_data_handle(), aBroadDims, bBroadDims);
            default:
                throw std::runtime_error("higher than 3 dimension's mul is nor supported");
        }
    } else if(a.get_desc().data_type() == dt::s32) {
        switch (dims) {
            case 1 :
                return utils::mul1d((int32_t*)a.get_data_handle(), (int32_t*)b.get_data_handle(), (int32_t*)dst.get_data_handle(), aBroadDims, bBroadDims);
            case 2 :
                return utils::mul2d((int32_t*)a.get_data_handle(), (int32_t*)b.get_data_handle(), (int32_t*)dst.get_data_handle(), aBroadDims, bBroadDims);
            case 3 :
                return utils::mul3d((int32_t*)a.get_data_handle(), (int32_t*)b.get_data_handle(), (int32_t*)dst.get_data_handle(), aBroadDims, bBroadDims);
            default:
                throw std::runtime_error("higher than 3 dimension's mul is nor supported");
        }
    }

}

eng::ExecUnsqueezeNode::ExecUnsqueezeNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::UnsqueezeNode> node,
        dnnl::engine eng) {
    auto data = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputName());
    auto newDims = data.get_desc().dims();
    newDims.erase(std::remove(newDims.begin(), newDims.end(), 1), newDims.end());
    dst = dnnl::memory({newDims, data.get_desc().data_type(), mkl::ChosseDefaultTag(newDims.size())}, eng, data.get_data_handle());
    inputs.insert({node->OutputName(), dst});
}

eng::ExecReshapeNode::ExecReshapeNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::ReshapeNode> node,
        dnnl::engine eng) {
    auto data = FindInputMemUtil(inputs, g.GetWeights(), eng, node->DataName());
    auto shape = FindInputMemUtil(inputs, g.GetWeights(), eng, node->ShapeName());
    dnnl::memory::dims newDims((char*)shape.get_data_handle(), (char*)(shape.get_data_handle())+shape.get_desc().get_size());
    dst = dnnl::memory({newDims, data.get_desc().data_type(), mkl::ChosseDefaultTag(newDims.size())}, eng, data.get_data_handle());
    inputs.insert({node->OutputName(), dst});
}

eng::ExecFlattenNode::ExecFlattenNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::FlattenNode> node,
        dnnl::engine eng,
        dnnl::stream stream) {
    auto srcMemory = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputsName());
    auto dims = srcMemory.get_desc().dims();
    int d1 = 1, d2 = 1;
    for(int i=0; i<node->Axis(); i++) {
        d1 *= dims[i];
    }
    for(int i=node->Axis(); i<dims.size(); i++) {
        d2 *= dims[i];
    }
    dnnl::memory::dims newDims({d1, d2});
    dnnl::memory::dims oldDims = srcMemory.get_desc().dims();
    dst = mkl::PossibleReorder(srcMemory,
            {oldDims, srcMemory.get_desc().data_type(), mkl::ChosseDefaultTag(oldDims.size())},
            stream,
            eng,
            to_string(node->ID()) + node->InputsName());
    dst = memory(dst.get_desc().reshape(newDims), eng, dst.get_data_handle());
    inputs.insert({node->OutputName(), dst});
}

eng::ExecPadNode::ExecPadNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::PadNode> node,
        dnnl::engine eng) {
    bool allZero = true;
    for(int pad : node->Pads()) {
        if(pad != 0) {
            allZero = false;
            break;
        }
    }
    auto dataMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputName());
    if(allZero) {
        inputs.insert({node->OutputName(), dataMem});
    } else {
        throw runtime_error("pad node is not supported");
    }
}

eng::ExecConcatNode::ExecConcatNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::ConcatNode> node,
        dnnl::engine eng,
        dnnl::stream stream) : ExecNodeMKL(stream) {

    vector<dnnl::memory::desc> srcMds;
    vector<dnnl::memory> srcMems;
    for (string name : node->InputsName()) {
        auto mem = FindInputMemUtil(inputs, g.GetWeights(), eng, name);
        srcMds.push_back(mem.get_desc());
        srcMems.push_back(mem);
    }

    auto concatPd = concat::primitive_desc(node->Axis(), srcMds, eng);
    auto dstMem = memory(concatPd.dst_desc(), eng);
    prim = concat(concatPd);

    for (int n = 0; n < srcMds.size(); ++n)
        args.insert({DNNL_ARG_MULTIPLE_SRC + n, srcMems[n]});
    args.insert({DNNL_ARG_DST, dstMem});

    inputs.insert({node->OutputName(), dstMem});
}

eng::ExecGemmNode::ExecGemmNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::GemmNode> node,
        dnnl::engine eng,
        dnnl::stream stream) : ExecNodeMKL(stream) {
    auto srcMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputName());
    auto weightMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->WeightName());
    auto biasMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->BiasName());

    if(node->TransA()) {
        auto desc = srcMem.get_desc().permute_axes({1, 0});
        srcMem = dnnl::memory(desc, eng, srcMem.get_data_handle());
    }
    if(node->TransB()) {
        auto desc = weightMem.get_desc().permute_axes({1, 0});
        weightMem = dnnl::memory(desc, eng, weightMem.get_data_handle());
    }

    auto srcDim = srcMem.get_desc().dims();
    auto weightDim = weightMem.get_desc().dims();

    if(srcDim[1] != weightDim[0]) {
        throw std::runtime_error("invalid gemm dims");
    }

    auto biasDim = biasMem.get_desc().dims();
    if(biasDim.size() > 2 || biasDim.empty()) {
        throw std::runtime_error("invalid bias dims");
    }
    // broadcast bias dims
    if(biasDim.size() != 2) {
        if(biasDim[0] == srcDim[0]) {
            biasMem = dnnl::memory({{biasDim[0], 1}, biasMem.get_desc().data_type(), tag::ab}, eng, biasMem.get_data_handle());
        } else {
            biasMem = dnnl::memory({{1, biasDim[0]}, biasMem.get_desc().data_type(), tag::ab}, eng, biasMem.get_data_handle());
        }
    }

    memory dstMem({{srcDim[0], weightDim[1]}, srcMem.get_desc().data_type(),tag::ab}, eng);

    primitive_attr attr;
    if(node->Bias()) {
        matmul::desc desc(srcMem.get_desc(), weightMem.get_desc(), biasMem.get_desc(), dstMem.get_desc());
        matmul::primitive_desc primDesc(desc, attr, eng);
        prim = matmul(primDesc);
    } else {
        matmul::desc desc(srcMem.get_desc(), weightMem.get_desc(), dstMem.get_desc());
        matmul::primitive_desc primDesc(desc, attr, eng);
        prim = matmul(primDesc);
    }


    args.insert({DNNL_ARG_SRC, srcMem});
    args.insert({DNNL_ARG_WEIGHTS, weightMem});
    args.insert({DNNL_ARG_BIAS, biasMem});
    args.insert({DNNL_ARG_DST, dstMem});
    inputs.insert({node->OutputName(), dstMem});
}

eng::ExecAvgPoolingNode2D::ExecAvgPoolingNode2D(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::AvgPoolingNode> node,
        dnnl::engine eng,
        dnnl::stream stream) : ExecNodeMKL(stream) {
    auto srcMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputName());

    auto srcDims = srcMem.get_desc().dims();
    auto dstDims = utils::ComputeConvOutputDims(
            srcDims[0], srcDims[2], srcDims[3], node->KernelShape()[0], node->KernelShape()[1],
            node->Pads()[0], node->Pads()[1], node->Pads()[2], node->Pads()[3], node->Strides()[0], node->Strides()[1],
            srcDims[1]);
    auto dstMem = memory({dstDims, dt::f32, tag::any}, eng);

    auto pads = node->Pads();
    memory::dims strides(node->Strides().begin(), node->Strides().end());
    memory::dims kernelShape(node->KernelShape().begin(), node->KernelShape().end());
    memory::dims paddingDimsLeft(pads.begin(), pads.begin()+pads.size()/2);
    memory::dims paddingDimsRight(pads.begin()+pads.size()/2, pads.end());
    memory::dims dilationDims(node->KernelShape().size(), 0);

    auto poolingDesc = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_avg,
                                                srcMem.get_desc(), dstMem.get_desc(),
                                                strides, kernelShape,
                                                dilationDims, paddingDimsLeft, paddingDimsRight);

    auto poolingPd = pooling_v2_forward::primitive_desc(poolingDesc, eng);

    prim = pooling_v2_forward(poolingPd);

    args.insert({DNNL_ARG_SRC, srcMem});
    args.insert({DNNL_ARG_DST, dstMem});
    inputs.insert({node->OutputName(), dstMem});
}

eng::ExecGlobalAvgPoolingNode2D::ExecGlobalAvgPoolingNode2D(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::GlobalAvgPoolingNode> node,
        dnnl::engine eng,
        dnnl::stream stream) : ExecNodeMKL(stream) {
    auto srcMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputName());

    auto srcDims = srcMem.get_desc().dims();
    if(srcDims.size() != 4) {
        throw runtime_error(to_string(srcDims.size()-2) + "d pooling is not supported now");
    }

    auto dstMem = memory({{srcDims[0], srcDims[1], 1, 1}, dt::f32, tag::nchw}, eng);

    auto poolingDesc = pooling_v2_forward::desc(prop_kind::forward_inference, algorithm::pooling_avg_exclude_padding,
                                                srcMem.get_desc(), dstMem.get_desc(),
                                                {1, 1}, {srcDims[2], srcDims[3]},
                                                {0, 0}, {0, 0}, {0, 0});

    auto poolingPd = pooling_v2_forward::primitive_desc(poolingDesc, eng);

    prim = pooling_v2_forward(poolingPd);

    args.insert({DNNL_ARG_SRC, srcMem});
    args.insert({DNNL_ARG_DST, dstMem});
    inputs.insert({node->OutputName(), dstMem});
}

eng::ExecAddNode::ExecAddNode(
        unordered_map<string, dnnl::memory>& inputs,
        grp::Graph& g,
        shared_ptr<node::AddNode> node,
        dnnl::engine eng,
        dnnl::stream stream) : ExecNodeMKL(stream) {

    auto srcAMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputA());
    auto srcBMem = FindInputMemUtil(inputs, g.GetWeights(), eng, node->InputB());

    vector<float> scales(2, 1);
    vector<memory::desc> srcMemDescs = {srcAMem.get_desc(), srcBMem.get_desc()};

    auto sumPD = sum::primitive_desc(scales, srcMemDescs, eng);
    prim = sum(sumPD);

    auto dstMem = memory(sumPD.dst_desc(), eng);

    args.insert({DNNL_ARG_DST, dstMem});
    args.insert({DNNL_ARG_MULTIPLE_SRC, srcAMem});
    args.insert({DNNL_ARG_MULTIPLE_SRC+1, srcBMem});
    inputs.insert({node->OutputName(), dstMem});
}