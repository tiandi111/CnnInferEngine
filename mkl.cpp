//
// Created by 田地 on 2020/9/25.
//

#include "mkl.h"
#include <iostream>

primitive mkl::CnnPrimitive(
        const engine& eng,
        const stream& stream,
        const vector<int>& srcDims,
        const vector<int>& wDims,
        const vector<int>& bDims,
        const vector<int>& dstDims,
        const vector<int>& strides,
        const vector<int>& padding) {
    // Source (src), weights, bias, and destination (dst) tensors
    // dimensions.
    memory::dims src_dims(srcDims.begin(), srcDims.end());
    memory::dims weights_dims(wDims.begin(), wDims.end());
    memory::dims bias_dims(bDims.begin(), bDims.end());
    memory::dims dst_dims(dstDims.begin(), dstDims.end());
    // Strides, padding dimensions.
    memory::dims strides_dims(strides.begin(), strides.end());
    memory::dims padding_dims_l;
    memory::dims padding_dims_r;
    for(int i=0; i<padding.size(); i+=2) {
        padding_dims_l.push_back(padding[i]);
        padding_dims_r.push_back(padding[i+1]);
    }
    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);
    // Create memory descriptor and memory object for input bias.
    auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
    // Create operation descriptor.
    auto conv_desc = convolution_forward::desc(prop_kind::forward_training,
                                               algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                               user_bias_md, conv_dst_md, strides_dims, padding_dims_l,
                                               padding_dims_r);
    primitive_attr conv_attr;
    // Create primitive descriptor.
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, eng);
    // Create the primitive.
    return convolution_forward(conv_pd);
}

dt mkl::TensorDtypeToMKLType(ten::DataType dt) {
    switch (dt) {
        case ten::DataType::f32 :
            return dt::f32;
        case ten::DataType::i64 :
            return dt::s32;
        case ten::DataType::i8 :
            return dt::s8;
        default:
            throw invalid_argument("unknown data type" + to_string(dt));
    }
}

tag mkl::ChosseDefaultTag(int dimLen) {
    switch (dimLen) {
        case 1 :
            return tag::a;
        case 2 :
            return tag::ab;
        case 3 :
            return tag::abc;
        case 4 :
            return tag::abcd;
        case 5 :
            return tag::abcde;
        case 6 :
            return tag::abcdef;
        default:
            throw std::runtime_error("data dimension higher than 6 is not supported now");
    }
}

memory mkl::PossibleReorder(memory& srcMemory,
        const memory::desc& targetDesc,
        const stream& stream,
        const engine& eng,
        const string& msg) {
    if(srcMemory.get_desc() != targetDesc) {
        auto newMem = memory(targetDesc, eng);
        reorder(srcMemory, newMem)
                .execute(stream, srcMemory, newMem);
//        cout<< "reorder " + msg <<endl;
        return newMem;
    }
    return srcMemory;
}
