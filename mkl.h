//
// Created by 田地 on 2020/9/25.
//

#ifndef SERVER_MKL_H
#define SERVER_MKL_H

#include "dnnl.hpp"
#include "tensor.h"
#include <vector>
#include <stdexcept>

using namespace std;
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace mkl {

    primitive CnnPrimitive(
            const engine& eng,
            const stream& stream,
            const vector<int>& srcDims,
            const vector<int>& wDims,
            const vector<int>& bDims,
            const vector<int>& dstDims,
            const vector<int>& strides,
            const vector<int>& padding);

    dt TensorDtypeToMKLType(ten::DataType dt);

    // input: the length of the dimensions
    tag ChosseDefaultTag(int dimLen);

    memory PossibleReorder(memory& srcMemory,
            const memory::desc& targetDesc,
            const stream& stream,
            const engine& eng,
            const string& msg);

    inline void ReadFromDnnlMemory(const void *handle, const dnnl::memory &mem) {
        dnnl::engine eng = mem.get_engine();
        size_t bytes = mem.get_desc().get_size();

        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
            for (size_t i = 0; i < bytes; ++i)
                ((uint8_t *)handle)[i] = src[i];
        }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();

        cl_int ret = clEnqueueReadBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueReadBuffer failed.");
    }
#endif
    }

    inline void WriteToDnnlMemory(const void *handle, dnnl::memory &mem) {
        dnnl::engine eng = mem.get_engine();
        size_t bytes = mem.get_desc().get_size();

        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
            for (size_t i = 0; i < bytes; ++i)
                dst[i] = ((uint8_t *)handle)[i];
        }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();
        size_t bytes = mem.get_desc().get_size();

        cl_int ret = clEnqueueWriteBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueWriteBuffer failed.");
    }
#endif
    }

    inline void WriteToDnnlMemoryFromTo(const void *handle, dnnl::memory &mem, size_t srcFrom, size_t dstFrom, size_t len) {
        size_t bytes = mem.get_desc().get_size();
        if(srcFrom<0 || dstFrom<0 || (dstFrom+len) > bytes) {
            throw invalid_argument("srcFrom:" + to_string(srcFrom) +
                                        " dstFrom:" + to_string(dstFrom) +
                                        " len:" + to_string(len) +
                                        " dnnl mem size:" + to_string(bytes));
        }
        dnnl::engine eng = mem.get_engine();

        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
            for (size_t i = 0; i < len; ++i)
                dst[dstFrom+i] = ((uint8_t *)handle)[srcFrom+i];
        }
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        else if (eng.get_kind() == dnnl::engine::kind::gpu) {
        dnnl::stream s(eng);
        cl_command_queue q = s.get_ocl_command_queue();
        cl_mem m = mem.get_ocl_mem_object();
        size_t bytes = mem.get_desc().get_size();

        cl_int ret = clEnqueueWriteBuffer(
                q, m, CL_TRUE, 0, bytes, handle, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            throw std::runtime_error("clEnqueueWriteBuffer failed.");
    }
#endif
    }

}

#endif //SERVER_MKL_H
