//
// Created by 田地 on 2020/10/26.
//

#include "utils.h"

using namespace std;

dnnl::memory::dims utils::ComputeConvOutputDims(
        int n,
        int h, int w,
        int hk, int wk,
        int hlp, int hhp,
        int wlp, int whp,
        int hs, int ws,
        int oc) {
    dnnl::memory::dims outDims = {
            n, oc,
            (h-hk+hlp+hhp)/hs+1,
            (w-wk+wlp+whp)/ws+1};
    return outDims;
}

vector<int64_t> utils::gatherDims(const vector<int64_t>& srcDims, const vector<int64_t>& indicesDims, int axis) {
    if(axis < 0 || axis >= srcDims.size()) {
        throw std::invalid_argument("axis out of range: " + to_string(axis) + ", [0, " + to_string(srcDims.size()) + "]");
    }
    vector<int64_t> gDims;
    gDims.insert(gDims.end(), srcDims.begin(), srcDims.begin()+axis-1);
    gDims.insert(gDims.end(), indicesDims.begin(), indicesDims.end());
    gDims.insert(gDims.end(), srcDims.begin()+axis+1, srcDims.end());
    return gDims;
}