//
// Created by 田地 on 2020/9/27.
//

#ifndef SERVER_HELPER_H
#define SERVER_HELPER_H

#include "dnnl.hpp"
#include "string"
#include <vector>

using namespace std;

namespace utils {
    dnnl::memory::dims ComputeConvOutputDims(int, int, int, int, int, int, int, int, int, int, int, int);
    vector<int64_t> gatherDims(const vector<int64_t>& srcDims, const vector<int64_t>& indicesDims, int axis);
    template<typename T>
    void gather(T* src, T* dst, const int64_t* indices, const vector<int64_t>& srcDims, const vector<int64_t>& indicesDims, int axis);
    template<typename T>
    void mul1d(T* a, T* b, T* dst, int adim[1], int bdim[1]);
    template<typename T>
    void mul2d(T* a, T* b, T* dst, int adim[2], int bdim[2]);
    template<typename T>
    void mul3d(T* a, T* b, T* dst, int adim[3], int bdim[3]);

// |          | axis |          |
//               |
//               V
// |       |  indices  |           |
    template<typename T>
    void gather(T* src, T* dst, const int64_t* indices, const vector<int64_t>& srcDims, const vector<int64_t>& indicesDims, int axis) {
        if(axis < 0 || axis >= srcDims.size()) {
            throw std::invalid_argument("axis out of range: " + to_string(axis) + ", [0, " + to_string(srcDims.size()) + "]");
        }
        int bfAxisProd = 1;
        int srcCpySize = 1;
        int axisSize = srcDims[axis];
        for(int i=0; i<srcDims.size(); i++) {
            if(i<axis) bfAxisProd *= srcDims[i];
            if(i>axis) srcCpySize *= srcDims[i];
        }
//    srcCpySize *= sizeof(T);
        int idxNumOfEle = 1;
        int idxUnitSize = indicesDims[indicesDims.size()-1];
        for(int i=0; i<indicesDims.size()-1; i++) {
            idxNumOfEle *= indicesDims[i];
        }
        for(int i=0; i<bfAxisProd; i++) {
            for(int j=0; j<idxNumOfEle; j++) {
                for(int k=0; k<idxUnitSize; k++) {
                    int64_t idx = indices[j*idxUnitSize + k];
                    if(idx < 0 || idx >= srcDims[axis]) {
                        throw std::invalid_argument("index out of range: " + to_string(idx) + ", [0, " + to_string(srcDims[axis]) + "]");
                    }
                    // index in src: src + i * axisSize * srcCpySize
                    // offset in src: idx * srcCpySize
                    void* st = src + i * axisSize * srcCpySize + idx * srcCpySize;
                    memcpy(dst, st, srcCpySize*sizeof(T));
                    dst += srcCpySize;
                }
            }
        }
    }

// with broadcasting
    template<typename T>
    void mul1d(T* a, T* b, T* dst, int adim[1], int bdim[1]) {
        int dia = adim[0] == 1? 0 : 1;
        int dib = bdim[0] == 1? 0 : 1;
        int l = max(adim[0], bdim[0]);
        int c = 0;
        for(int ia=0, ib=0; ia<l && ib<l; ia+=dia, ib+=dib) {
            dst[c++] = a[ia] * b[ib];
        }
    }

    template<typename T>
    void mul2d(T* a, T* b, T* dst, int adim[2], int bdim[2]) {
        int dia = adim[0] == 1? 0 : adim[1];
        int dja = adim[1] == 1? 0 : 1;
        int dib = bdim[0] == 1? 0 : bdim[1];
        int djb = bdim[1] == 1? 0 : 1;
        int l1 = max(adim[0]*adim[1], bdim[0]*bdim[1]);
        int l2 = max(adim[1], bdim[1]);
        int c = 0;
        for(int ia=0, ib=0; ia<l1 && ib<l1; ia+=dia, ib+=dib) {
            for(int ja=0, jb=0; ja<l2 && jb <l2; ja+=dja, jb+=djb) {
                dst[c++] = a[ia+ja] * b[ib+jb];
            }
        }
    }

// with broadcasting
    template<typename T>
    void mul3d(T* a, T* b, T* dst, int adim[3], int bdim[3]) {
        int dia = adim[0] == 1? 0 : adim[1]*adim[2];
        int dja = adim[1] == 1? 0 : adim[2];
        int dka = adim[2] == 1? 0 : 1;
        int dib = bdim[0] == 1? 0 : bdim[1]*bdim[2];
        int djb = bdim[1] == 1? 0 : bdim[2];
        int dkb = bdim[2] == 1? 0 : 1;
        int l1 = max(adim[0]*adim[1]*adim[2], bdim[0]*bdim[1]*bdim[2]);
        int l2 = max(adim[1]*adim[2], bdim[1]*bdim[2]);
        int l3 = max(adim[2], bdim[2]);
        int c = 0;
        for(int ia=0, ib=0; ia<l1 && ib<l1; ia+=dia, ib+=dib) {
            for(int ja=0, jb=0; ja<l2 && jb <l2; ja+=dja, jb+=djb) {
                for(int ka=0, kb=0; ka<l3 && kb<l3; ka+=dka, kb+=dkb) {
                    dst[c++] = a[ia+ja+ka] * b[ib+jb+kb];
                }
            }
        }
    }
}



#endif //SERVER_HELPER_H
