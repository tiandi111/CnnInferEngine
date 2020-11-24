//
// Created by 田地 on 2020/9/26.
//

#ifndef SERVER_TENSOR_H
#define SERVER_TENSOR_H

#include <vector>

using namespace std;

namespace ten {
    enum DataType {
        unknown,
        f32,
        i64,
        i8,
        ui8
    };

    class Tensor {
    private:
        vector<int64_t> dims;
        DataType dtype;
        vector<char> data;
        int64_t bytes;

        void computeBytes();
        void checkBuffer();
    public:
        Tensor() = default;
        ~Tensor() = default;
        Tensor(vector<int64_t> dims, DataType t);
        Tensor(vector<int64_t> dims, DataType t, vector<char>& data);
        Tensor(vector<int64_t> dims, DataType t, void* first, int64_t byteSize) :
                dims(dims), dtype(t), bytes(byteSize) {
            data = vector<char>((char*)first, ((char*)first)+byteSize);
        };
        void Write(void* handle);
        const vector<char>& Data() const;
        const DataType Type() const;
        const vector<int64_t>& Dims() const;
        void* GetDataHandle() { return data.data(); }
        uint64_t ArgMax1D();
        void GlobalNormalization();
        inline vector<char>& MutableData() { return data; }
        uint64_t Len();
        // todo: finish this element wise multiplcation
//        void Multiply(Tensor& A, vector<int>& Axes, bool InPlace);
    };
}

#endif //SERVER_TENSOR_H

