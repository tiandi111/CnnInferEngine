//
// Created by 田地 on 2020/10/24.
//

#ifndef SERVER_TEST_H
#define SERVER_TEST_H

#include <stdexcept>
#include <string>

using namespace std;

template<typename T>
inline void AssertEqual(T a, T b, string msg) {
    if(a != b) {
        throw std::runtime_error("assert failed: " + msg);
    }
}

inline void AssertFalse(bool a, string msg) {
    if(a) {
        throw std::runtime_error("assert failed: " + msg);
    }
}

#endif //SERVER_TEST_H
