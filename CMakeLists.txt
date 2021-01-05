cmake_minimum_required(VERSION 3.10)
project(server)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_PREFIX_PATH /Users/tiandi03/OpenSource/libtorch)

include_directories(.)
link_directories(/usr/local/lib)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(tests
        test/main.cpp
        test/test.h
        test/util_test.h
        test/util_test.cpp
        test/engine_test.h
        test/engine_test.cpp
        test/tensor_test.h
        test/tensor_test.cpp
        test/intergration_test.h
        test/intergration_test.cpp
        test/utils_test.h
        test/utils_test.cpp
        engine.cpp
        engine.h
        graph.cpp
        graph.h
        context.cpp
        context.h
        loader.cpp
        loader.h
        mkl.cpp
        mkl.h
        node.cpp
        node.h
        onnx.pb.cc
        onnx.pb.h
        tensor.cpp
        tensor.h
        utils.h
        utils.cpp
        utils/dataset/cifar.h
        utils/dataset/cifar.cpp)

target_link_libraries(tests ${Protobuf_LIBRARIES})
target_link_libraries(tests -ldnnl)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(benchmark
        benchmark/benchmark.cpp
        benchmark/benchmark.h
        benchmark/main.cpp
        engine.cpp
        engine.h
        graph.cpp
        graph.h
        context.cpp
        context.h
        loader.cpp
        loader.h
        mkl.cpp
        mkl.h
        node.cpp
        node.h
        onnx.pb.cc
        onnx.pb.h
        tensor.cpp
        tensor.h
        utils.h
        utils.cpp
        utils/dataset/cifar.h
        utils/dataset/cifar.cpp
        ${PROTO_SRCS}
        ${PROTO_HDRS})

target_link_libraries(benchmark ${Protobuf_LIBRARIES})
target_link_libraries(benchmark -ldnnl)
target_link_libraries(benchmark ${TORCH_LIBRARIES})





