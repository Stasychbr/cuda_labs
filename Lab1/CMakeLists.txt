cmake_minimum_required(VERSION 3.26)
project(ndarray_add_lab CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

file(GLOB SOURCE_FILES *.cpp *.cu)

add_executable(ndarray_add_lab ${SOURCE_FILES})

set_target_properties(ndarray_add_lab PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
