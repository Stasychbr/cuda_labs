#pragma once
#include <device_launch_parameters.h>

#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "cuda_runtime.h"
#include "cpundarray.h"
#include "printer.tpp"

#include <cstdio>

constexpr size_t TILE_SIZE = 32;

void dev_sync() {
    cudaError_t rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
        throw std::runtime_error(std::format("Device synchronization error: {}",
                                             cudaGetErrorString(rc)));
    }
}

template <typename T>
__global__ void fill_kernel_val(T* array, const T value) {
    int i = threadIdx.x;
    array[i] = value;
}

template <typename T1, typename T2>
__global__ void fill_kernel_buf(T1* array, const T2* buffer) {
    int i = threadIdx.x;
    array[i] = static_cast<T1>(buffer[i]);
}

template <typename T1, typename T2>
void fill_helper_buf(T1* array, size_t elems_n, const T2* buffer) {
    fill_kernel_buf<<<1, elems_n>>>(array, buffer);
    dev_sync();
}

template <typename T1, typename T2>
void fill_helper_val(T1* array, size_t elems_n, const T2 value) {
    T1 casted_value = static_cast<T1>(value);
    fill_kernel_val<<<1, elems_n>>>(array, casted_value);
    dev_sync();
}

template <typename T1, typename T2>
__global__ void add_kernel_buf(T1* a, const T2* b) {
    int i = threadIdx.x;
    a[i] += static_cast<T1>(b[i]);
}

template <typename T1, typename T2>
void add_helper_buf(T1* a, const T2* b, size_t elems_n) {
    add_kernel_buf<<<1, elems_n>>>(a, b);
    dev_sync();
}

template <typename T>
__global__ void add_kernel_val(T* a, const T b) {
    int i = threadIdx.x;
    a[i] += b;
}

template <typename T1, typename T2>
void add_helper_val(T1* a, const T2 value, size_t elems_n) {
    T1 casted_value = static_cast<T1>(value);
    add_kernel_val<<<1, elems_n>>>(a, casted_value);
    dev_sync();
}

template <typename T1, typename T2>
__global__ void diff_kernel(T1* a, const T2* b) {
    int i = threadIdx.x;
    a[i] -= static_cast<T1>(b[i]);
}

template <typename T1, typename T2>
void diff_helper(T1* a, const T2* b, size_t elems_n) {
    diff_kernel<<<1, elems_n>>>(a, b);
    dev_sync();
}

__host__ __device__ inline size_t ceil_int_div(size_t x, size_t y) {
    return 1 + (x - 1) / y;
}

template <typename T>
__global__ void mlp_kernel_shared(T* A, T* B, T* C, size_t rA, size_t cA, size_t rB, size_t cB) {
    __shared__ T sA[TILE_SIZE][TILE_SIZE];
    __shared__ T sB[TILE_SIZE][TILE_SIZE];

    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    T c_sum = 0;

    for (size_t k = 0; k < ceil_int_div(cA, TILE_SIZE); k++) {
        if (row < rA && ((size_t)threadIdx.x + k * TILE_SIZE) < cA) {
            sA[threadIdx.y][threadIdx.x] = A[row * cA + threadIdx.x + k * TILE_SIZE];
        }
        else {
            sA[threadIdx.y][threadIdx.x] = 0;
        }
        if (col < cB && ((size_t)threadIdx.y + k * TILE_SIZE) < rB) {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k * TILE_SIZE) * cB + col];
        }
        else {
            sB[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (size_t j = 0; j < TILE_SIZE; j++) {
            c_sum += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < rA && col < cB) {
        C[row * cB + col] = c_sum;
    }
}

template <typename T>
void mlp_helper(T* A, T* B, T* C, size_t rA, size_t cA, size_t rB, size_t cB) {
    dim3 grid_shape(ceil_int_div(cB, TILE_SIZE), ceil_int_div(rA, TILE_SIZE), 1);
    dim3 block_shape(TILE_SIZE, TILE_SIZE, 1);
    mlp_kernel_shared<<<grid_shape, block_shape>>>(A, B, C, rA, cA, rB, cB);
    dev_sync();
}

template <typename T>
__global__ void mlp_kernel_naive(T* A, T* B, T* C, size_t rA, size_t cA, size_t rB, size_t cB) {
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;

    T c_sum = 0;

    if (row < rA && col < cB) {
        for (size_t i = 0; i < cA; i++) {
            c_sum += A[row * cA + i] * B[i * cB + col];
        }
        C[row * cB + col] = c_sum;
    }
}

template <typename T>
void naive_mlp_helper(T* A, T* B, T* C, size_t rA, size_t cA, size_t rB, size_t cB) {
    dim3 grid_shape(ceil_int_div(cB, TILE_SIZE), ceil_int_div(rA, TILE_SIZE), 1);
    dim3 block_shape(TILE_SIZE, TILE_SIZE, 1);
    mlp_kernel_naive<<<grid_shape, block_shape>>>(A, B, C, rA, cA, rB, cB);
    dev_sync();
}

template <typename T>
class GPUNDArray : public NDArray {
    class GPUMemory {
       public:
        T* _ptr;

        GPUMemory(size_t elems_n) {
            cudaError_t rc = cudaMalloc((void**)&_ptr, elems_n * sizeof(T));
            if (rc != cudaSuccess) {
                throw std::runtime_error(
                    std::format("CUDA memory allocation error: {}",
                                cudaGetErrorString(rc)));
            }
        }

        T* get_ptr() { return _ptr; }

        ~GPUMemory() { 
            cudaError_t rc = cudaFree((void*)_ptr); 
            if (rc != cudaSuccess) {
                std::cerr << std::format("CUDA memory free error: {}",
                                cudaGetErrorString(rc));
            }
        }
    };

    size_t _ndim = 0;
    size_t _elems_n = 0;
    std::shared_ptr<size_t[]> _shape;
    size_t _shape_shift = 0;
    std::shared_ptr<GPUMemory> _data;
    size_t _elems_shift = 0;

    T* get_data_ptr() const { return _data->get_ptr() + _elems_shift; }

    GPUNDArray(const GPUNDArray<T>& parent, size_t slice_idx)
        : _data(parent._data), _shape(parent._shape) {
        if (parent._ndim == 0) {
            throw std::runtime_error("Attempted to slice a scalar");
        }
        _ndim = parent._ndim - 1;
        _shape_shift = parent._shape_shift + 1;
        _elems_n = parent._elems_n / parent.get_shape()[0];
        _elems_shift = parent._elems_shift + slice_idx * _elems_n;
    }

   protected:
    void* get_data_ptr_v() const { return (void*)get_data_ptr(); }

   public:
    GPUNDArray(size_t ndim, const size_t* shape, const T* host_plain_data = nullptr) {
        _ndim = ndim;
        _shape = std::make_shared<size_t[]>(ndim);
        _elems_n = 1;
        for (size_t i = 0; i < ndim; i++) {
            if (shape[i] == 0) {
                throw std::runtime_error("Empty shape is detected");
            }
            _shape[i] = shape[i];
            _elems_n *= shape[i];
        }
        _data = std::make_shared<GPUMemory>(_elems_n);
        if (host_plain_data) {
            fill(host_plain_data);
        }
    }
    void fill(const T value) {
        fill_helper_val(get_data_ptr(), _elems_n, value);
    }
    void fill(const T* host_buffer) {
        GPUMemory dev_buffer(_elems_n);
        cudaError_t rc =
            cudaMemcpy(get_data_ptr(), host_buffer, _elems_n * sizeof(T),
                       cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) {
            throw std::runtime_error(std::format("CUDA memory copy error: {}",
                                                 cudaGetErrorString(rc)));
        }
    }
    void copy_data(const GPUNDArray<T>& source) {
        if (_ndim != source._ndim || _elems_n != source._elems_n) {
            throw std::runtime_error("Elements number mismatch is detected");
        }
        cudaError_t rc =
            cudaMemcpy(get_data_ptr(), source._data.get(), _elems_n * sizeof(T),
                       cudaMemcpyDeviceToDevice);
        if (rc != cudaSuccess) {
            throw std::runtime_error(std::format("CUDA memory copy error: {}",
                                                 cudaGetErrorString(rc)));
        }
    }
    std::string print_shape() const {
        std::string shape_str("(");
        size_t* shape_arr = get_shape();
        for (int i = 0; i < _ndim; i++) {
            shape_str.append(std::to_string(shape_arr[i]));
            if (i < _ndim - 1) {
                shape_str.append(", ");
            }
        }
        shape_str.append(")");
        return shape_str;
    }

    std::string print() const {
        auto host_data = std::make_unique<T[]>(_elems_n);
        cudaError_t rc =
            cudaMemcpy(host_data.get(), get_data_ptr(), _elems_n * sizeof(T),
                       cudaMemcpyDeviceToHost);
        if (rc != cudaSuccess) {
            throw std::runtime_error(std::format("CUDA memory copy error: {}",
                                                 cudaGetErrorString(rc)));
        }
        std::string result;
        print_dim(result, _ndim, get_shape(), host_data.get());
        return result;
    }

    GPUNDArray<T> operator[](int idx) {
        size_t slice_idx = 0;
        if (idx < 0) {
            slice_idx = _shape[0] + idx;
        } else {
            slice_idx = (size_t)idx;
        }
        if (slice_idx > get_shape()[0]) {
            std::string msg = std::format("Wrong slice index {} for shape ",
                                          idx, print_shape());
            throw std::runtime_error(msg);
        }
        return GPUNDArray<T>(*this, slice_idx);
    }

    template <typename T2>
    GPUNDArray<T>& operator=(const GPUNDArray<T2>& ndarray) {
        check_shape(ndarray);
        fill_helper_buf(get_data_ptr(), _elems_n, ndarray.get_data_ptr());
        return *this;
    }

    template <typename T2>
    GPUNDArray<T>& operator+=(const GPUNDArray<T2>& ndarray) {
        check_shape(ndarray);
        add_helper_buf(get_data_ptr(), (T2*)get_other_data_ptr_v(ndarray),
                       _elems_n);
        return *this;
    }

    template <typename T2>
    GPUNDArray<T> operator+(const GPUNDArray<T2>& ndarray) {
        check_shape(ndarray);
        GPUNDArray<T> result(_ndim, get_shape(), get_data_ptr());
        add_helper_buf(result.get_data_ptr(),
                       (T2*)get_other_data_ptr_v(ndarray), _elems_n);
        return result;
    }

    template <typename T2>
    GPUNDArray<T>& operator-=(const GPUNDArray<T2>& ndarray) {
        check_shape(ndarray);
        diff_helper(get_data_ptr(), (T2*)get_other_data_ptr_v(ndarray),
                    _elems_n);
        return *this;
    }
    template <typename T2>
    GPUNDArray<T> operator-(const GPUNDArray<T2>& ndarray) {
        check_shape(ndarray);
        GPUNDArray<T> result(_ndim, get_shape(), get_data_ptr());
        diff_helper(result.get_data_ptr(), (T2*)get_other_data_ptr_v(ndarray),
                    _elems_n);
        return result;
    }
    template <typename T2>
    GPUNDArray<T>& operator=(const T2& value) {
        fill_helper_val(get_data_ptr(), _elems_n, value);
        return *this;
    }
    template <typename T2>
    GPUNDArray<T>& operator+=(const T2& value) {
        add_helper_val(get_data_ptr(), value, _elems_n);
    }

    template <typename T2>
    GPUNDArray<T> operator+(const T2& value) {
        GPUNDArray<T> result(_ndim, get_shape(), get_data_ptr());
        add_helper_val(result.get_data_ptr(), value, _elems_n);
        return result;
    }
    template <typename T2>
    GPUNDArray<T>& operator-=(const T2& value) {
        add_helper_val(get_data_ptr(), -value, _elems_n);
        return *this;
    }
    template <typename T2>
    GPUNDArray<T> operator-(const T2& value) {
        GPUNDArray<T> result(_ndim, get_shape(), get_data_ptr());
        add_helper_val(result.get_data_ptr(), -value, _elems_n);
        return result;
    }

    GPUNDArray<T> mtx_mlp(const GPUNDArray<T>& ndarray) const {
        if (_ndim != 2 || ndarray._ndim != 2) {
            throw std::runtime_error("Matrix multiplication is implemented only for 2 dim matrices");
        }
        const size_t* shape = get_shape();
        const size_t* other_shape = ndarray.get_shape();
        if (shape[1] != other_shape[0]) {
            throw std::runtime_error("Shape mismatch for matrix multiplication to be performed");
        }
        size_t res_shape[2] = {shape[0], other_shape[1]};
        GPUNDArray<T> res = GPUNDArray<T>(2, res_shape);
        mlp_helper(get_data_ptr(), ndarray.get_data_ptr(), res.get_data_ptr(), 
            shape[0], shape[1], other_shape[0], other_shape[1]);
        return res;
    }

    GPUNDArray<T> mtx_mlp_naive(const GPUNDArray<T>& ndarray) const {
        if (_ndim != 2 || ndarray._ndim != 2) {
            throw std::runtime_error("Matrix multiplication is implemented only for 2 dim matrices");
        }
        const size_t* shape = get_shape();
        const size_t* other_shape = ndarray.get_shape();
        if (shape[1] != other_shape[0]) {
            throw std::runtime_error("Shape mismatch for matrix multiplication to be performed");
        }
        size_t res_shape[2] = {shape[0], other_shape[1]};
        GPUNDArray<T> res = GPUNDArray<T>(2, res_shape);
        naive_mlp_helper(get_data_ptr(), ndarray.get_data_ptr(), res.get_data_ptr(), 
            shape[0], shape[1], other_shape[0], other_shape[1]);
        return res;
    }

    CPUNDArray<T> to_cpu() const {
        auto res = CPUNDArray<T>(_ndim, get_shape());
        cudaMemcpy(get_other_data_ptr_v(res), (void*)get_data_ptr(), _elems_n * sizeof(T), cudaMemcpyDeviceToHost);
        return res;
    }

    size_t get_ndim() const { return _ndim; }
    size_t get_elems_num() const { return _elems_n; }
    const size_t* get_shape() const { return _shape.get() + _shape_shift; }

    ~GPUNDArray() = default;
};