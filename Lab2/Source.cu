#include <Windows.h>
#include <time.h>

#include <cstdlib>
#include <iostream>
#include <memory>

#include "cpundarray.h"
#include "gpundarray.cuh"

template <typename T>
void fill_random_array(T* buffer, size_t elem_num) {
    const T a = -2;
    const T b = 2;
    for (size_t i = 0; i < elem_num; i++) {
        buffer[i] = a + ((double)std::rand() / RAND_MAX) * (b - a);
    }
}

template <typename T>
T check_error(const CPUNDArray<T>& A, const CPUNDArray<T>& B) {
    const T epsilon = 10 * std::numeric_limits<T>::epsilon();
    auto arr_a = A.get_raw_data();
    auto arr_b = B.get_raw_data();
    size_t counter = 0;
    T total_error = 0;
    for (size_t i = 0; i < A.get_elems_num(); i++) {
        T error = abs(arr_a[i] - arr_b[i]);
        if (error > epsilon) {
            counter++;
        }
        total_error += error;
    }
    if (counter > 0) {
        std::cout << "WARNING: in " << counter
                  << " elements an error more than " << epsilon
                  << " was found\n";
    }
    return total_error / A.get_elems_num();
}

void draw_prog_bar(size_t cur_iter, size_t all_iters, size_t width) {
    double cur_progress = (double)cur_iter / all_iters;
    std::cout << "[";
    for (size_t i = 0; i < width; i++) {
        double norm_width = (double)i / width;
        std::cout << (cur_progress <= norm_width ? " " : "=");
    }
    std::cout << "]";
    for (size_t i = 0; i < width + 2; i++) {
        std::cout << '\b';
    }
    std::cout << std::flush;
}

constexpr size_t n = 1000;
constexpr size_t A_SHAPE[] = {n, n};
constexpr size_t B_SHAPE[] = {n, n};
constexpr size_t ITER_N = 100;
constexpr size_t PROG_BAR_SIZE = 50;
#define TYPE float

int main() {
    std::srand((unsigned int)(time(nullptr) % 2048));
    std::cout << "--- GPU/CPU matrix multiplication test ---\n";
    int64_t time_cpu = 0, time_gpu = 0, time_gpu_naive = 0;
    LARGE_INTEGER frequency = {0}, time_stamp[2];
    try {
        if (!QueryPerformanceFrequency(&frequency)) {
            throw std::runtime_error("Performance counter error");
        }
        const size_t a_size = A_SHAPE[0] * A_SHAPE[1];
        const size_t b_size = B_SHAPE[0] * B_SHAPE[1];
        auto a_buffer = std::make_unique<TYPE[]>(a_size);
        auto b_buffer = std::make_unique<TYPE[]>(b_size);
        TYPE error_sum = 0;
        TYPE error_sum_naive = 0;
        for (size_t i = 0; i < ITER_N; i++) {
            fill_random_array(a_buffer.get(), a_size);
            fill_random_array(b_buffer.get(), b_size);
            CPUNDArray<TYPE> A_cpu(2, A_SHAPE, a_buffer.get());
            CPUNDArray<TYPE> B_cpu(2, B_SHAPE, b_buffer.get());
            ZeroMemory(time_stamp, sizeof(time_stamp));
            QueryPerformanceCounter(time_stamp);
            auto C_cpu = A_cpu.mtx_mlp(B_cpu);
            QueryPerformanceCounter(time_stamp + 1);
            time_cpu += time_stamp[1].QuadPart - time_stamp[0].QuadPart;
            
            GPUNDArray<TYPE> A_gpu(2, A_SHAPE, a_buffer.get());
            GPUNDArray<TYPE> B_gpu(2, B_SHAPE, b_buffer.get());
            ZeroMemory(time_stamp, sizeof(time_stamp));
            QueryPerformanceCounter(time_stamp);
            auto C_gpu_naive = A_gpu.mtx_mlp_naive(B_gpu);
            QueryPerformanceCounter(time_stamp + 1);
            time_gpu_naive +=
                time_stamp[1].QuadPart - time_stamp[0].QuadPart;
            error_sum_naive += check_error(C_cpu, C_gpu_naive.to_cpu());
            
            ZeroMemory(time_stamp, sizeof(time_stamp));
            QueryPerformanceCounter(time_stamp);
            auto C_gpu = A_gpu.mtx_mlp(B_gpu);
            QueryPerformanceCounter(time_stamp + 1);
            time_gpu += time_stamp[1].QuadPart - time_stamp[0].QuadPart;
            error_sum += check_error(C_cpu, C_gpu.to_cpu());
            draw_prog_bar(i, ITER_N, PROG_BAR_SIZE);
        }
        draw_prog_bar(ITER_N, ITER_N, PROG_BAR_SIZE);
        std::cout << "\nTest complete!\n";
        std::cout << "Avg. CPU time: "
                  << ((long double)time_cpu /
                      (frequency.QuadPart)) * 1000.0 / ITER_N
                  << " ms.\n";
        std::cout << "Avg. GPU time: "
                  << ((long double)time_gpu /
                      (frequency.QuadPart)) * 1000.0 / ITER_N
                  << " ms. \n";
        std::cout << "Avg. GPU naive time: "
                  << ((long double)time_gpu_naive /
                      (frequency.QuadPart)) * 1000.0 / ITER_N
                  << " ms. \n";
        std::cout << "Avg. computational difference: " << error_sum / ITER_N << "\n";
        std::cout << "Avg. naive computational difference: " << error_sum_naive / ITER_N << "\n";
    } catch (std::exception& e) {
        std::cout << "An exception occured during the test: " << e.what() << std::endl;
    }
    return 0;
}
