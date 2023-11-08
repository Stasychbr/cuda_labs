#include <Windows.h>
#include <time.h>

#include <cstdlib>
#include <iostream>
#include <memory>

#include "cpundarray.h"
#include "gpundarray.cuh"

template <typename T>
void fill_random_array(T* buffer, size_t elem_num) {
    std::unique_ptr<T[]> res = std::make_unique<T[]>(elem_num);
    const T a = -2;
    const T b = 2;
    for (size_t i = 0; i < elem_num; i++) {
        res[i] = a + ((double)std::rand() / RAND_MAX) * (b - a);
    }
}

template <typename T>
void check_error(const CPUNDArray<T>& A, const CPUNDArray<T>& B) {
    const T epsilon = std::numeric_limits<T>::epsilon();
    auto arr_a = A.get_raw_data();
    auto arr_b = B.get_raw_data();
    size_t counter = 0;
    for (size_t i = 0; i < A.get_elems_num(); i++) {
        if (abs(arr_a[i] - arr_b[i]) > epsilon) {
            counter++;
        }
    }
    if (counter > 0) {
        std::cout << "WARNING: in " << counter
                  << " elements an error more than " << epsilon
                  << " was found\n";
    }
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

constexpr size_t A_SHAPE[] = {1000, 2000};
constexpr size_t B_SHAPE[] = {2000, 1000};
constexpr size_t ITER_N = 10;
constexpr size_t PROG_BAR_SIZE = 50;

int main() {
    std::srand((unsigned int)(clock() % 2048));
    std::cout << "--- GPU/CPU matrix multiplication test ---\n";
    LARGE_INTEGER time_stamp[2], time_cpu = {0}, time_gpu = {0};
    LARGE_INTEGER frequency = {0};
    ZeroMemory(time_stamp, sizeof(time_stamp));
    try {
        if (!QueryPerformanceFrequency(&frequency)) {
            throw std::runtime_error("Performance counter error");
        }
        const size_t a_size = A_SHAPE[0] * A_SHAPE[1];
        const size_t b_size = B_SHAPE[0] * B_SHAPE[1];
        auto a_buffer = std::make_unique<float[]>(a_size);
        auto b_buffer = std::make_unique<float[]>(b_size);
        for (size_t i = 0; i < ITER_N; i++) {
            fill_random_array(a_buffer.get(), a_size);
            fill_random_array(a_buffer.get(), b_size);
            CPUNDArray<float> A_cpu(2, A_SHAPE, a_buffer.get());
            CPUNDArray<float> B_cpu(2, B_SHAPE, b_buffer.get());
            QueryPerformanceCounter(time_stamp);
            auto C_cpu = A_cpu.mtx_mlp(B_cpu);
            QueryPerformanceCounter(time_stamp + 1);
            time_cpu.QuadPart +=
                time_stamp[1].QuadPart - time_stamp[0].QuadPart;
            GPUNDArray<float> A_gpu(2, A_SHAPE, a_buffer.get());
            GPUNDArray<float> B_gpu(2, B_SHAPE, b_buffer.get());
            QueryPerformanceCounter(time_stamp);
            auto C_gpu = A_gpu.mtx_mlp(B_gpu);
            QueryPerformanceCounter(time_stamp + 1);
            time_gpu.QuadPart +=
                time_stamp[1].QuadPart - time_stamp[0].QuadPart;
            check_error(C_cpu, C_gpu.to_cpu());
            draw_prog_bar(i, ITER_N, PROG_BAR_SIZE);
        }
        draw_prog_bar(ITER_N, ITER_N, PROG_BAR_SIZE);
        std::cout << "\nTest complete!\n";
        std::cout << "Avg. CPU time: "
                  << ((long double)time_cpu.QuadPart /
                      (frequency.QuadPart * ITER_N)) * 1000.0
                  << " ms.\n";
        std::cout << "Avg. GPU time: "
                  << ((long double)time_gpu.QuadPart /
                      (frequency.QuadPart * ITER_N)) * 1000.0
                  << " ms. \n";
    } catch (std::exception& e) {
        std::cout << "An exception occured during the test: " << e.what() << std::endl;
    }
    return 0;
}
