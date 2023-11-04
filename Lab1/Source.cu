#include <iostream>

#include "cpundarray.h"
#include "gpundarray.cuh"

void cpu_demo() {
    size_t shape1[] = {3};
    double one_dim_content[] = {-1.0, 2, 3};
    CPUNDArray<double> v(1, shape1, one_dim_content);
    std::cout << "v shape: " << v.print_shape() << "\n";
    std::cout << "v content: " << v.print() << "\n";
    std::cout << "Filling with zeros..."
              << "\n";
    v.fill(0.0);
    std::cout << "v content: " << v.print() << "\n";
    auto z = v + 100;
    std::cout << "z = v + 100 (integer): " << z.print() << "\n";
    std::cout << "v content: " << v.print() << "\n";
    auto v1 = v[1];
    std::cout << "Getting slice idx 1 from v: " << v1.print() << "\n";
    std::cout << "Changing this scalar inplace to -1...\n";
    v1 = -1;
    std::cout << "Getting slice idx -2 from v: " << v[-2].print() << "\n\n";

    double three_dim_content[] = {
        -1.8732071409224678,  -1.1089909812878078, 0.7377113781023115,
        1.4519906500436282,   0.22204955051988828, 0.6495974865035191,
        0.279975150793058,    0.4571495722742245,  0.022630420049000458,
        0.12713319667955753,  1.0078233092179332,  1.0619258206428754,
        0.3660369027791437,   -1.719545852945494,  1.5126308233647745,
        0.014623785376117361, -1.4832803104931869, -1.1415884131923562};
    size_t shape3d[] = {3, 2, 3};
    CPUNDArray<double> A(3, shape3d, three_dim_content);
    std::cout << "A shape:" << A.print_shape() << "\n";
    std::cout << "A\n" << A.print() << "\n";
    std::cout << "A[0]: \n" << A[0].print() << "\n";
    std::cout << "A[-1]: \n" << A[-1].print() << "\n";
    std::cout << "A[0, 1]: \n" << A[0][1].print() << "\n";
    std::cout << "A[0, 1] + v inplace...\n";
    A[0][1] += v;
    std::cout << "A[0]: \n" << A[0].print() << "\n";
    A = 0;
    v.fill(one_dim_content);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            A[i][j] += v + j;
        }
        v += v;
    }
    std::cout << "Modified A:\n" << A.print() << "\n";
    int B_content[] = {1, -2, -3, 0, -3, -4,  2, -4, -6,
                       1, -5, -7, 4, -8, -12, 3, -9, -13};
    CPUNDArray<int> B(3, shape3d, B_content);
    std::cout << "B (integer):\n" << B.print() << "\n";
    std::cout << "A + B: \n" << (A + B).print() << "\n";
}

void gpu_demo() {
    size_t shape1[] = {3};
    double one_dim_content[] = {-1.0, 2, 3};
    GPUNDArray<double> v(1, shape1, one_dim_content);
    std::cout << "v shape: " << v.print_shape() << "\n";
    std::cout << "v content: " << v.print() << "\n";
    std::cout << "Filling with zeros..."
              << "\n";
    v.fill(0.0);
    std::cout << "v content: " << v.print() << "\n";
    auto z = v + 100;
    std::cout << "z = v + 100 (integer): " << z.print() << "\n";
    std::cout << "v content: " << v.print() << "\n";
    auto v1 = v[1];
    std::cout << "Getting slice idx 1 from v: " << v1.print() << "\n";
    std::cout << "Changing this scalar inplace to -1...\n";
    v1 = -1;
    std::cout << "Getting slice idx -2 from v: " << v[-2].print() << "\n\n";

    double three_dim_content[] = {
        -1.8732071409224678,  -1.1089909812878078, 0.7377113781023115,
        1.4519906500436282,   0.22204955051988828, 0.6495974865035191,
        0.279975150793058,    0.4571495722742245,  0.022630420049000458,
        0.12713319667955753,  1.0078233092179332,  1.0619258206428754,
        0.3660369027791437,   -1.719545852945494,  1.5126308233647745,
        0.014623785376117361, -1.4832803104931869, -1.1415884131923562};
    size_t shape3d[] = {3, 2, 3};
    GPUNDArray<double> A(3, shape3d, three_dim_content);
    std::cout << "A shape:" << A.print_shape() << "\n";
    std::cout << "A\n" << A.print() << "\n";
    std::cout << "A[0]: \n" << A[0].print() << "\n";
    std::cout << "A[-1]: \n" << A[-1].print() << "\n";
    std::cout << "A[0, 1]: \n" << A[0][1].print() << "\n";
    std::cout << "A[0, 1] + v inplace...\n";
    A[0][1] += v;
    std::cout << "A[0]: \n" << A[0].print() << "\n";
    A = 0;
    v.fill(one_dim_content);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            A[i][j] += v + j;
        }
        v += v;
    }
    std::cout << "Modified A:\n" << A.print() << "\n";
    int B_content[] = {1, -2, -3, 0, -3, -4,  2, -4, -6,
                       1, -5, -7, 4, -8, -12, 3, -9, -13};
    GPUNDArray<int> B(3, shape3d, B_content);
    std::cout << "B (integer):\n" << B.print() << "\n";
    std::cout << "A + B: \n" << (A + B).print() << "\n";
}

int main() {
    try {
        std::cout << "--- CPU demo: ---\n";
        cpu_demo();
        std::cout << "--- GPU demo: ---\n";
        gpu_demo();
    } catch (std::exception& e) {
        std::cout << "Exception occured!\n" << e.what() << std::endl;
    }
    return 0;
}
