#pragma once
#include <string>

template <typename T>
void print_dim(std::string& res, size_t ndim, const size_t* shape, const T* data_ptr,
               size_t alignment = 1, bool f_last = true) {
    if (ndim == 0) {
        res.append(std::to_string(*data_ptr));
        return;
    }
    if (ndim == 1) {
        res.append("[ ");
        for (int i = 0; i < *shape; i++) {
            res.append(std::to_string(data_ptr[i]));
            res.append(" ");
        }
        res.append("]");
        return;
    }
    res.append("[");
    size_t sub_shape_prod = 1;
    for (size_t i = ndim - 1; i > 0; i--) {
        sub_shape_prod *= shape[i];
    }
    for (int i = 0; i < shape[0]; i++) {
        if (i != 0) {
            for (int j = 0; j < alignment; j++) {
                res.append(" ");
            }
        }
        print_dim(res, ndim - 1, shape + 1, data_ptr + i * sub_shape_prod,
                  alignment + 1, i == shape[0] - 1);
        if (i < shape[0] - 1) {
            res.append("\n");
        }
    }
    res.append("]");
    if (!f_last) {
        res.append("\n");
    }
}