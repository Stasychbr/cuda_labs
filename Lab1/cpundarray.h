#pragma once
#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "ndarray.h"
#include "printer.tpp"

template <typename T>
class CPUNDArray : public NDArray {
    size_t _ndim = 0;
    size_t _elems_n = 0;
    std::shared_ptr<size_t[]> _shape;
    size_t _shape_shift = 0;
    std::shared_ptr<T[]> _data;
    size_t _elems_shift = 0;

    T* get_data_ptr() const;

    CPUNDArray(const CPUNDArray<T>& parent, size_t slice_idx);

   protected:
    void* get_data_ptr_v() const { return (void*)get_data_ptr(); }
    size_t get_ndim() const { return _ndim; }
    size_t* get_shape() const;

   public:
    CPUNDArray(size_t ndim, size_t* shape, const T* plain_data = nullptr);
    void fill(const T value);
    void fill(const T* value_buffer);
    void copy_data(const CPUNDArray<T>& source);
    std::string print_shape() const;
    std::string print() const;

    CPUNDArray<T> operator[](int idx);
    template <typename T2>
    CPUNDArray<T>& operator=(const CPUNDArray<T2>& ndarray);
    template <typename T2>
    CPUNDArray<T>& operator+=(const CPUNDArray<T2>& ndarray);
    template <typename T2>
    CPUNDArray<T> operator+(const CPUNDArray<T2>& ndarray);
    template <typename T2>
    CPUNDArray<T>& operator-=(const CPUNDArray<T2>& ndarray);
    template <typename T2>
    CPUNDArray<T> operator-(const CPUNDArray<T2>& ndarray);
    template <typename T2>
    CPUNDArray<T>& operator=(const T2& value);
    template <typename T2>
    CPUNDArray<T>& operator+=(const T2& value);
    template <typename T2>
    CPUNDArray<T> operator+(const T2& value);
    template <typename T2>
    CPUNDArray<T>& operator-=(const T2& value);
    template <typename T2>
    CPUNDArray<T> operator-(const T2& value);
    ~CPUNDArray();
};

/* template class realization */

template <typename T>
CPUNDArray<T>::CPUNDArray(size_t ndim, size_t* shape, const T* plain_data) {
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
    _data = std::make_shared<T[]>(_elems_n);
    if (plain_data) {
        fill(plain_data);
    }
}

template <typename T>
CPUNDArray<T>::CPUNDArray(const CPUNDArray<T>& parent, size_t slice_idx)
    : _data(parent._data), _shape(parent._shape) {
    if (parent._ndim == 0) {
        throw std::runtime_error("Attempted to slice a scalar");
    }
    _ndim = parent._ndim - 1;
    _shape_shift = parent._shape_shift + 1;
    _elems_n = parent._elems_n / parent.get_shape()[0];
    _elems_shift = parent._elems_shift + slice_idx * _elems_n;
}

template <typename T>
T* CPUNDArray<T>::get_data_ptr() const {
    return _data.get() + _elems_shift;
}

template <typename T>
void CPUNDArray<T>::fill(const T value) {
    T* data_ptr = get_data_ptr();
    for (size_t i = 0; i < _elems_n; i++) {
        data_ptr[i] = value;
    }
}

template <typename T>
size_t* CPUNDArray<T>::get_shape() const {
    return _shape.get() + _shape_shift;
}

template <typename T>
void CPUNDArray<T>::fill(const T* value_buffer) {
    std::memmove((void*)_data.get(), (void*)value_buffer, _elems_n * sizeof(T));
}

template <typename T>
void CPUNDArray<T>::copy_data(const CPUNDArray<T>& source) {
    if (_ndim != source._ndim || _elems_n != source._elems_n) {
        throw std::runtime_error("Elements number mismatch is detected");
    }
    fill(source._data.get());
}

template <typename T>
std::string CPUNDArray<T>::print_shape() const {
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

template <typename T>
CPUNDArray<T> CPUNDArray<T>::operator[](int idx) {
    size_t slice_idx = 0;
    if (idx < 0) {
        slice_idx = _shape[0] + idx;
    } else {
        slice_idx = (size_t)idx;
    }
    if (slice_idx > get_shape()[0]) {
        std::string msg =
            std::format("Wrong slice index {} for shape ", idx, print_shape());
        throw std::runtime_error(msg);
    }
    return CPUNDArray<T>(*this, slice_idx);
}

template <typename T>
CPUNDArray<T>::~CPUNDArray() {}

template <typename T>
template <typename T2>
CPUNDArray<T>& CPUNDArray<T>::operator=(const CPUNDArray<T2>& ndarray) {
    check_shape(ndarray);
    T* data_dst = get_data_ptr();
    T2* data_src = ndarray.get_data_ptr();
    for (int i = 0; i < _elems_n; i++) {
        data_dst[i] = static_cast<T>(data_src[i]);
    }
    return *this;
}

template <typename T>
template <typename T2>
CPUNDArray<T>& CPUNDArray<T>::operator+=(const CPUNDArray<T2>& ndarray) {
    check_shape(ndarray);
    T* data_dst = get_data_ptr();
    T2* data_src = (T2*)get_other_data_ptr_v(ndarray);
    for (int i = 0; i < _elems_n; i++) {
        data_dst[i] += static_cast<T>(data_src[i]);
    }
    return *this;
}

template <typename T>
template <typename T2>
CPUNDArray<T> CPUNDArray<T>::operator+(const CPUNDArray<T2>& ndarray) {
    check_shape(ndarray);
    CPUNDArray<T> result(_ndim, get_shape());
    result = *this;
    result += ndarray;
    return result;
}

template <typename T>
template <typename T2>
CPUNDArray<T>& CPUNDArray<T>::operator-=(const CPUNDArray<T2>& ndarray) {
    check_shape(ndarray);
    T* data_dst = get_data_ptr();
    T2* data_src = ndarray.get_data_ptr();
    for (int i = 0; i < _elems_n; i++) {
        data_dst[i] -= static_cast<T>(data_src[i]);
    }
    return *this;
}

template <typename T>
template <typename T2>
CPUNDArray<T> CPUNDArray<T>::operator-(const CPUNDArray<T2>& ndarray) {
    check_shape(ndarray);
    CPUNDArray<T> result(_ndim, get_shape());
    result = *this;
    result -= ndarray;
    return result;
}

template <typename T>
template <typename T2>
CPUNDArray<T>& CPUNDArray<T>::operator=(const T2& value) {
    T casted_value = static_cast<T>(value);
    fill(casted_value);
    return *this;
}

template <typename T>
template <typename T2>
CPUNDArray<T>& CPUNDArray<T>::operator+=(const T2& value) {
    T* data_ptr = get_data_ptr();
    T casted_value = static_cast<T>(value);
    for (int i = 0; i < _elems_n; i++) {
        data_ptr[i] += casted_value;
    }
    return *this;
}

template <typename T>
template <typename T2>
CPUNDArray<T> CPUNDArray<T>::operator+(const T2& value) {
    CPUNDArray<T> res = CPUNDArray<T>(_ndim, get_shape());
    res.fill(get_data_ptr());
    res += value;
    return res;
}

template <typename T>
template <typename T2>
CPUNDArray<T>& CPUNDArray<T>::operator-=(const T2& value) {
    *this += (-value);
    return *this;
}

template <typename T>
template <typename T2>
CPUNDArray<T> CPUNDArray<T>::operator-(const T2& value) {
    return *this + (-value);
}

template <typename T>
std::string CPUNDArray<T>::print() const {
    std::string result;
    print_dim(result, _ndim, get_shape(), get_data_ptr());
    return result;
}