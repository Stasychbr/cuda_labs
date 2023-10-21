#include <iostream>
#include <stdexcept>
#include <cstring>
#include <format>
#include <string>
#include "ndarray.h"

template <typename T>
NDArray<T>::NDArray(size_t ndim, size_t* shape) {
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
}

template <typename T> 
NDArray<T>::NDArray(const NDArray<T>& parent, size_t slice_idx): _data(parent._data), _shape(parent._shape) {
	if (parent._ndim == 0) {
		throw std::runtime_error("Attempted to slice a scalar");
	}
	_ndim = parent._ndim - 1;
	_shape_shift = parent._shape_shift + 1;
	_elems_n = parent._elems_n / parent.get_shape()[0];
	_addr_shift = parent._addr_shift + slice_idx * _elems_n;
}

template<typename T>
T* NDArray<T>::get_data_ptr() const {
	return _data.get() + _addr_shift;
}

template<typename T>
void NDArray<T>::fill(const T value) {
	T* data_ptr = get_data_ptr();
	for (size_t i = 0; i < _elems_n; i++) {
		data_ptr[i] = value;
	}
}

template<typename T>
size_t* NDArray<T>::get_shape() const {
	return _shape.get() + _shape_shift;
}

template<typename T>
void NDArray<T>::fill(const T* value_buffer) {
	std::memmove((void*)_data.get(), (void*)value_buffer, _elems_n * sizeof(T));
}

template<typename T>
void NDArray<T>::copy_data(const NDArray<T>& source) {
	if (_ndim != source._ndim || _elems_n != source._elems_n) {
		throw std::runtime_error("Elements number mismatch is detected");
	}
	fill(source._data.get());
}

template<typename T>
std::string NDArray<T>::print_shape() const {
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

template<typename T>
NDArray<T> NDArray<T>::operator[](int idx) {
	size_t slice_idx = 0;
	if (idx < 0) {
		slice_idx = _shape[0] - 1 + idx;
	}
	else {
		slice_idx = (size_t)idx;
	}
	if (slice_idx < 0 || slice_idx > get_shape()[0]) {
		std::string msg = std::format("Wrong slice index %i for shape ", idx, print_shape());
		throw std::runtime_error(msg);
	}
	return NDArray<T>(*this, slice_idx);
}

template<typename T>
NDArray<T>::~NDArray() {
}


template<typename T>
template<typename T2>
void NDArray<T>::check_shape(const NDArray<T2>& other) const {
	if (_ndim != other._ndim) {
		throw std::runtime_error("Dimensions mismatch");
	}
	size_t* shape1 = get_shape();
	size_t* shape2 = other.get_shape();
	for (int i = 0; i < _ndim; i++) {
		if (shape1[i] != shape2[i]) {
			throw std::runtime_error("Shape mismatch");
		}
	}
}

template<typename T>
template<typename T2>
NDArray<T>& NDArray<T>::operator=(const NDArray<T2>& ndarray) {
	check_shape(ndarray);
	T* data_dst = get_data_ptr();
	T2* data_src = ndarray.get_data_ptr();
	for (int i = 0; i < _elems_n; i++) {
		data_dst[i] = static_cast<T>(data_src[i]);
	}
	return *this;
}

template<typename T>
template<typename T2>
NDArray<T>& NDArray<T>::operator+=(const NDArray<T2>& ndarray) {
	check_shape(ndarray);
	T* data_dst = get_data_ptr();
	T2* data_src = ndarray.get_data_ptr();
	for (int i = 0; i < _elems_n; i++) {
		data_dst[i] += static_cast<T>(data_src[i]);
	}
	return *this;
}

template<typename T>
template<typename T2>
NDArray<T> NDArray<T>::operator+(const NDArray<T2>& ndarray) {
	check_shape(ndarray);
	NDArray<T> result(_ndim, get_shape());
	result = *this;
	result += ndarray;
	return result;
}

template<typename T>
template<typename T2>
NDArray<T>& NDArray<T>::operator-=(const NDArray<T2>& ndarray) {
	check_shape(ndarray);
	T* data_dst = get_data_ptr();
	T2* data_src = ndarray.get_data_ptr();
	for (int i = 0; i < _elems_n; i++) {
		data_dst[i] -= static_cast<T>(data_src[i]);
	}
	return *this;
}

template<typename T>
template<typename T2>
NDArray<T> NDArray<T>::operator-(const NDArray<T2>& ndarray) {
	check_shape(ndarray);
	NDArray<T> result(_ndim, get_shape());
	result = *this;
	result -= ndarray;
	return result;
}

template<typename T>
template<typename T2>
NDArray<T>& NDArray<T>::operator=(const T2& value) {
	T casted_value = static_cast<T>(value);
	fill(casted_value);
	return *this;
}

template<typename T>
template<typename T2>
NDArray<T>& NDArray<T>::operator+=(const T2& value) {
	T* data_ptr = get_data_ptr();
	T casted_value = static_cast<T>(value);
	for (int i = 0; i < _elems_n; i++) {
		data_ptr[i] += casted_value;
	}
	return *this;
}

template<typename T>
template<typename T2>
NDArray<T> NDArray<T>::operator+(const T2& value) {
	NDArray<T> res = NDArray<T>(_ndim, get_shape());
	res.fill(get_data_ptr());
	res += value;
	return res;
}

template<typename T>
template<typename T2>
NDArray<T>& NDArray<T>::operator-=(const T2& value) {
	*this += (-value);
	return *this;
}

template<typename T>
template<typename T2>
NDArray<T> NDArray<T>::operator-(const T2& value) {
	return *this + (-value);
}

template<typename T>
void print_dim(std::string& res, size_t ndim, size_t* shape, T* data_ptr, size_t alignment=1, bool f_last=true) {
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
		print_dim(res, ndim - 1, shape + 1, data_ptr + i * sub_shape_prod, alignment + 1, i == shape[0] - 1);
		if (i < shape[0] - 1) {
			res.append("\n");
		}
	}
	res.append("]");
	if (!f_last) {
		res.append("\n");
	}
}

template<typename T>
std::string NDArray<T>::print() const {
	std::string result;
	print_dim(result, _ndim, get_shape(), get_data_ptr());
	return result;
}

int main() {
	size_t shape1[] = { 3 };
	double one_dim_content[] = { -1.0, 2, 3 };
	NDArray<double> one_dim(1, shape1);
	one_dim.fill(0.0);
	std::cout << "Shape: " << one_dim.print_shape() << "\n";
	std::cout << "Zeros: " << one_dim.print() << "\n";
	one_dim.fill(one_dim_content);
	std::cout << "Content: " << one_dim.print() << "\n";
	auto another_mtx = one_dim + 100;
	std::cout << "Content + 100: " << another_mtx.print() << "\n";
	auto scalar = another_mtx[0];
	std::cout << "Content + 100 first elem: " << scalar.print() << "\n";
	scalar -= 5;
	std::cout << "Content + 100 first elem - 5: " << scalar.print() << "\n\n";
	double three_dim_content[] = { -1.8732071409224678 , -1.1089909812878078 , 0.7377113781023115 , 
		1.4519906500436282 , 0.22204955051988828 , 0.6495974865035191 , 0.279975150793058 , 0.4571495722742245 , 
		0.022630420049000458 , 0.12713319667955753 , 1.0078233092179332 , 1.0619258206428754 , 0.3660369027791437 , 
		-1.719545852945494 , 1.5126308233647745 , 0.014623785376117361 , -1.4832803104931869 , -1.1415884131923562 };
	size_t shape3d[] = { 3, 2, 3 };
	NDArray<double> mtx(3, shape3d);
	mtx.fill(three_dim_content);
	std::cout << "3d shape:" << mtx.print_shape() << "\n";
	std::cout << "A\n" << mtx.print() << "\n";
	std::cout << "A[0]: \n" << mtx[0].print() << "\n";
	std::cout << "A[2]: \n" << mtx[2].print() << "\n";
	std::cout << "A[0, 1]: \n" << mtx[0][1].print() << "\n";
	mtx[0][1] += one_dim;
	std::cout << "A[0, 1] + content above (inplace): " << mtx[0][1].print() << "\n";
	std::cout << "A: \n" << mtx.print() << "\n";
	return 0;
}
