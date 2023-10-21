#pragma once
#include <cstdlib>
#include <memory>
#include <string>

template <typename T>
class NDArray {
	size_t _ndim = 0;
	size_t _elems_n = 0;
	std::shared_ptr<size_t[]> _shape;
	size_t _shape_shift = 0;
	std::shared_ptr<T[]> _data;
	size_t _addr_shift = 0;

	T* get_data_ptr() const;
	size_t* get_shape() const;

	NDArray(const NDArray<T>& parent, size_t slice_idx);
	
public:
	NDArray(size_t ndim, size_t* shape);
	void fill(const T value);
	void fill(const T* value_buffer);
	void copy_data(const NDArray<T>& source);
	std::string print_shape() const;
	std::string print() const;
	template <typename T2>
	void check_shape(const NDArray<T2>& other) const;

	NDArray<T> operator [](int idx);
	template <typename T2>
	NDArray<T>& operator=(const NDArray<T2>& ndarray);
	template <typename T2>
	NDArray<T>& operator+=(const NDArray<T2>& ndarray);
	template <typename T2>
	NDArray<T> operator+(const NDArray<T2>& ndarray);
	template <typename T2>
	NDArray<T>& operator-=(const NDArray<T2>& ndarray);
	template <typename T2>
	NDArray<T> operator-(const NDArray<T2>& ndarray);
	template <typename T2>
	NDArray<T>& operator=(const T2& value);
	template <typename T2>
	NDArray<T>& operator+=(const T2& value);
	template <typename T2>
	NDArray<T> operator+(const T2& value);
	template <typename T2>
	NDArray<T>& operator-=(const T2& value);
	template <typename T2>
	NDArray<T> operator-(const T2& value);
	~NDArray();
};
