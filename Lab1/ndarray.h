#pragma once

class NDArray {
   protected:
    virtual void* get_data_ptr_v() const = 0;
    virtual size_t* get_shape() const = 0;
    virtual size_t get_ndim() const = 0;

    // C++ is awesome!
    void* get_other_data_ptr_v(const NDArray& array) const {
        return array.get_data_ptr_v();
    }

    void check_shape(const NDArray& other) const {
        size_t ndim = get_ndim();
        if (ndim != other.get_ndim()) {
            throw std::runtime_error("Dimensions mismatch");
        }
        size_t* shape1 = get_shape();
        size_t* shape2 = other.get_shape();
        for (int i = 0; i < ndim; i++) {
            if (shape1[i] != shape2[i]) {
                throw std::runtime_error("Shape mismatch");
            }
        }
    }
};
