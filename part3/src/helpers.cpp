#include <cstddef>

#include "helpers.hpp"

void reallocate_double_array(double** array, std::size_t* old_size, std::size_t size) {
    delete[] *array;
    *array = new double[size];
    *old_size = size;

    for (std::size_t i = 0; i < size; i++) {
        (*array)[i] = 0.0;
    }
}
