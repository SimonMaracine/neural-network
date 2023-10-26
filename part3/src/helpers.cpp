#include <cstddef>
#include <cstdlib>
#include <ctime>

#include "helpers.hpp"

void reallocate_double_array_random(double** array, std::size_t* old_size, std::size_t size) {
    delete[] *array;
    *array = new double[size];
    *old_size = size;

    for (std::size_t i = 0; i < size; i++) {
        (*array)[i] = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
    }
}

TrainingSet load_training_set(std::string_view file_name) {
    TrainingSet set;
    set.loaded = true;

    return set;
}
