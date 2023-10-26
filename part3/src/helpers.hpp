#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

struct TrainingSet {
    std::vector<double*> data;
    unsigned int inputs {};
    unsigned int outputs {};
    bool loaded = false;
};

void reallocate_double_array_random(double** array, std::size_t* old_size, std::size_t size);
TrainingSet load_training_set(std::string_view file_name);
