#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstring>
#include <cassert>
#include <utility>
#include <iterator>

#include "helpers.hpp"

static Instance::Token parse_token(char* token) {
    if (std::strcmp(token, "P") == 0) {
        return Instance::Positive;
    } else if (std::strcmp(token, "A") == 0) {
        return Instance::Average;
    } else if (std::strcmp(token, "N") == 0) {
        return Instance::Negative;
    } else if (std::strcmp(token, "B") == 0) {
        return Instance::Bankrupt;
    } else if (std::strcmp(token, "NB") == 0) {
        return Instance::NonBankrupt;
    }

    assert(false);

    return {};
}

static float normalize_token(Instance::Token token) {
    switch (token) {
        case Instance::Token::Positive:
            return 0.83333f;
        case Instance::Token::Average:
            return 0.5f;
        case Instance::Token::Negative:
            return 0.16666f;
        case Instance::Token::Bankrupt:
            return 0.25f;
        case Instance::Token::NonBankrupt:
            return 0.75f;
    }

    assert(false);

    return {};
}

void reallocate_double_array_random(double** array, std::size_t* old_size, std::size_t size) {
    delete[] *array;
    *array = new double[size];
    *old_size = size;

    for (std::size_t i {0}; i < size; i++) {
        const double normalized {static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX)};
        (*array)[i] = normalized * 2.0f - 1.0f;
    }
}

void TrainingSet::load(std::string_view file_name) {
    // TODO check that data file is valid with regex

    std::ifstream stream {std::string(file_name)};

    if (!stream.is_open()) {
        return;
    }

    while (true) {
        std::string line;

        if (!std::getline(stream, line)) {
            break;
        }

        Instance instance;

        char* string = new char[line.size() - 1];
        std::strncpy(string, line.c_str(), line.size() - 1);

        char* token {nullptr};

        token = std::strtok(string, ",");
        instance.unnormalized.industrial_risk = parse_token(token);

        token = std::strtok(nullptr, ",");
        instance.unnormalized.management_risk = parse_token(token);

        token = std::strtok(nullptr, ",");
        instance.unnormalized.financial_flexibility = parse_token(token);

        token = std::strtok(nullptr, ",");
        instance.unnormalized.credibility = parse_token(token);

        token = std::strtok(nullptr, ",");
        instance.unnormalized.competitiveness = parse_token(token);

        token = std::strtok(nullptr, ",");
        instance.unnormalized.operating_risk = parse_token(token);

        token = std::strtok(nullptr, ",");
        instance.unnormalized.classification = parse_token(token);

        delete[] string;

        data.push_back(instance);
    }

    loaded = true;
}

void TrainingSet::shuffle() {
    std::vector<Instance> new_data;
    new_data.reserve(data.size());

    while (!data.empty()) {
        const std::size_t index {static_cast<std::size_t>(std::rand() % data.size())};
        new_data.push_back(data[index]);
        data.erase(std::next(data.cbegin(), index));
    }

    data = std::move(new_data);
}

void TrainingSet::normalize() {
    for (Instance& instance : data) {
        Instance new_instance;
        new_instance.normalized.industrial_risk = normalize_token(instance.unnormalized.industrial_risk);
        new_instance.normalized.management_risk = normalize_token(instance.unnormalized.management_risk);
        new_instance.normalized.financial_flexibility = normalize_token(instance.unnormalized.financial_flexibility);
        new_instance.normalized.credibility = normalize_token(instance.unnormalized.credibility);
        new_instance.normalized.competitiveness = normalize_token(instance.unnormalized.competitiveness);
        new_instance.normalized.operating_risk = normalize_token(instance.unnormalized.operating_risk);
        new_instance.normalized.classification = normalize_token(instance.unnormalized.classification);

        instance.normalized = new_instance.normalized;
    }

    normalized = true;
}
