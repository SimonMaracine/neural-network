#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

struct Instance {
    enum Token {
        Positive,
        Average,
        Negative,
        Bankrupt,
        NonBankrupt
    };

    union {
        struct {
            double industrial_risk;
            double management_risk;
            double financial_flexibility;
            double credibility;
            double competitiveness;
            double operating_risk;

            double classification;
        } normalized;

        struct {
            Token industrial_risk;
            Token management_risk;
            Token financial_flexibility;
            Token credibility;
            Token competitiveness;
            Token operating_risk;

            Token classification;
        } unnormalized;
    };
};

struct TrainingSet {
    std::vector<Instance> data;
    bool loaded = false;
    bool normalized = false;

    void load(std::string_view file_name);
    void shuffle();
    void normalize();
};

void reallocate_double_array_random(double** array, std::size_t* old_size, std::size_t size);
