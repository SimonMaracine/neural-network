#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

struct Instance {
    double current_assets;
    double cost_of_goods_sold;
    double depreciation_and_amortization;
    double financial_performance;
    double inventory;
    double net_income;
    double total_receivables;
    double market_value;
    double net_sales;
    double total_assets;
    double total_long_term_debt;
    double earnings_before_interest_and_taxes;
    double gross_profit;
    double total_current_liabilities;
    double retained_earnings;
    double total_revenue;
    double total_liabilities;
    double total_operating_expenses;

    double classification;
};

struct TrainingSet {
    std::vector<Instance> data;
    bool loaded = false;
    bool normalized = false;
    std::size_t training_instance_count {0};

    bool load(std::string_view file_name, float percent_for_testing);
    void shuffle();
    void normalize();
    void set_testing(float percent_for_testing);
};

void normalize_instance(Instance& instance);
void reallocate_double_array_random(double** array, std::size_t* old_size, std::size_t size);
