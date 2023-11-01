#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstring>
#include <cassert>
#include <utility>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <optional>
#include <memory>

#include "helpers.hpp"

static std::optional<double> parse_classification(const char* string) {
    if (std::strcmp(string, "alive") == 0) {
        return std::make_optional(1.0);
    } else if (std::strcmp(string, "failed") == 0) {
        return std::make_optional(0.0);
    }

    return std::nullopt;
}

static std::optional<double> parse_number(const char* string) {
    double result;

    try {
        result = std::stod(string);
    } catch (const std::invalid_argument& e) {
        return std::nullopt;
    } catch (const std::out_of_range& e) {
        return std::nullopt;
    }

    return std::make_optional(result);
}

static constexpr double map(double x, double in_min, double in_max, double out_min, double out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

static bool check_line(const std::string& line) {
    return line.find("company_name,status_label,year,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18");
}

bool TrainingSet::load(std::string_view file_name, float percent_for_testing) {
    std::ifstream stream {std::string(file_name)};

    if (!stream.is_open()) {
        return false;
    }

    data.clear();

    std::string line;

    // Consume first line
    if (!std::getline(stream, line)) {
        return false;
    }

    if (!check_line(line)) {
        loaded = false;

        return false;
    }

    while (true) {
        if (!std::getline(stream, line)) {
            break;
        }

        Instance instance;

        if (line[line.size() - 1] == '\r') {
            line.pop_back();
        }

        char* token {nullptr};
        std::optional<double> value;

        token = std::strtok(line.data(), ",");
        // Skip name

        token = std::strtok(nullptr, ",");
        value = parse_classification(token);
        if (value) {
            instance.classification = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        // Skip year

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.current_assets = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.cost_of_goods_sold = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.depreciation_and_amortization = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.financial_performance = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.inventory = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.net_income = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.total_receivables = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.market_value = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.net_sales = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.total_assets = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.total_long_term_debt = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.earnings_before_interest_and_taxes = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.gross_profit = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.total_current_liabilities = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.retained_earnings = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.total_revenue = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.total_liabilities = *value;
        } else {
            return false;
        }

        token = std::strtok(nullptr, ",");
        value = parse_number(token);
        if (value) {
            instance.total_operating_expenses = *value;
        } else {
            return false;
        }

        data.push_back(instance);
    }

    loaded = true;

    set_testing(percent_for_testing);

    return true;
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
    if (normalized) {
        return;
    }

    for (Instance& instance : data) {
        normalize_instance(instance);
    }

    normalized = true;
}

void TrainingSet::set_testing(float percent_for_testing) {
    assert(percent_for_testing > 0.0f && percent_for_testing < 100.0f);

    const double instances_for_testing {(static_cast<double>(data.size()) * percent_for_testing) / 100.0f};
    training_instance_count = data.size() - static_cast<std::size_t>(instances_for_testing);
}

void normalize_instance(Instance& instance) {
    instance.current_assets =                     map(instance.current_assets, -7.76, 170'000.0, 0.0, 1.0);
    instance.cost_of_goods_sold =                 map(instance.cost_of_goods_sold, -367.0, 375'000.0, 0.0, 1.0);
    instance.depreciation_and_amortization =      map(instance.depreciation_and_amortization, 0.0, 28'400.0, 0.0, 1.0);
    instance.financial_performance =              map(instance.financial_performance, -21'900.0, 81'700.0, 0.0, 1.0);
    instance.inventory =                          map(instance.inventory, 0.0, 62'600.0, 0.0, 1.0);
    instance.net_income =                         map(instance.net_income, -98'700.0, 105'000.0, 0.0, 1.0);
    instance.total_receivables =                  map(instance.total_receivables, -0.01, 65'800.0, 0.0, 1.0);
    instance.market_value =                       map(instance.market_value, 0.0, 1'070'000.0, 0.0, 1.0);
    instance.net_sales =                          map(instance.net_sales, -1'960.0, 512'000.0, 0.0, 1.0);
    instance.total_assets =                       map(instance.total_assets, 0.0, 532'000.0, 0.0, 1.0);
    instance.total_long_term_debt =               map(instance.total_long_term_debt, -0.02, 166'000.0, 0.0, 1.0);
    instance.earnings_before_interest_and_taxes = map(instance.earnings_before_interest_and_taxes, -25'900.0, 71'200.0, 0.0, 1.0);
    instance.gross_profit =                       map(instance.gross_profit, -21'500.0, 137'000.0, 0.0, 1.0);
    instance.total_current_liabilities =          map(instance.total_current_liabilities, 0.0, 117'000.0, 0.0, 1.0);
    instance.retained_earnings =                  map(instance.retained_earnings, -102'000.0, 402'000.0, 0.0, 1.0);
    instance.total_revenue =                      map(instance.total_revenue, -1'960.0, 512'000.0, 0.0, 1.0);
    instance.total_liabilities =                  map(instance.total_liabilities, 0.0, 338'000.0, 0.0, 1.0);
    instance.total_operating_expenses =           map(instance.total_operating_expenses, -317.0, 482'000.0, 0.0, 1.0);
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
