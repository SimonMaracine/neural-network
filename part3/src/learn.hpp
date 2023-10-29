#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include <thread>

#include "network.hpp"
#include "helpers.hpp"

struct ErrorGraph {
    void push(std::size_t index, double error) {
        errors.push_back(error);
        indices.push_back(static_cast<double>(index));
    }

    void clear() {
        errors.clear();
        indices.clear();
    }

    std::vector<double> errors;
    std::vector<double> indices;
};

template<std::size_t Inputs, std::size_t Outputs>
class Learn {
public:
    struct Options {
        double rate {1.0};
        double epsilon {0.1};
        unsigned long max_epochs {100'000};
    } options;

    unsigned long epoch_index {0};
    std::size_t step_index {0};
    double current_error {1.0};  // Epoch error

    ErrorGraph error_graph;

    TrainingSet training_set;

    void start(neuron::Network<Inputs, Outputs>& network);
    void stop();
    void reset();
    bool is_running() const { return running; }
private:
    bool running = false;

    std::array<double, Inputs> inputs {};
    std::array<double, Outputs> outputs {};
    std::array<double, Outputs> expected_outputs {};

    std::thread thread;

    bool update(neuron::Network<Inputs, Outputs>& network);  // Return true when it should stop
    double calculate_step_error(double* outputs, double* expected_outputs);
};

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::start(neuron::Network<Inputs, Outputs>& network) {
    thread = std::thread([this, &network]() {
        running = true;

        while (running) {
            if (update(network)) {
                break;
            }
        }

        running = false;
    });
}

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::stop() {
    running = false;
}

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::reset() {
    running = false;

    if (thread.joinable()) {
        thread.join();
    }

    options = Options();
    epoch_index = 0;
    step_index = 0;
    current_error = 1.0;
    error_graph.clear();
    inputs = {};
    outputs = {};
    expected_outputs = {};
}

template<std::size_t Inputs, std::size_t Outputs>
bool Learn<Inputs, Outputs>::update(neuron::Network<Inputs, Outputs>& network) {
    if (epoch_index == options.max_epochs || current_error < options.epsilon) {
        return true;
    }

    // Retreive training set instance
    const auto& instance = training_set.data[step_index];

    // Setup inputs and expected outputs
    inputs[0] = instance.normalized.industrial_risk;
    inputs[1] = instance.normalized.management_risk;
    inputs[2] = instance.normalized.financial_flexibility;
    inputs[3] = instance.normalized.credibility;
    inputs[4] = instance.normalized.competitiveness;
    inputs[5] = instance.normalized.operating_risk;
    expected_outputs[0] = instance.normalized.classification;

    // Forward pass
    network.run(inputs.data(), outputs.data());

    // Calculate error
    const double error = calculate_step_error(outputs.data(), expected_outputs.data());

    // TODO compute error and do backpropagation; return true when it should stop





    // Next training set instance
    step_index++;

    if (step_index == training_set.data.size()) {
        // Next epoch

        // TODO calculate epoch error
        const double e = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);

        error_graph.push(epoch_index, e);

        epoch_index++;
        step_index = 0;
    }

    return false;
}

template<std::size_t Inputs, std::size_t Outputs>
double Learn<Inputs, Outputs>::calculate_step_error(double* outputs, double* expected_outputs) {
    return {};
}
