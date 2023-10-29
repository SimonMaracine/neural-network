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
        double rate {0.5};
        double epsilon {0.1};
        unsigned long max_epochs {100'000};
    } options;

    unsigned long epoch_index {0};
    std::size_t step_index {0};
    double epoch_error {1.0};  // Epoch error
    std::vector<double> step_errors;

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
    double calculate_epoch_error();
    void backpropagation(double* outputs, double* expected_outputs, neuron::Network<Inputs, Outputs>& network);
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
    epoch_error = 1.0;
    error_graph.clear();
    inputs = {};
    outputs = {};
    expected_outputs = {};
}

template<std::size_t Inputs, std::size_t Outputs>
bool Learn<Inputs, Outputs>::update(neuron::Network<Inputs, Outputs>& network) {
    if (epoch_index == options.max_epochs || epoch_error < options.epsilon) {
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
    step_errors.push_back(error);

    // Learn pass
    backpropagation(outputs.data(), expected_outputs.data(), network);

    // Next training set instance
    step_index++;

    if (step_index == training_set.training_instance_count) {
        // Next epoch

        epoch_error = calculate_epoch_error();
        step_errors.clear();

        error_graph.push(epoch_index, epoch_error);

        epoch_index++;
        step_index = 0;
    }

    return false;
}

template<std::size_t Inputs, std::size_t Outputs>
double Learn<Inputs, Outputs>::calculate_step_error(double* outputs, double* expected_outputs) {
    double error_sum {0.0};

    for (std::size_t i {0}; i < Outputs; i++) {
        const double error = outputs[i] - expected_outputs[i];
        error_sum += error * error;
    }

    return error_sum / 2.0;  // FIXME is it right?
}

template<std::size_t Inputs, std::size_t Outputs>
double Learn<Inputs, Outputs>::calculate_epoch_error() {
    double result_error {0.0};

    for (const double error : step_errors) {
        result_error += error;
    }

    result_error /= static_cast<double>(step_errors.size());

    return result_error;
}

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::backpropagation(double* outputs, double* expected_outputs, neuron::Network<Inputs, Outputs>& network) {
    for (std::size_t i {0}; i < Outputs; i++) {
        neuron::Neuron& neuron {network.output_layer.neurons[i]};

        const double delta {outputs[i] - expected_outputs[i] * neuron::functions::sigmoid_derivative(outputs[i])};
        auto& last_hidden_layer {network.hidden_layers[network.hidden_layers.size() - 1]};

        for (std::size_t j {0}; j < neuron.n; j++) {
            const double delta_weight {options.rate * last_hidden_layer.neurons[j].output * delta};
            neuron.weights[j] += delta_weight;  // FIXME is it right?
        }
    }

    // TODO hidden layers
}
