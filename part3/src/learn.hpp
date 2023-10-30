#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include <thread>
#include <iterator>
#include <cmath>

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
    struct {
        double learning_rate {0.05};
        double epsilon {0.01};
        unsigned long max_epochs {100'000};
    } options;

    unsigned long epoch_index {0};
    std::size_t step_index {0};
    double epoch_error {1.0};
    std::vector<double> step_errors;

    ErrorGraph error_graph;

    TrainingSet training_set;

    void start_learning(neuron::Network<Inputs, Outputs>& network);
    void stop_learning();
    void reset();
    double test(const neuron::Network<Inputs, Outputs>& network) const;
    bool is_running() const { return running; }
private:
    bool running = false;

    mutable std::array<double, Inputs> inputs {};
    mutable std::array<double, Outputs> outputs {};
    mutable std::array<double, Outputs> expected_outputs {};

    std::thread thread;

    // Return true when it should stop
    bool update(neuron::Network<Inputs, Outputs>& network);
    static double calculate_step_error(double* outputs, double* expected_outputs);
    static double calculate_error_testing(double* outputs, double* expected_outputs);
    double calculate_epoch_error();
    void backpropagation(double* outputs, double* expected_outputs, neuron::Network<Inputs, Outputs>& network);
};

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::start_learning(neuron::Network<Inputs, Outputs>& network) {
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
void Learn<Inputs, Outputs>::stop_learning() {
    running = false;

    if (thread.joinable()) {
        thread.join();
    }
}

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::reset() {
    stop_learning();

    epoch_index = 0;
    step_index = 0;
    epoch_error = 1.0;
    error_graph.clear();
    inputs = {};
    outputs = {};
    expected_outputs = {};
}

template<std::size_t Inputs, std::size_t Outputs>
double Learn<Inputs, Outputs>::test(const neuron::Network<Inputs, Outputs>& network) const {
    const double threshold = 0.01;

    std::size_t passed {0};

    for (std::size_t i {training_set.training_instance_count}; i < training_set.data.size(); i++) {
        const auto& instance = training_set.data[i];

        inputs[0] = instance.normalized.industrial_risk;
        inputs[1] = instance.normalized.management_risk;
        inputs[2] = instance.normalized.financial_flexibility;
        inputs[3] = instance.normalized.credibility;
        inputs[4] = instance.normalized.competitiveness;
        inputs[5] = instance.normalized.operating_risk;
        expected_outputs[0] = instance.normalized.classification;

        network.run(inputs.data(), outputs.data());

        // Do the test
        const double error = calculate_error_testing(outputs.data(), expected_outputs.data());

        if (error < threshold) {
            passed++;
        }
    }

    const std::size_t testing_instance_count {training_set.data.size() - training_set.training_instance_count};

    return static_cast<double>(passed) / static_cast<double>(testing_instance_count) * 100.0;
}

template<std::size_t Inputs, std::size_t Outputs>
bool Learn<Inputs, Outputs>::update(neuron::Network<Inputs, Outputs>& network) {
    if (epoch_index == options.max_epochs || epoch_error < options.epsilon) {
        return true;
    }

    // Retrieve training set instance
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

    // Learning pass
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
double Learn<Inputs, Outputs>::calculate_error_testing(double* outputs, double* expected_outputs) {  // FIXME wrong
    double error_sum {0.0};

    for (std::size_t i {0}; i < Outputs; i++) {
        const double error = std::abs(outputs[i] - expected_outputs[i]);
        error_sum += error;
    }

    return error_sum / Outputs;
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
    // Output layer
    for (std::size_t i {0}; i < Outputs; i++) {
        neuron::Neuron& neuron {network.output_layer.neurons[i]};

        const double layer_error {outputs[i] - expected_outputs[i]};

        neuron.delta = layer_error * neuron::functions::sigmoid_derivative(outputs[i]);

        for (std::size_t j {0}; j < neuron.n; j++) {
            auto& last_hidden_layer {network.hidden_layers[network.hidden_layers.size() - 1]};

            const double change {options.learning_rate * last_hidden_layer.neurons[j].output * neuron.delta};
            neuron.weights[j] -= change;
        }
    }

    // Hidden layers
    for (auto iter = network.hidden_layers.rbegin(); iter != network.hidden_layers.rend(); iter++) {
        const bool is_last_hidden_layer {iter == network.hidden_layers.rbegin()};
        const bool is_first_hidden_layer {iter == std::prev(network.hidden_layers.rend())};

        for (std::size_t i {0}; i < iter->neurons.size(); i++) {
            neuron::Neuron& neuron {iter->neurons[i]};

            double layer_error {0.0};

            if (is_last_hidden_layer) {
                for (auto iter2 = network.output_layer.neurons.begin(); iter2 != network.output_layer.neurons.end(); iter2++) {
                    neuron::Neuron& neuron {*iter2};

                    layer_error += neuron.weights[i] * neuron.delta;
                }
            } else {
                for (auto iter2 = std::prev(iter)->neurons.begin(); iter2 != std::prev(iter)->neurons.end(); iter2++) {
                    neuron::Neuron& neuron {*iter2};

                    layer_error += neuron.weights[i] * neuron.delta;
                }
            }

            neuron.delta = layer_error * neuron::functions::tanh_derivative(neuron.output);

            if (is_first_hidden_layer) {
                for (std::size_t j {0}; j < neuron.n; j++) {
                    const double change {options.learning_rate * inputs[j] * neuron.delta};
                    neuron.weights[j] -= change;
                }
            } else {
                for (std::size_t j {0}; j < neuron.n; j++) {
                    const double change {options.learning_rate * std::next(iter)->neurons[j].output * neuron.delta};
                    neuron.weights[j] -= change;
                }
            }
        }
    }
}
