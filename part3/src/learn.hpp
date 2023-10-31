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
    void push_back(std::size_t index, double error) {
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

struct Test {
    Instance instance;
    double output {0.0};
    bool passed {false};
};

template<std::size_t Inputs, std::size_t Outputs>
class Learn {
public:
    struct {
        double learning_rate {0.05};
        double epsilon {0.01};
        unsigned long max_epochs {100'000};
    } options;

    struct {
        unsigned long epoch_index {0};
        std::size_t step_index {0};
        double epoch_error {1.0};

        std::vector<double> step_errors;
        ErrorGraph error_graph;
    } learning;

    mutable struct {
        std::vector<Test> tests;
    } testing;

    TrainingSet training_set;

    void start(network::Network<Inputs, Outputs>& network);
    void stop();
    void reset();
    double test(const network::Network<Inputs, Outputs>& network) const;
    bool is_running() const { return running; }
private:
    mutable struct {
        std::array<double, Inputs> inputs {};
        std::array<double, Outputs> outputs {};
        std::array<double, Outputs> expected_outputs {};
    } data;

    std::thread thread;
    bool running = false;

    // Return true when it should stop
    bool update(network::Network<Inputs, Outputs>& network);
    static double calculate_step_error(double* outputs, double* expected_outputs);
    static double calculate_error_testing(double* outputs, double* expected_outputs);
    static double calculate_epoch_error(const std::vector<double>& step_errors);
    void backpropagation(double* outputs, double* expected_outputs, network::Network<Inputs, Outputs>& network) const;
};

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::start(network::Network<Inputs, Outputs>& network) {
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

    if (thread.joinable()) {
        thread.join();
    }
}

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::reset() {
    stop();

    learning.epoch_index = 0;
    learning.step_index = 0;
    learning.epoch_error = 1.0;
    learning.step_errors.clear();
    learning.error_graph.clear();
}

template<std::size_t Inputs, std::size_t Outputs>
double Learn<Inputs, Outputs>::test(const network::Network<Inputs, Outputs>& network) const {
    testing.tests.clear();

    std::size_t passed {0};

    for (std::size_t i {training_set.training_instance_count}; i < training_set.data.size(); i++) {
        const auto& instance = training_set.data[i];

        data.inputs[0] = instance.normalized.industrial_risk;
        data.inputs[1] = instance.normalized.management_risk;
        data.inputs[2] = instance.normalized.financial_flexibility;
        data.inputs[3] = instance.normalized.credibility;
        data.inputs[4] = instance.normalized.competitiveness;
        data.inputs[5] = instance.normalized.operating_risk;
        data.expected_outputs[0] = instance.normalized.classification;

        network.run(data.inputs.data(), data.outputs.data());

        // The error for this specific network is either 0 or 1
        const double error = calculate_error_testing(data.outputs.data(), data.expected_outputs.data());

        Test test;
        test.instance = instance;
        test.output = data.outputs[0];

        if (error == 0.0) {
            passed++;
            test.passed = true;
        }

        testing.tests.push_back(test);
    }

    const std::size_t testing_instance_count {training_set.data.size() - training_set.training_instance_count};

    return static_cast<double>(passed) / static_cast<double>(testing_instance_count) * 100.0;
}

template<std::size_t Inputs, std::size_t Outputs>
bool Learn<Inputs, Outputs>::update(network::Network<Inputs, Outputs>& network) {
    if (learning.epoch_index == options.max_epochs || learning.epoch_error < options.epsilon) {
        return true;
    }

    // Retrieve training set instance
    const auto& instance = training_set.data[learning.step_index];

    // Setup inputs and expected outputs
    data.inputs[0] = instance.normalized.industrial_risk;
    data.inputs[1] = instance.normalized.management_risk;
    data.inputs[2] = instance.normalized.financial_flexibility;
    data.inputs[3] = instance.normalized.credibility;
    data.inputs[4] = instance.normalized.competitiveness;
    data.inputs[5] = instance.normalized.operating_risk;
    data.expected_outputs[0] = instance.normalized.classification;

    // Forward pass
    network.run(data.inputs.data(), data.outputs.data());

    // Calculate error
    const double error = calculate_step_error(data.outputs.data(), data.expected_outputs.data());
    learning.step_errors.push_back(error);

    // Learning pass
    backpropagation(data.outputs.data(), data.expected_outputs.data(), network);

    // Next training set instance
    learning.step_index++;

    if (learning.step_index == training_set.training_instance_count) {
        // Next epoch

        learning.epoch_error = calculate_epoch_error(learning.step_errors);
        learning.step_errors.clear();

        learning.error_graph.push_back(learning.epoch_index, learning.epoch_error);

        learning.epoch_index++;
        learning.step_index = 0;
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
double Learn<Inputs, Outputs>::calculate_error_testing(double* outputs, double* expected_outputs) {
    double error_sum {0.0};

    for (std::size_t i {0}; i < Outputs; i++) {
        const double error = std::abs(network::functions::binary(outputs[i]) - expected_outputs[i]);
        error_sum += error;
    }

    return error_sum / Outputs;
}

template<std::size_t Inputs, std::size_t Outputs>
double Learn<Inputs, Outputs>::calculate_epoch_error(const std::vector<double>& step_errors) {
    double result_error {0.0};

    for (const double error : step_errors) {
        result_error += error;
    }

    result_error /= static_cast<double>(step_errors.size());

    return result_error;
}

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::backpropagation(double* outputs, double* expected_outputs, network::Network<Inputs, Outputs>& network) const {
    // Output layer
    for (std::size_t i {0}; i < Outputs; i++) {
        network::Neuron& neuron {network.output_layer.neurons[i]};

        const double layer_error {outputs[i] - expected_outputs[i]};

        neuron.delta = layer_error * network::functions::sigmoid_derivative(outputs[i]);

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
            network::Neuron& neuron {iter->neurons[i]};

            double layer_error {0.0};

            if (is_last_hidden_layer) {
                for (auto iter2 = network.output_layer.neurons.begin(); iter2 != network.output_layer.neurons.end(); iter2++) {
                    network::Neuron& neuron {*iter2};

                    layer_error += neuron.weights[i] * neuron.delta;
                }
            } else {
                for (auto iter2 = std::prev(iter)->neurons.begin(); iter2 != std::prev(iter)->neurons.end(); iter2++) {
                    network::Neuron& neuron {*iter2};

                    layer_error += neuron.weights[i] * neuron.delta;
                }
            }

            neuron.delta = layer_error * network::functions::tanh_derivative(neuron.output);

            if (is_first_hidden_layer) {
                for (std::size_t j {0}; j < neuron.n; j++) {
                    const double change {options.learning_rate * data.inputs[j] * neuron.delta};
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
