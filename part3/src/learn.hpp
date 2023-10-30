#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include <thread>
#include <iterator>

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
        double learning_rate {0.05};
        double epsilon {0.01};
        unsigned long max_epochs {10'000};
    } options;

    unsigned long epoch_index {0};
    std::size_t step_index {0};
    double epoch_error {1.0};
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

    // Return true when it should stop
    bool update(neuron::Network<Inputs, Outputs>& network);
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

    if (thread.joinable()) {
        thread.join();
    }
}

template<std::size_t Inputs, std::size_t Outputs>
void Learn<Inputs, Outputs>::reset() {
    stop();

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

    // // Last hidden layer
    // for (std::size_t i {0}; i < network.hidden_layers.rbegin()->neurons.size(); i++) {
    //     neuron::Neuron& neuron {network.hidden_layers[network.hidden_layers.size() - 1].neurons[i]};

    //     double layer_error {0.0};
    //     for (auto iter = network.output_layer.neurons.begin(); iter != network.output_layer.neurons.end(); iter++) {
    //         neuron::Neuron& neuron {*iter};

    //         layer_error += neuron.weights[i] * neuron.delta;
    //     }

    //     neuron.delta = layer_error * neuron::functions::tanh_derivative(neuron.output);

    //     for (std::size_t j {0}; j < neuron.n; j++) {
    //         if (i != std::prev(network.hidden_layers.rend())) {
    //             const double delta_weight {options.learning_rate * std::next(iter)->neurons[j].output * neuron.delta};
    //             neuron.weights[j] += delta_weight;
    //         } else {
    //             const double delta_weight {options.learning_rate * inputs[j] * neuron.delta};  // FIXME
    //             neuron.weights[j] += delta_weight;
    //         }
    //     }
    // }

    // Rest of hidden layers
    // for (auto iter = std::next(network.hidden_layers.rbegin()); iter != network.hidden_layers.rend(); iter++) {
    //     for (std::size_t i {0}; i < iter->neurons.size(); i++) {
    //         neuron::Neuron& neuron {iter->neurons[i]};

    //         double delta_sum {0.0};
    //         for (auto iter2 = std::prev(iter)->neurons.begin(); iter2 != std::prev(iter)->neurons.end(); iter2++) {
    //             neuron::Neuron& neuron {*iter2};

    //             delta_sum += neuron.weights[i] * neuron.delta;
    //         }

    //         neuron.delta = delta_sum * neuron::functions::tanh_derivative(neuron.output);  // FIXME

    //         for (std::size_t j {0}; j < neuron.n; j++) {
    //             if (iter != std::prev(network.hidden_layers.rend())) {
    //                 const double delta_weight {options.rate * std::next(iter)->neurons[j].output * neuron.delta};
    //                 neuron.weights[j] += delta_weight;
    //             } else {
    //                 const double delta_weight {options.rate * inputs[j] * 1.0f};  // FIXME
    //                 neuron.weights[j] += delta_weight;
    //             }
    //         }
    //     }
    // }
}
