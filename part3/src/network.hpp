#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <limits>
#include <numbers>
#include <functional>
#include <utility>
#include <cassert>
#include <cmath>

#include "helpers.hpp"

namespace neuron {
    namespace functions {
        constexpr double sum(const double* inputs, const double* weights, std::size_t size) {
            double result = 0.0;

            for (std::size_t i = 0; i < size; i++) {
                result += inputs[i] * weights[i];
            }

            return result;
        }

        constexpr double sigmoid(double x, double theta, double g) {
            constexpr double e = std::numbers::e_v<double>;
            constexpr double one = 1.0;

            return one / (one + std::pow(e, -g * (x - theta)));
        }

        constexpr double sigmoid_derivative(double x) {
            return x * (1.0 - x);
        }

        constexpr double tanh(double x, double theta, double g) {
            constexpr double e = std::numbers::e_v<double>;

            const double a = std::pow(e, g * (x - theta));
            const double b = std::pow(e, -g * (x - theta));

            return (a - b) / (a + b);
        }

        constexpr double tanh_derivative(double x) {
            return 1.0 - x * x;
        }

        constexpr double binary(double x) {
            if (x >= 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        constexpr double binary2(double x) {
            if (x >= 0.0) {
                return 1.0;
            } else {
                return -1.0;
            }
        }
    }

    struct Neuron {
        double* weights = nullptr;
        std::size_t n = 0;
        double output = 0.0;
        double delta = 0.0;
    };

    struct HiddenLayer {
        std::vector<Neuron> neurons;
    };

    template<std::size_t Size>
    struct OutputLayer {
        std::array<Neuron, Size> neurons {};
    };

    struct HiddenLayers {
        std::vector<std::size_t> layers;
    };

    template<std::size_t Inputs, std::size_t Outputs>
    class Network {
    public:
        void run(const double* inputs, double* outputs);
        void setup(HiddenLayers&& hidden_layers);
        void initialize_neurons();

        constexpr std::size_t get_inputs() const {
            return Inputs;
        }

        constexpr std::size_t get_outputs() const {
            return Outputs;
        }

        OutputLayer<Outputs> output_layer;
        std::vector<HiddenLayer> hidden_layers;
    private:
        void clear();
        void allocate_current_inputs(double** inputs, std::size_t* n, const std::vector<Neuron>& neurons);
        void process_neuron_tanh(Neuron& neuron, const double* inputs, std::size_t n);
        void process_neuron_sigmoid(Neuron& neuron, const double* inputs, std::size_t n);
    };

    template<std::size_t Inputs, std::size_t Outputs>
    void Network<Inputs, Outputs>::run(const double* inputs, double* outputs) {
        std::size_t i = 0;

        auto& layer = hidden_layers[i];

        for (Neuron& neuron : layer.neurons) {
            process_neuron_tanh(neuron, inputs, Inputs);
        }

        std::size_t current_n = 0;
        double* current_inputs = nullptr;
        allocate_current_inputs(&current_inputs, &current_n, hidden_layers[i].neurons);

        i++;

        for (; i < hidden_layers.size(); i++) {
            auto& layer = hidden_layers[i];

            for (Neuron& neuron : layer.neurons) {
                process_neuron_tanh(neuron, current_inputs, current_n);
            }

            allocate_current_inputs(&current_inputs, &current_n, hidden_layers[i].neurons);
        }

        for (Neuron& neuron : output_layer.neurons) {
            process_neuron_sigmoid(neuron, current_inputs, current_n);
        }

        // Delete last allocation
        delete[] current_inputs;

        for (std::size_t j = 0; j < output_layer.neurons.size(); j++) {
            outputs[j] = output_layer.neurons[j].output;
        }
    }

    template<std::size_t Inputs, std::size_t Outputs>
    void Network<Inputs, Outputs>::setup(HiddenLayers&& hidden_layers) {
        static_assert(Inputs > 0);
        static_assert(Outputs > 0);
        assert(!hidden_layers.layers.empty());

        clear();

        this->hidden_layers.reserve(hidden_layers.layers.size());

        for (std::size_t neuron_count : hidden_layers.layers) {
            HiddenLayer layer;
            layer.neurons.resize(neuron_count);

            this->hidden_layers.push_back(std::move(layer));
        }

        initialize_neurons();
    }

    template<std::size_t Inputs, std::size_t Outputs>
    void Network<Inputs, Outputs>::initialize_neurons() {
        std::size_t current_inputs = Inputs;

        for (HiddenLayer& layer : hidden_layers) {
            for (Neuron& neuron : layer.neurons) {
                reallocate_double_array_random(&neuron.weights, &neuron.n, current_inputs);
            }

            current_inputs = layer.neurons.size();
        }

        for (Neuron& neuron : output_layer.neurons) {
            reallocate_double_array_random(&neuron.weights, &neuron.n, current_inputs);
        }
    }

    template<std::size_t Inputs, std::size_t Outputs>
    void Network<Inputs, Outputs>::clear() {
        output_layer = {};
        hidden_layers.clear();
    }

    template<std::size_t Inputs, std::size_t Outputs>
    void Network<Inputs, Outputs>::allocate_current_inputs(double** inputs, std::size_t* n, const std::vector<Neuron>& neurons) {
        delete[] *inputs;

        *n = neurons.size();
        *inputs = new double[*n];

        for (std::size_t i = 0; i < *n; i++) {
            (*inputs)[i] = neurons[i].output;
        }
    }

    template<std::size_t Inputs, std::size_t Outputs>
    void Network<Inputs, Outputs>::process_neuron_tanh(Neuron& neuron, const double* inputs, std::size_t n) {
        const double global_input = functions::sum(inputs, neuron.weights, n);
        const double activation = functions::tanh(global_input, 0.0, 1.0);
        neuron.output = activation;
    }

    template<std::size_t Inputs, std::size_t Outputs>
    void Network<Inputs, Outputs>::process_neuron_sigmoid(Neuron& neuron, const double* inputs, std::size_t n) {
        const double global_input = functions::sum(inputs, neuron.weights, n);
        const double activation = functions::sigmoid(global_input, 0.0, 1.0);
        neuron.output = activation;
    }
}
