#pragma once

#include <vector>
#include <cstddef>
#include <limits>
#include <cmath>
#include <numbers>
#include <functional>

namespace neuron {
    namespace functions {
        double sum(const double* inputs, const double* weights, std::size_t size);

        double sigmoid(double x, double theta, double g);
        double tanh(double x, double theta, double g);

        double binary(double x);
        double binary2(double x);
    }

    struct Neuron {
        double* weights = nullptr;
        std::size_t n = 0;
        double output = 0.0;
    };

    class Network;

    struct Layer {
        std::vector<Neuron> neurons;
    };

    struct Network {
        struct HiddenLayers {
            std::vector<std::size_t> layers;
        };

        void run(const double* inputs, double* outputs);
        void setup(std::size_t input_neurons, std::size_t output_neurons, HiddenLayers&& hidden_layers);
        void clear();

        void initialize_neurons();
        void allocate_current_inputs(double** inputs, std::size_t* n, const std::vector<Neuron>& neurons);
        void process_neuron_tanh(Neuron& neuron, const double* inputs, std::size_t n);
        void process_neuron_sigmoid(Neuron& neuron, const double* inputs, std::size_t n);

        std::size_t input_neurons {};
        Layer output_layer;
        std::vector<Layer> hidden_layers;
    };
}
