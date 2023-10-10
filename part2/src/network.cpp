#include <vector>
#include <cstddef>
#include <limits>
#include <cmath>
#include <numbers>
#include <functional>
#include <utility>
#include <cassert>

#include "network.hpp"

namespace neuron {
    double ActivationFunction::heaviside(double x) {
        return functions::heaviside(x, theta);
    }

    double ActivationFunction::sigmoid(double x) {
        return functions::sigmoid(x, theta, g);
    }

    double ActivationFunction::signum(double x) {
        return functions::signum(x, theta);
    }

    double ActivationFunction::tanh(double x) {
        return functions::tanh(x, theta, g);
    }

    double ActivationFunction::ramp(double x) {
        return functions::ramp(x, a);
    }

    void Layer::set_input_function(const InputFunction& input_function) {
        this->input_function = input_function;
    }

    void Layer::set_activation_function(const ActivationFunction::Function& activation_function) {
        this->activation_function.set(activation_function);
    }

    void Layer::set_output_function(const OutputFunction& output_function) {
        this->output_function = output_function;
    }

    void Network::run(const double* inputs, const double* outputs) {
        for (const Layer& layer : hidden_layers) {
            for (Neuron neuron : layer.neurons) {
                // layer.input_function(inputs, , input_neurons);
            }
        }
    }

    void Network::setup(std::size_t input_neurons, std::size_t output_neurons, HiddenLayers&& hidden_layers) {
        assert(input_neurons > 0);
        assert(output_neurons > 0);
        assert(!hidden_layers.layers.empty());

        clear();

        this->input_neurons = input_neurons;
        output_layer.neurons.resize(output_neurons);
        this->hidden_layers.reserve(hidden_layers.layers.size());

        for (std::size_t neuron_count : hidden_layers.layers) {
            Layer layer;
            layer.neurons.resize(neuron_count);

            this->hidden_layers.push_back(std::move(layer));
        }

        initialize_neurons();
    }

    void Network::clear() {
        input_neurons = 0;
        output_layer.neurons.clear();
        hidden_layers.clear();
    }

    void Network::initialize_neurons() {
        std::size_t current_inputs = input_neurons;

        for (const Layer& layer : this->hidden_layers) {
            for (Neuron neuron : layer.neurons) {
                neuron.weights = new double[current_inputs];
                neuron.n = current_inputs;
            }

            current_inputs = layer.neurons.size();
        }

        for (Neuron neuron : output_layer.neurons) {
            neuron.weights = new double[current_inputs];
            neuron.n = current_inputs;
        }
    }

    namespace functions {
        double sum(const double* inputs, const double* weights, std::size_t size) {
            double result = 0.0;

            for (std::size_t i = 0; i < size; i++) {
                result += inputs[i] * weights[i];
            }

            return result;
        }

        double product(const double* inputs, const double* weights, std::size_t size) {
            double result = 1.0;

            for (std::size_t i = 0; i < size; i++) {
                result *= inputs[i] * weights[i];
            }

            return result;
        }

        double max(const double* inputs, const double* weights, std::size_t size) {
            double result = std::numeric_limits<double>::min();

            for (std::size_t i = 0; i < size; i++) {
                result = std::max(result, inputs[i] * weights[i]);
            }

            return result;
        }

        double min(const double* inputs, const double* weights, std::size_t size) {
            double result = std::numeric_limits<double>::max();

            for (std::size_t i = 0; i < size; i++) {
                result = std::min(result, inputs[i] * weights[i]);
            }

            return result;
        }

        double heaviside(double x, double theta) {
            if (x >= theta) {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        double sigmoid(double x, double theta, double g) {
            static constexpr double e = std::numbers::e_v<double>;
            static constexpr double one = 1.0;

            return one / (one + std::pow(e, -g * (x - theta)));
        }

        double signum(double x, double theta) {
            if (x >= theta) {
                return 1.0;
            } else {
                return -1.0;
            }
        }

        double tanh(double x, double theta, double g) {
            static constexpr double e = std::numbers::e_v<double>;

            const double a = std::pow(e, g * (x - theta));
            const double b = std::pow(e, -g * (x - theta));

            return (a - b) / (a + b);
        }

        double ramp(double x, double a) {
            if (x < -a) {
                return -1.0;
            } else if (x > a) {
                return 1.0;
            } else {
                return x / a;
            }
        }

        double identity(double x) {
            return x;
        }

        double clamp_binary(double x) {
            if (x >= 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        double clamp_binary2(double x) {
            if (x >= 0.0) {
                return 1.0;
            } else {
                return -1.0;
            }
        }
    }
}
