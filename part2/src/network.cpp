#include <vector>
#include <cstddef>
#include <limits>
#include <cmath>
#include <numbers>
#include <functional>
#include <utility>
#include <cassert>
#include <cstring>

#include "network.hpp"

namespace neuron {
    double ActivationFunction::heaviside(const ActivationFunction* self, double x) {
        return functions::heaviside(x, self->theta);
    }

    double ActivationFunction::sigmoid(const ActivationFunction* self, double x) {
        return functions::sigmoid(x, self->theta, self->g);
    }

    double ActivationFunction::signum(const ActivationFunction* self, double x) {
        return functions::signum(x, self->theta);
    }

    double ActivationFunction::tanh(const ActivationFunction* self, double x) {
        return functions::tanh(x, self->theta, self->g);
    }

    double ActivationFunction::ramp(const ActivationFunction* self, double x) {
        return functions::ramp(x, self->a);
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

        double binary(double x) {
            if (x >= 0.5) {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        double binary2(double x) {
            if (x >= 0.0) {
                return 1.0;
            } else {
                return -1.0;
            }
        }
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

    void Network::run(const double* inputs, double* outputs) {
        std::size_t i = 0;

        auto& layer = hidden_layers[i];

        for (Neuron& neuron : layer.neurons) {
            process_neuron(neuron, layer, inputs, input_neurons);
        }

        std::size_t current_n = 0;
        double* current_inputs = nullptr;
        allocate_current_inputs(&current_inputs, &current_n, hidden_layers[i].neurons);

        i++;

        for (; i < hidden_layers.size(); i++) {
            auto& layer = hidden_layers[i];

            for (Neuron& neuron : layer.neurons) {
                process_neuron(neuron, layer, current_inputs, current_n);
            }

            allocate_current_inputs(&current_inputs, &current_n, hidden_layers[i].neurons);
        }

        for (Neuron& neuron : output_layer.neurons) {
            process_neuron(neuron, output_layer, current_inputs, current_n);
        }

        if (outputs != nullptr) {
            for (std::size_t j = 0; j < output_layer.neurons.size(); j++) {
                outputs[j] = output_layer.neurons[j].result.output;
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

        for (Layer& layer : hidden_layers) {
            for (Neuron& neuron : layer.neurons) {
                delete[] neuron.weights;
                neuron.weights = new double[current_inputs];
                neuron.n = current_inputs;
                std::memset(neuron.weights, 0, current_inputs);  // FIXME doesn't work sometimes
            }

            current_inputs = layer.neurons.size();
        }

        for (Neuron& neuron : output_layer.neurons) {
            delete[] neuron.weights;
            neuron.weights = new double[current_inputs];
            neuron.n = current_inputs;
            std::memset(neuron.weights, 0, current_inputs);
        }
    }

    void Network::allocate_current_inputs(double** inputs, std::size_t* n, const std::vector<Neuron>& neurons) {
        delete[] *inputs;

        *n = neurons.size();
        *inputs = new double[*n];

        for (std::size_t i = 0; i < *n; i++) {
            (*inputs)[i] = neurons[i].result.output;
        }
    }

    void Network::process_neuron(Neuron& neuron, const Layer& layer, const double* inputs, std::size_t n) {
        neuron.result.global_input = layer.input_function(inputs, neuron.weights, n);
        neuron.result.activation = layer.activation_function(neuron.result.global_input);
        neuron.result.output = layer.output_function(neuron.result.activation);
    }
}
