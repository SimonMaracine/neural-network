#pragma once

#include <vector>
#include <cstddef>
#include <limits>
#include <cmath>
#include <numbers>
#include <functional>

namespace neuron {
    using InputFunction = std::function<double(const double*, const double*, std::size_t)>;
    using OutputFunction = std::function<double(double)>;

    class ActivationFunction {
    public:
        using Function = std::function<double(const ActivationFunction*, double)>;

        ActivationFunction() = default;
        ActivationFunction(const Function& function)
            : function(function) {}

        double theta = 0.0f;
        double g = 1.0f;
        double a = 1.0f;

        double operator()(double x) const {
            return function(this, x);
        }

        void set(const Function& function) {
            this->function = function;
        }

        static double heaviside(const ActivationFunction* self, double x);
        static double sigmoid(const ActivationFunction* self, double x);
        static double signum(const ActivationFunction* self, double x);
        static double tanh(const ActivationFunction* self, double x);
        static double ramp(const ActivationFunction* self, double x);
    private:
        Function function = sigmoid;
    };

    namespace functions {
        double sum(const double* inputs, const double* weights, std::size_t size);
        double product(const double* inputs, const double* weights, std::size_t size);
        double max(const double* inputs, const double* weights, std::size_t size);
        double min(const double* inputs, const double* weights, std::size_t size);

        double heaviside(double x, double theta);
        double sigmoid(double x, double theta, double g);
        double signum(double x, double theta);
        double tanh(double x, double theta, double g);
        double ramp(double x, double a);

        double identity(double x);
        double binary(double x);
        double binary2(double x);
    }

    struct Neuron {
        double* weights = nullptr;
        std::size_t n = 0;

        struct {
            double global_input = 0.0f;
            double activation = 0.0f;
            double output = 0.0f;
        } result;
    };

    class Network;

    struct Layer {
        void set_input_function(const InputFunction& input_function);
        void set_activation_function(const ActivationFunction::Function& activation_function);
        void set_output_function(const OutputFunction& output_function);

        std::vector<Neuron> neurons;

        InputFunction input_function = functions::sum;
        ActivationFunction activation_function = ActivationFunction(ActivationFunction::sigmoid);
        OutputFunction output_function = functions::identity;
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
        void process_neuron(Neuron& neuron, const Layer& layer, const double* inputs, std::size_t n);

        std::size_t input_neurons {};
        Layer output_layer;
        std::vector<Layer> hidden_layers;
    };
}
