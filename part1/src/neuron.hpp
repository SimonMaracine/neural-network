#pragma once

#include <functional>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numbers>
#include <cstring>

/*
    It is assumed that Real is a IEEE-754 floating point type, which means that comparisons
    between whole numbers (0.0, 1.0, 2.0 etc.) work reliably.
*/

namespace neuron {
    template<typename Real>
    using InputFunction = std::function<Real(const Real*, const Real*, std::size_t)>;

    template<typename Real>
    using ActivationFunction = std::function<Real(Real)>;

    template<typename Real>
    using OutputFunction = std::function<Real(Real)>;

    template<typename Real>
    class Neuron {
    public:
        Neuron() = default;
        ~Neuron() = default;
        Neuron(const Neuron&) = delete;
        Neuron& operator=(const Neuron&) = delete;
        Neuron(Neuron&&) = delete;
        Neuron& operator=(Neuron&&) = delete;

        void setup_inputs(std::size_t n) {
            if (n == 0) {
                delete[] weights;
                weights = nullptr;
                this->n = n;
            }

            Real* new_weights = new Real[n];

            if (this->n > 0) {
                std::memcpy(new_weights, weights, std::min(this->n, n) * sizeof(Real));
            }

            delete[] weights;
            weights = new_weights;
            this->n = n;
        }

        Real* get_weights() const {
            return weights;
        }

        void set_input_function(InputFunction<Real> input_function) {
            this->input_function = input_function;
        }

        void set_activation_function(ActivationFunction<Real> activation_function) {
            this->activation_function = activation_function;
        }

        void set_output_function(OutputFunction<Real> output_function) {
            this->output_function = output_function;
        }

        bool is_valid() const {
            return n > 0 && input_function && activation_function && output_function;
        }

        Real process(const Real* inputs) {
            const Real global_input = input_function(inputs, weights, n);
            const Real activation = activation_function(global_input);
            return output_function(activation);
        }

        void process_in_steps(const Real* inputs, Real& global_input, Real& activation, Real& output) {
            global_input = input_function(inputs, weights, n);
            activation = activation_function(global_input);
            output = output_function(activation);
        }
    private:
        Real* weights = nullptr;
        std::size_t n = 0;

        InputFunction<Real> input_function;
        ActivationFunction<Real> activation_function;
        OutputFunction<Real> output_function;
    };

    namespace input_function {
        template<typename Real>
        Real sum(const Real* inputs, const Real* weights, std::size_t size) {
            Real result = static_cast<Real>(0.0);

            for (std::size_t i = 0; i < size; i++) {
                result += inputs[i] * weights[i];
            }

            return result;
        }

        template<typename Real>
        Real product(const Real* inputs, const Real* weights, std::size_t size) {
            Real result = static_cast<Real>(1.0);

            for (std::size_t i = 0; i < size; i++) {
                result *= inputs[i] * weights[i];
            }

            return result;
        }

        template<typename Real>
        Real max(const Real* inputs, const Real* weights, std::size_t size) {
            Real result = std::numeric_limits<Real>::min();

            for (std::size_t i = 0; i < size; i++) {
                result = std::max(result, inputs[i] * weights[i]);
            }

            return result;
        }

        template<typename Real>
        Real min(const Real* inputs, const Real* weights, std::size_t size) {
            Real result = std::numeric_limits<Real>::max();

            for (std::size_t i = 0; i < size; i++) {
                result = std::min(result, inputs[i] * weights[i]);
            }

            return result;
        }
    }

    namespace activation_function {
        template<typename Real>
        Real heaviside(Real x, Real theta) {
            if (x >= theta) {
                return static_cast<Real>(1.0);
            } else {
                return static_cast<Real>(0.0);
            }
        }

        template<typename Real>
        Real sigmoid(Real x, Real theta, Real g) {
            static constexpr Real e = std::numbers::e_v<Real>;
            static constexpr Real one = static_cast<Real>(1.0);

            return one / (one + std::pow<Real>(e, -g * (x - theta)));
        }

        template<typename Real>
        Real signum(Real x, Real theta) {
            if (x >= theta) {
                return static_cast<Real>(1.0);
            } else {
                return static_cast<Real>(-1.0);
            }
        }

        template<typename Real>
        Real tanh(Real x, Real theta, Real g) {
            static constexpr Real e = std::numbers::e_v<Real>;

            const Real a = std::pow<Real>(e, g * (x - theta));
            const Real b = std::pow<Real>(e, -g * (x - theta));

            return (a - b) / (a + b);
        }

        template<typename Real>
        Real ramp(Real x, Real a) {
            if (x < -a) {
                return static_cast<Real>(-1.0);
            } else if (x > a) {
                return static_cast<Real>(1.0);
            } else {
                return x / a;
            }
        }
    }

    namespace output_function {
        template<typename Real>
        Real identity(Real x) {
            return x;
        }

        template<typename Real>
        Real clamp_binary(Real x) {
            if (x >= static_cast<Real>(0.5)) {
                return static_cast<Real>(1.0);
            } else {
                return static_cast<Real>(0.0);
            }
        }

        template<typename Real>
        Real clamp_binary2(Real x) {
            if (x >= static_cast<Real>(0.0)) {
                return static_cast<Real>(1.0);
            } else {
                return static_cast<Real>(-1.0);
            }
        }
    }
}
