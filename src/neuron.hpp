#pragma once

#include <functional>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <cmath>
#include <numbers>

/*
    It is assumed that Real is a IEEE-754 floating point type, which means that comparisons
    between whole numbers (0.0, 1.0, 2.0 etc.) work reliably.
*/

namespace neuron {
    template<typename Real>
    struct Neuron {

    };

    template<typename Real>
    using InputFunction = std::function<Real(const Real*, const Real*, std::size_t)>;

    template<typename Real>
    using ActivationFunction = std::function<Real(Real)>;

    template<typename Real>
    using OutputFunction = std::function<Real(Real)>;

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

            return static_cast<Real>(1.0) / std::pow<Real>(e, -g * (x - theta));
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
            if (x >= static_cast<Real>(0.5)) {
                return static_cast<Real>(1.0);
            } else {
                return static_cast<Real>(-1.0);
            }
        }
    }
}
