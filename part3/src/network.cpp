#include <vector>
#include <array>
#include <cstddef>
#include <limits>
#include <numbers>
#include <functional>
#include <utility>
#include <cassert>
#include <cmath>

#include "network.hpp"

namespace neuron {
    namespace functions {
        double sum(const double* inputs, const double* weights, std::size_t size) {
            double result = 0.0;

            for (std::size_t i = 0; i < size; i++) {
                result += inputs[i] * weights[i];
            }

            return result;
        }

        double sigmoid(double x, double theta, double g) {
            static constexpr double e = std::numbers::e_v<double>;
            static constexpr double one = 1.0;

            return one / (one + std::pow(e, -g * (x - theta)));
        }

        double tanh(double x, double theta, double g) {
            static constexpr double e = std::numbers::e_v<double>;

            const double a = std::pow(e, g * (x - theta));
            const double b = std::pow(e, -g * (x - theta));

            return (a - b) / (a + b);
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
}
