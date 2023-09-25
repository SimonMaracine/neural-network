#include <functional>

#include <gui_base/gui_base.hpp>

#include "application.hpp"

void NnApplication::start() {
    update_activation_function();
}

void NnApplication::update() {
    ImGui::ShowDemoWindow();

    constants_control();
    functions_control();
}

void NnApplication::dispose() {

}

void NnApplication::constants_control() {
    if (ImGui::Begin("Constants Control")) {
        if (ImGui::DragFloat("theta", &constants.theta, 0.1f, -100.0f, 100.0f)) {
            update_activation_function();
        }

        if (ImGui::DragFloat("g", &constants.g, 0.1f, -100.0f, 100.0f)) {
            update_activation_function();
        }

        if (ImGui::DragFloat("a", &constants.a, 0.1f, -100.0f, 100.0f)) {
            update_activation_function();
        }

        ImGui::End();
    }
}

void NnApplication::functions_control() {
    if (ImGui::Begin("Functions Control")) {
        {
            const char* items[] = { "sum", "product", "max", "min" };
            static int item_current = 0;
            const neuron::InputFunction<float> values[] = {
                neuron::input_function::sum<float>,
                neuron::input_function::product<float>,
                neuron::input_function::max<float>,
                neuron::input_function::min<float>
            };

            if (ImGui::Combo("Input Function", &item_current, items, 4)) {
                input_function = values[item_current];
            }
        }

        {
            using namespace std::placeholders;

            const char* items[] = { "heaviside", "sigmoid", "signum", "tanh", "ramp" };
            const neuron::ActivationFunction<float> values[] = {
                std::bind(neuron::activation_function::heaviside<float>, _1, constants.theta),
                std::bind(neuron::activation_function::sigmoid<float>, _1, constants.theta, constants.g),
                std::bind(neuron::activation_function::signum<float>, _1, constants.theta),
                std::bind(neuron::activation_function::tanh<float>, _1, constants.theta, constants.g),
                std::bind(neuron::activation_function::ramp<float>, _1, constants.a)
            };

            if (ImGui::Combo("Activation Function", &activation_function_current, items, 5)) {
                activation_function = values[activation_function_current];
            }
        }

        {
            const char* items[] = { "identity", "clamp_binary", "clamp_binary2" };
            static int item_current = 0;
            const neuron::OutputFunction<float> values[] = {
                neuron::output_function::identity<float>,
                neuron::output_function::clamp_binary<float>,
                neuron::output_function::clamp_binary2<float>
            };

            if (ImGui::Combo("Output Function", &item_current, items, 3)) {
                output_function = values[item_current];
            }
        }

        ImGui::End();
    }
}

void NnApplication::activation_function_plot() {
    auto function = [](void* self, int i) {
        NnApplication* app = static_cast<NnApplication*>(self);

        return app->activation_function(static_cast<float>(i) * 0.1f);
    };

    if (ImGui::Begin("Activation Function")) {
        ImGui::PlotLines("Lines", function, this, 128, -64, nullptr, -1.0f, 2.0f, ImVec2(0, 100));

        ImGui::End();
    }
}

void NnApplication::update_activation_function() {
    using namespace std::placeholders;

    const neuron::ActivationFunction<float> functions[] = {
        std::bind(neuron::activation_function::heaviside<float>, _1, constants.theta),
        std::bind(neuron::activation_function::sigmoid<float>, _1, constants.theta, constants.g),
        std::bind(neuron::activation_function::signum<float>, _1, constants.theta),
        std::bind(neuron::activation_function::tanh<float>, _1, constants.theta, constants.g),
        std::bind(neuron::activation_function::ramp<float>, _1, constants.a)
    };

    activation_function = functions[activation_function_current];
}
