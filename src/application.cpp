#include <cstddef>
#include <functional>
#include <cstring>
#include <type_traits>

#include <gui_base/gui_base.hpp>

#include "application.hpp"

void NnApplication::start() {
    neuron.set_input_function(neuron::input_function::sum<Float>);
    neuron.set_activation_function(get_activation_function());
    neuron.set_output_function(neuron::output_function::identity<Float>);
}

void NnApplication::update() {
    // ImGui::ShowDemoWindow();

    constants_control();
    functions_control();
    neuron_inputs_control();
    neuron_output();
}

void NnApplication::dispose() {

}

void NnApplication::constants_control() {
    static constexpr auto TYPE = std::is_same_v<Float, float> ? ImGuiDataType_Float : ImGuiDataType_Double;
    constexpr Float MIN = static_cast<Float>(-100.0);
    constexpr Float MAX = static_cast<Float>(100.0);

    if (ImGui::Begin("Constants Control")) {
        if (ImGui::DragScalar("theta", TYPE, &constants.theta, 0.1f, &MIN, &MAX)) {
            neuron.set_activation_function(get_activation_function());
        }

        if (ImGui::DragScalar("g", TYPE, &constants.g, 0.1f, &MIN, &MAX)) {
            neuron.set_activation_function(get_activation_function());
        }

        if (ImGui::DragScalar("a", TYPE, &constants.a, 0.1f, &MIN, &MAX)) {
            neuron.set_activation_function(get_activation_function());
        }

        ImGui::End();
    }
}

void NnApplication::functions_control() {
    if (ImGui::Begin("Functions Control")) {
        {
            const char* items[] = { "sum", "product", "max", "min" };
            static int item_current = 0;
            const neuron::InputFunction<Float> values[] = {
                neuron::input_function::sum<Float>,
                neuron::input_function::product<Float>,
                neuron::input_function::max<Float>,
                neuron::input_function::min<Float>
            };

            if (ImGui::Combo("Input Function", &item_current, items, 4)) {
                neuron.set_input_function(values[item_current]);
            }
        }

        {
            using namespace std::placeholders;

            const char* items[] = { "heaviside", "sigmoid", "signum", "tanh", "ramp" };
            const neuron::ActivationFunction<Float> values[] = {
                std::bind(neuron::activation_function::heaviside<Float>, _1, constants.theta),
                std::bind(neuron::activation_function::sigmoid<Float>, _1, constants.theta, constants.g),
                std::bind(neuron::activation_function::signum<Float>, _1, constants.theta),
                std::bind(neuron::activation_function::tanh<Float>, _1, constants.theta, constants.g),
                std::bind(neuron::activation_function::ramp<Float>, _1, constants.a)
            };

            if (ImGui::Combo("Activation Function", &activation_function_current, items, 5)) {
                neuron.set_activation_function(values[activation_function_current]);
            }
        }

        {
            const char* items[] = { "identity", "clamp_binary", "clamp_binary2" };
            static int item_current = 0;
            const neuron::OutputFunction<Float> values[] = {
                neuron::output_function::identity<Float>,
                neuron::output_function::clamp_binary<Float>,
                neuron::output_function::clamp_binary2<Float>
            };

            if (ImGui::Combo("Output Function", &item_current, items, 3)) {
                neuron.set_output_function(values[item_current]);
            }
        }

        ImGui::End();
    }
}

void NnApplication::neuron_inputs_control() {
    if (ImGui::Begin("Neuron Inputs Control")) {
        ImGui::Text("Number of inputs: %lu", number_of_inputs);

        ImGui::SameLine();

        if (ImGui::ArrowButton("Left", ImGuiDir_Left)) {
            if (number_of_inputs > 0) {
                reallocate_inputs(number_of_inputs - 1, number_of_inputs);
                neuron.setup_inputs(number_of_inputs - 1);

                number_of_inputs--;
            }
        }

        ImGui::SameLine();

        if (ImGui::ArrowButton("Right", ImGuiDir_Right)) {
            reallocate_inputs(number_of_inputs + 1, number_of_inputs);
            neuron.setup_inputs(number_of_inputs + 1);

            number_of_inputs++;
        }

        ImGui::Separator();

        if (number_of_inputs == 0) {
            ImGui::Text("There must be at least one input.");
        }

        static constexpr auto TYPE = std::is_same_v<Float, float> ? ImGuiDataType_Float : ImGuiDataType_Double;
        constexpr Float MIN = static_cast<Float>(-50.0);
        constexpr Float MAX = static_cast<Float>(50.0);

        for (std::size_t i = 0; i < number_of_inputs; i++) {
            ImGui::PushID(i);

            ImGui::Text("x%lu", i);
            ImGui::SameLine();
            ImGui::DragScalar("##x", TYPE, inputs + i, 0.1f, &MIN, &MAX);

            ImGui::Text("w%lu", i);
            ImGui::SameLine();
            ImGui::DragScalar("##w", TYPE, neuron.get_weights() + i, 0.1f, &MIN, &MAX);

            ImGui::Spacing();

            ImGui::PopID();
        }

        ImGui::End();
    }
}

void NnApplication::neuron_output() {
    if (ImGui::Begin("Neuron Output")) {
        static bool intermediary_steps = false;
        ImGui::Checkbox("Show intermediary steps", &intermediary_steps);

        ImGui::Separator();

        if (intermediary_steps) {
            if (neuron.is_valid()) {
                Float global_input, activation, output;
                neuron.process_in_steps(inputs, global_input, activation, output);

                ImGui::Text("Global input:  %f", global_input);
                ImGui::Text("Activation:  %f", activation);
                ImGui::Text("Output:  %f", output);
            } else {
                ImGui::Text("Neuron is invalid.");
            }
        } else {
            if (neuron.is_valid()) {
                const Float output = neuron.process(inputs);

                ImGui::Text("Output:  %f", output);
            } else {
                ImGui::Text("Neuron is invalid.");
            }
        }

        ImGui::End();
    }
}

void NnApplication::activation_function_plot() {

}

neuron::ActivationFunction<NnApplication::Float> NnApplication::get_activation_function() {
    using namespace std::placeholders;

    const neuron::ActivationFunction<Float> functions[] = {
        std::bind(neuron::activation_function::heaviside<Float>, _1, constants.theta),
        std::bind(neuron::activation_function::sigmoid<Float>, _1, constants.theta, constants.g),
        std::bind(neuron::activation_function::signum<Float>, _1, constants.theta),
        std::bind(neuron::activation_function::tanh<Float>, _1, constants.theta, constants.g),
        std::bind(neuron::activation_function::ramp<Float>, _1, constants.a)
    };

    return functions[activation_function_current];
}

void NnApplication::reallocate_inputs(std::size_t new_size, std::size_t old_size) {
    if (new_size == 0) {
        delete[] inputs;
        inputs = nullptr;

        return;
    }

    Float* new_inputs = new Float[new_size];

    if (old_size > 0) {
        std::memcpy(new_inputs, inputs, old_size * sizeof(Float));
    }

    delete[] inputs;
    inputs = new_inputs;
}
