#include <cstddef>
#include <algorithm>
#include <array>
#include <utility>

#include "ui.hpp"

namespace ui {
    struct Neuron {
        neuron::Neuron* neuron = nullptr;
        ImVec2 position;
    };

    struct Network {
        std::vector<std::vector<Neuron>> layers;
    };

    static std::size_t network_height(const neuron::Network& network) {
        std::size_t height = 0;

        height = std::max(height, network.input_neurons);
        height = std::max(height, network.output_layer.neurons.size());

        for (std::size_t i = 0; i < network.hidden_layers.size(); i++) {
            const neuron::Layer& layer = network.hidden_layers[i];

            height = std::max(height, layer.neurons.size());
        }

        return height;
    }

    static void build_input_layer(const neuron::Network& network, float x, float y, float network_height, Network& result) {
        result.layers.emplace_back();

        const float spacing = network_height / static_cast<float>(network.input_neurons + 1);

        for (std::size_t i = 0; i < network.input_neurons; i++) {
            Neuron neuron;
            neuron.position = ImVec2(x, y + i * spacing + spacing);

            result.layers.back().push_back(neuron);
        }
    }

    static float build_hidden_layers(const neuron::Network& network, float x, float y, float network_height, Network& result) {
        for (std::size_t i = 0; i < network.hidden_layers.size(); i++) {
            result.layers.emplace_back();

            const neuron::Layer& layer = network.hidden_layers[i];

            const float spacing = network_height / static_cast<float>(network.hidden_layers[i].neurons.size() + 1);

            for (std::size_t j = 0; j < layer.neurons.size(); j++) {
                Neuron neuron;
                neuron.position = ImVec2(x, y + j * spacing + spacing);
                neuron.neuron = const_cast<neuron::Neuron*>(layer.neurons.data()) + j;  // Original data is non-const

                result.layers.back().push_back(neuron);
            }

            x += 100.0f;
        }

        return x;
    }

    static void build_output_layer(const neuron::Network& network, float x, float y, float network_height, Network& result) {
        result.layers.emplace_back();

        const float spacing = network_height / static_cast<float>(network.output_layer.neurons.size() + 1);

        for (std::size_t i = 0; i < network.output_layer.neurons.size(); i++) {
            Neuron neuron;
            neuron.position = ImVec2(x, y + i * spacing + spacing);
            neuron.neuron = const_cast<neuron::Neuron*>(network.output_layer.neurons.data()) + i;  // Original data is non-const

            result.layers.back().push_back(neuron);
        }
    }

    static void build_network_struct(const neuron::Network& network, ImVec2 canvas, float offset, Network& result) {
        result.layers.clear();

        const float neuron_spacing = 60.0f;
        const float height = static_cast<float>(network_height(network)) * neuron_spacing;

        float x = offset;

        build_input_layer(network, canvas.x + x, canvas.y, height, result);

        x += 100.0f;

        x = build_hidden_layers(network, canvas.x + x, canvas.y, height, result);

        build_output_layer(network, x, canvas.y, height, result);
    }

    static void neuron_controls(neuron::Neuron* neuron, bool* neuron_window, bool& other_neuron_window) {
        if (!*neuron_window) {
            return;
        }

        if (ImGui::Begin("Neuron Controls", neuron_window)) {
            ImGui::Text("Values");
            ImGui::Spacing();

            ImGui::Text("Input  %f", 0.0);
            ImGui::Text("Activation  %f", 0.0);
            ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.3f, 1.0f), "Output  %f", 0.0);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("Weights");
            ImGui::Spacing();

            for (std::size_t i = 0; i < neuron->n; i++) {
                ImGui::PushID(i);

                ImGui::InputDouble("##", neuron->weights + i, 0.01);
                ImGui::SameLine();
                ImGui::Text("%lu", i);

                ImGui::PopID();
            }

            other_neuron_window = !ImGui::IsWindowHovered();
        }

        ImGui::End();
    }

    void draw_network(const neuron::Network& network) {
        static neuron::Neuron* selected_neuron = nullptr;
        static bool neuron_window = false;
        static bool other_neuron_window = true;

        const ImGuiWindowFlags flags = selected_neuron != nullptr ? ImGuiWindowFlags_NoBringToFrontOnFocus : 0;

        if (ImGui::Begin("Neural Network", nullptr, flags)) {
            ImDrawList* list = ImGui::GetWindowDrawList();
            const ImVec2 canvas = ImGui::GetCursorScreenPos();
            const float OFFSET = 30.0f;
            const float NEURON_SIZE = 20.0f;
            static constexpr auto NEURON_COLOR = IM_COL32(200, 200, 200, 255);
            static constexpr auto LINK_COLOR = IM_COL32(255, 255, 255, 255);

            Network result;
            build_network_struct(network, canvas, OFFSET, result);

            for (const auto& layer : result.layers) {
                for (const Neuron& neuron : layer) {
                    list->AddCircleFilled(neuron.position, NEURON_SIZE, NEURON_COLOR);

                    const auto upper_left = ImVec2(neuron.position.x - NEURON_SIZE, neuron.position.y - NEURON_SIZE);
                    const auto lower_right = ImVec2(neuron.position.x + NEURON_SIZE, neuron.position.y + NEURON_SIZE);

                    if (ImGui::IsMouseHoveringRect(upper_left, lower_right) && other_neuron_window) {
                        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                            if (neuron.neuron == nullptr) {
                                // TODO input neuron
                            } else {
                                selected_neuron = neuron.neuron;
                                neuron_window = true;

                                ImGui::SetNextWindowPos(neuron.position);
                            }
                        }
                    }
                }
            }

            for (std::size_t i = 1; i < result.layers.size(); i++) {
                for (const Neuron& neuron : result.layers[i]) {
                    for (const Neuron& neuron2 : result.layers[i - 1]) {
                        list->AddLine(neuron.position, neuron2.position, LINK_COLOR);
                    }
                }
            }
        }

        ImGui::End();

        if (selected_neuron != nullptr) {
            ImGui::SetNextWindowFocus();
            neuron_controls(selected_neuron, &neuron_window, other_neuron_window);

            if (!neuron_window) {
                selected_neuron = nullptr;
                other_neuron_window = true;
            }
        }
    }

    void build_network(neuron::Network& network) {
        static int input_layer_neurons = 2;
        static int output_layer_neurons = 1;
        static int hidden_layers = 1;
        static std::array<int, 3> hidden_layer_neurons = { 3, 1, 1 };

        if (ImGui::Begin("Build Network")) {
            ImGui::Text("Layers");
            ImGui::Spacing();

            if (ImGui::InputInt("Input", &input_layer_neurons)) {
                input_layer_neurons = std::max(input_layer_neurons, 1);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::InputInt("Hidden", &hidden_layers)) {
                hidden_layers = std::max(hidden_layers, 1);
                hidden_layers = std::min(hidden_layers, 3);
            }

            ImGui::Spacing();

            for (int i = 0; i < hidden_layers; i++) {
                ImGui::PushID(i);

                if (ImGui::InputInt("##", &hidden_layer_neurons[i])) {
                    hidden_layer_neurons[i] = std::max(hidden_layer_neurons[i], 1);
                }

                ImGui::SameLine();

                ImGui::Text("%d", i);

                ImGui::PopID();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::InputInt("Output", &output_layer_neurons)) {
                output_layer_neurons = std::max(output_layer_neurons, 1);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("Build")) {
                neuron::Network::HiddenLayers layers;

                for (int i = 0; i < hidden_layers; i++) {
                    layers.layers.push_back(hidden_layer_neurons[i]);
                }

                network.setup(input_layer_neurons, output_layer_neurons, std::move(layers));
            }
        }

        ImGui::End();
    }

    void network_controls(neuron::Network& network) {
        if (ImGui::Begin("Network Controls")) {
            ImGui::Text("Functions");
            ImGui::Spacing();

            for (std::size_t i = 0; i < network.hidden_layers.size(); i++) {
                ImGui::PushID(i);

                auto& layer = network.hidden_layers[i];

                ImGui::Text("Hidden Layer %lu", i);

                {
                    const char* items[] = { "sum", "product", "max", "min" };
                    static int item_current = 0;
                    const neuron::InputFunction values[] = {
                        neuron::functions::sum,
                        neuron::functions::product,
                        neuron::functions::max,
                        neuron::functions::min
                    };

                    if (ImGui::Combo("Input", &item_current, items, 4)) {
                        layer.set_input_function(values[item_current]);
                    }
                }

                {
                    const char* items[] = { "heaviside", "sigmoid", "signum", "tanh", "ramp" };
                    static int item_current = 1;
                    const neuron::ActivationFunction::Function values[] = {
                        neuron::ActivationFunction::heaviside,
                        neuron::ActivationFunction::sigmoid,
                        neuron::ActivationFunction::signum,
                        neuron::ActivationFunction::tanh,
                        neuron::ActivationFunction::ramp
                    };

                    if (ImGui::Combo("Activation", &item_current, items, 5)) {
                        layer.set_activation_function(values[item_current]);
                    }
                }

                {
                    const char* items[] = { "identity", "binary", "binary2" };
                    static int item_current = 0;
                    const neuron::OutputFunction values[] = {
                        neuron::functions::identity,
                        neuron::functions::binary,
                        neuron::functions::binary2
                    };

                    if (ImGui::Combo("Output", &item_current, items, 3)) {
                        layer.set_output_function(values[item_current]);
                    }
                }

                ImGui::Spacing();

                ImGui::PopID();
            }

            ImGui::Text("Output Layer");  // FIXME code is not dry

            {
                const char* items[] = { "sum", "product", "max", "min" };
                static int item_current = 0;
                const neuron::InputFunction values[] = {
                    neuron::functions::sum,
                    neuron::functions::product,
                    neuron::functions::max,
                    neuron::functions::min
                };

                if (ImGui::Combo("Input", &item_current, items, 4)) {
                    network.output_layer.set_input_function(values[item_current]);
                }
            }

            {
                const char* items[] = { "heaviside", "sigmoid", "signum", "tanh", "ramp" };
                static int item_current = 1;
                const neuron::ActivationFunction::Function values[] = {
                    neuron::ActivationFunction::heaviside,
                    neuron::ActivationFunction::sigmoid,
                    neuron::ActivationFunction::signum,
                    neuron::ActivationFunction::tanh,
                    neuron::ActivationFunction::ramp
                };

                if (ImGui::Combo("Activation", &item_current, items, 5)) {
                    network.output_layer.set_activation_function(values[item_current]);
                }
            }

            {
                const char* items[] = { "identity", "binary", "binary2" };
                static int item_current = 0;
                const neuron::OutputFunction values[] = {
                    neuron::functions::identity,
                    neuron::functions::binary,
                    neuron::functions::binary2
                };

                if (ImGui::Combo("Output", &item_current, items, 3)) {
                    network.output_layer.set_output_function(values[item_current]);
                }
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Constants");
        ImGui::Spacing();

        for (std::size_t i = 0; i < network.hidden_layers.size(); i++) {
            ImGui::PushID(i);

            auto& layer = network.hidden_layers[i];

            ImGui::Text("Hidden Layer %lu", i);

            ImGui::InputDouble("theta", &layer.activation_function.theta, 0.1f);
            ImGui::InputDouble("g", &layer.activation_function.g, 0.1f);
            ImGui::InputDouble("a", &layer.activation_function.a, 0.1f);

            ImGui::Spacing();

            ImGui::PopID();
        }

        ImGui::Text("Output Layer");  // FIXME code is not dry

        ImGui::InputDouble("theta", &network.output_layer.activation_function.theta, 0.1f);
        ImGui::InputDouble("g", &network.output_layer.activation_function.g, 0.1f);
        ImGui::InputDouble("a", &network.output_layer.activation_function.a, 0.1f);

        ImGui::End();
    }
}
