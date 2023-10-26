#include <cstddef>
#include <algorithm>
#include <array>
#include <utility>
#include <cstdio>

#include "ui.hpp"
#include "helpers.hpp"

namespace ui {
    bool build_network(neuron::Network& network, double** inputs, std::size_t* n) {
        static int input_layer_neurons = 2;
        static int output_layer_neurons = 1;
        static int hidden_layers = 1;
        static std::array<int, 32> hidden_layer_neurons = { 3, 1, 1 };

        bool built = false;

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
                hidden_layers = std::min(hidden_layers, 32);
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

                reallocate_double_array(inputs, n, input_layer_neurons);

                built = true;
            }
        }

        ImGui::End();

        return built;
    }

    void learning_controls() {
        if (ImGui::Begin("Learning Controls")) {

        }

        ImGui::End();
    }
}
