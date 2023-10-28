#include <cstddef>
#include <algorithm>
#include <array>
#include <utility>
#include <cstdio>

#include "ui.hpp"
#include "helpers.hpp"

namespace ui {
    bool learning_setup(Learn& learn, neuron::Network& network) {
        static int hidden_layers = 1;
        static std::array<int, 32> hidden_layer_neurons = { 1, 1, 1 };

        bool start = false;

        if (ImGui::Begin("Learning Setup")) {
            if (ImGui::InputInt("Hidden layers", &hidden_layers)) {
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

            ImGui::InputDouble("Learning rate", &learn.rate);
            ImGui::InputDouble("Epsilon", &learn.epsilon);
            ImGui::InputScalar("Max epochs", ImGuiDataType_U64, &learn.max_epochs);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (learn.training_set.loaded) {
                if (ImGui::Button("Start")) {
                    neuron::Network::HiddenLayers layers;

                    for (int i = 0; i < hidden_layers; i++) {
                        layers.layers.push_back(hidden_layer_neurons[i]);
                    }

                    network.setup(6, 1, std::move(layers));

                    start = true;
                }
            } else {
                if (ImGui::Button("Choose training set")) {
                    learn.training_set.load("Qualitative_Bankruptcy.data.txt");  // TODO
                }
            }
        }

        ImGui::End();

        return start;
    }

    bool learning_process(const Learn& learn) {
        bool stop = false;

        if (ImGui::Begin("Learning Process")) {
            ImGui::Text("Epoch index: %lu", learn.epoch_index);
            ImGui::Text("Step index: %lu", learn.step_index);
            ImGui::Text("Current error: %f", learn.current_error);
            ImGui::Separator();
            ImGui::Text("Learning rate: %f", learn.rate);
            ImGui::Text("Epsilon: %f", learn.epsilon);
            ImGui::Text("Max epochs: %lu", learn.max_epochs);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("Stop")) {
                stop = true;
            }
        }

        ImGui::End();

        return stop;
    }

    void learning_graph(const Learn& learn) {
        if (ImGui::Begin("Learning Graph")) {
            // TODO graph
        }

        ImGui::End();
    }
}
