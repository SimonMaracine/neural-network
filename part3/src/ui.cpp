#include <functional>
#include <string>
#include <cstddef>
#include <algorithm>
#include <array>
#include <utility>
#include <cstdio>

#include <gui_base/gui_base.hpp>
#include <ImGuiFileDialog.h>
#include <implot.h>

#include "network.hpp"
#include "learn.hpp"
#include "ui.hpp"
#include "helpers.hpp"

namespace ui {
    static const char* token_to_string(Instance::Token token) {
        switch (token) {
            case Instance::Token::Positive:
                return "P";
            case Instance::Token::Average:
                return "A";
            case Instance::Token::Negative:
                return "N";
            case Instance::Token::Bankrupt:
                return "B";
            case Instance::Token::NonBankrupt:
                return "NB";
        }

        return nullptr;
    }

    bool learning_setup(Learn<6, 1>& learn, neuron::Network<6, 1>& network) {
        static int hidden_layers = 1;
        static std::array<int, 32> hidden_layer_neurons = { 32, 32, 32 };

        bool start = false;

        if (ImGui::Begin("Learning Setup")) {
            ImGui::Text("Inputs: %lu", network.get_inputs());
            ImGui::Text("Outputs: %lu", network.get_outputs());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

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

            ImGui::InputDouble("Learning rate", &learn.options.rate);
            ImGui::InputDouble("Epsilon", &learn.options.epsilon);
            ImGui::InputScalar("Max epochs", ImGuiDataType_U64, &learn.options.max_epochs);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (learn.training_set.loaded && learn.training_set.normalized) {
                if (ImGui::Button("Start")) {
                    neuron::HiddenLayers layers;

                    for (int i = 0; i < hidden_layers; i++) {
                        layers.layers.push_back(hidden_layer_neurons[i]);
                    }

                    network.setup(std::move(layers));

                    start = true;
                }
            } else {
                if (ImGui::Button("Choose training set")) {
                    ui::open_file_browser();
                }
            }
        }

        ImGui::End();

        return start;
    }

    int learning_process(const Learn<6, 1>& learn) {
        int result {0};

        if (ImGui::Begin("Learning Process")) {
            ImGui::Text("Epoch index: %lu", learn.epoch_index);
            ImGui::Text("Step index: %lu", learn.step_index);
            ImGui::Text("Current error: %f", learn.epoch_error);
            ImGui::Separator();
            ImGui::Text("Learning rate: %f", learn.options.rate);
            ImGui::Text("Epsilon: %f", learn.options.epsilon);
            ImGui::Text("Max epochs: %lu", learn.options.max_epochs);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (learn.is_running()) {
                if (ImGui::Button("Stop")) {
                    result = -1;
                }
            } else {
                if (ImGui::Button("Start")) {
                    result = 1;
                }
            }
        }

        ImGui::End();

        return result;
    }

    void learning_graph(const Learn<6, 1>& learn) {
        if (ImGui::Begin("Learning Graph")) {
            ImPlot::SetNextAxesToFit();

            if (ImPlot::BeginPlot("Epoch Error", ImVec2(-1.0f, 0.0f), ImPlotAxisFlags_AutoFit)) {
                ImPlot::SetupAxes("Index", "Error");
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0);

                ImPlot::PlotLine(
                    "Epoch Error",
                    learn.error_graph.indices.data(),
                    learn.error_graph.errors.data(),
                    static_cast<int>(learn.error_graph.indices.size())
                );

                ImPlot::EndPlot();
            }
        }

        ImGui::End();
    }

    void training_set(TrainingSet& training_set) {
        if (ImGui::Begin("Training Set")) {
            if (ImGui::Button("Shuffle")) {
                training_set.shuffle();
            }

            ImGui::SameLine();

            if (ImGui::Button("Normalize")) {
                training_set.normalize();
            }

            ImGui::Spacing();

            if (ImGui::BeginTable("Training", 8, ImGuiTableFlags_Borders)) {
                ImGui::TableSetupColumn("Index");
                ImGui::TableSetupColumn("Industrial risk");
                ImGui::TableSetupColumn("Management risk");
                ImGui::TableSetupColumn("Financial flexibility");
                ImGui::TableSetupColumn("Credibility");
                ImGui::TableSetupColumn("Competitiveness");
                ImGui::TableSetupColumn("Operating risk");
                ImGui::TableSetupColumn("Class");
                ImGui::TableHeadersRow();

                for (std::size_t i {1}; const Instance& instance : training_set.data) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%lu", i);

                    if (i > training_set.training_instance_count) {
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(45, 45, 55, 255));
                    }

                    i++;

                    if (training_set.normalized) {
                        ImGui::TableNextColumn();
                        ImGui::Text("%f", instance.normalized.industrial_risk);

                        ImGui::TableNextColumn();
                        ImGui::Text("%f", instance.normalized.management_risk);

                        ImGui::TableNextColumn();
                        ImGui::Text("%f", instance.normalized.financial_flexibility);

                        ImGui::TableNextColumn();
                        ImGui::Text("%f", instance.normalized.credibility);

                        ImGui::TableNextColumn();
                        ImGui::Text("%f", instance.normalized.competitiveness);

                        ImGui::TableNextColumn();
                        ImGui::Text("%f", instance.normalized.operating_risk);

                        ImGui::TableNextColumn();
                        ImGui::Text("%f", instance.normalized.classification);
                    } else {
                        ImGui::TableNextColumn();
                        ImGui::Text("%s", token_to_string(instance.unnormalized.industrial_risk));

                        ImGui::TableNextColumn();
                        ImGui::Text("%s", token_to_string(instance.unnormalized.management_risk));

                        ImGui::TableNextColumn();
                        ImGui::Text("%s", token_to_string(instance.unnormalized.financial_flexibility));

                        ImGui::TableNextColumn();
                        ImGui::Text("%s", token_to_string(instance.unnormalized.credibility));

                        ImGui::TableNextColumn();
                        ImGui::Text("%s", token_to_string(instance.unnormalized.competitiveness));

                        ImGui::TableNextColumn();
                        ImGui::Text("%s", token_to_string(instance.unnormalized.operating_risk));

                        ImGui::TableNextColumn();
                        ImGui::Text("%s", token_to_string(instance.unnormalized.classification));
                    }
                }

                ImGui::EndTable();
            }
        }

        ImGui::End();
    }

    void open_file_browser() {
        ImGuiFileDialog::Instance()->OpenDialog(
            "FileDialog",
            "Choose File",
            ".txt",
            ".",
            1,
            nullptr,
            ImGuiFileDialogFlags_Modal
        );
    }

    void file_browser(const std::function<void(const std::string&)>& callback) {
        if (ImGuiFileDialog::Instance()->Display("FileDialog", 32, ImVec2(768, 432))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                const auto file_path = ImGuiFileDialog::Instance()->GetFilePathName();
                callback(file_path);
            }

            ImGuiFileDialog::Instance()->Close();
        }
    }
}
