#include <cstddef>
#include <algorithm>
#include <array>
#include <utility>

#include "ui.hpp"

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

static void draw_input_layer(ImDrawList* list, const neuron::Network& network, float x, float y, float network_height, float neuron_size, unsigned int neuron_color) {
    const float spacing = network_height / static_cast<float>(network.input_neurons + 1);

    for (std::size_t i = 0; i < network.input_neurons; i++) {
        list->AddCircleFilled(ImVec2(x, y + i * spacing + spacing), neuron_size, neuron_color);
    }
}

static float draw_hidden_layers(ImDrawList* list, const neuron::Network& network, float x, float y, float network_height, float neuron_size, unsigned int neuron_color) {
    for (std::size_t i = 0; i < network.hidden_layers.size(); i++) {
        const neuron::Layer& layer = network.hidden_layers[i];

        const float spacing = network_height / static_cast<float>(network.hidden_layers[i].neurons.size() + 1);

        for (std::size_t j = 0; j < layer.neurons.size(); j++) {
            list->AddCircleFilled(ImVec2(x, y + j * spacing + spacing), neuron_size, neuron_color);
        }

        x += 100.0f;
    }

    return x;
}

static void draw_output_layer(ImDrawList* list, const neuron::Network& network, float x, float y, float network_height, float neuron_size, unsigned int neuron_color) {
    const float spacing = network_height / static_cast<float>(network.output_layer.neurons.size() + 1);

    for (std::size_t i = 0; i < network.output_layer.neurons.size(); i++) {
        list->AddCircleFilled(ImVec2(x, y + i * spacing + spacing), neuron_size, neuron_color);
    }
}

void draw_network(const neuron::Network& network) {
    ImGui::Begin("Neural Network");

    ImDrawList* list = ImGui::GetWindowDrawList();
    const ImVec2 canvas = ImGui::GetCursorScreenPos();
    const float OFFSET = 30.0f;
    const float NEURON_SIZE = 20.0f;
    static constexpr auto NEURON_COLOR = IM_COL32(200, 200, 200, 255);

    const float neuron_spacing = 60.0f;
    const float height = static_cast<float>(network_height(network)) * neuron_spacing;

    float x = OFFSET;

    draw_input_layer(list, network, canvas.x + OFFSET + x, canvas.y + OFFSET, height, NEURON_SIZE, NEURON_COLOR);

    x += 100.0f;

    x = draw_hidden_layers(list, network, canvas.x + OFFSET + x, canvas.y + OFFSET, height, NEURON_SIZE, NEURON_COLOR);

    draw_output_layer(list, network, x, canvas.y + OFFSET, height, NEURON_SIZE, NEURON_COLOR);

    ImGui::End();
}

void network_controls(neuron::Network& network) {
    static int input_layer_neurons = 2;
    static int output_layer_neurons = 1;
    static int hidden_layers = 1;
    static std::array<int, 3> hidden_layer_neurons = { 3, 1, 1 };

    ImGui::Begin("Network Controls");

    if (ImGui::InputInt("Input Layer", &input_layer_neurons)) {
        input_layer_neurons = std::max(input_layer_neurons, 1);
    }

    ImGui::Separator();

    if (ImGui::InputInt("Hidden Layers", &hidden_layers)) {
        hidden_layers = std::max(hidden_layers, 1);
        hidden_layers = std::min(hidden_layers, 3);
    }

    for (int i = 0; i < hidden_layers; i++) {
        ImGui::PushID(i);

        if (ImGui::InputInt("##", &hidden_layer_neurons[i])) {
            hidden_layer_neurons[i] = std::max(hidden_layer_neurons[i], 1);
        }

        ImGui::SameLine();

        ImGui::Text("%d", i);

        ImGui::PopID();
    }

    ImGui::Separator();

    if (ImGui::InputInt("Output Layer", &output_layer_neurons)) {
        output_layer_neurons = std::max(output_layer_neurons, 1);
    }

    ImGui::Separator();

    if (ImGui::Button("Apply")) {
        neuron::Network::HiddenLayers layers;

        for (int i = 0; i < hidden_layers; i++) {
            layers.layers.push_back(hidden_layer_neurons[i]);
        }

        network.setup(input_layer_neurons, output_layer_neurons, std::move(layers));
    }

    ImGui::End();
}
