#include <utility>
#include <cstddef>

#include <gui_base/gui_base.hpp>

#include "application.hpp"
#include "ui.hpp"

void NnApplication::start() {
#if 0
    neuron::Network::HiddenLayers layers;
    layers.layers.push_back(3);
    layers.layers.push_back(3);
    layers.layers.push_back(4);

    network.setup(2, 1, std::move(layers));
#endif
}

void NnApplication::update() {
    ImGui::ShowDemoWindow();

    // network.run();

    ui::build_network(network, &inputs, &n);
    ui::network_controls(network);
    ui::draw_network(network);
    ui::inputs_controls(inputs, n);
}

void NnApplication::dispose() {

}
