#include <utility>

#include <gui_base/gui_base.hpp>

#include "application.hpp"
#include "ui.hpp"

void NnApplication::start() {
    neuron::Network::HiddenLayers layers;
    layers.layers.push_back(3);
    layers.layers.push_back(3);
    layers.layers.push_back(4);

    network.setup(2, 1, std::move(layers));
}

void NnApplication::update() {
    ImGui::ShowDemoWindow();

    network_controls(network);

    // network.run();


    draw_network(network);
}

void NnApplication::dispose() {

}
