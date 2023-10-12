#include <utility>
#include <cstddef>

#include <gui_base/gui_base.hpp>

#include "application.hpp"
#include "ui.hpp"

void NnApplication::start() {

}

void NnApplication::update() {
    if (inputs != nullptr) {
        network.run(inputs, nullptr);
    }

    const bool built = ui::build_network(network, &inputs, &n);
    ui::network_controls(network, built);
    ui::draw_network(network);
    ui::inputs_controls(inputs, n);
}

void NnApplication::dispose() {

}
