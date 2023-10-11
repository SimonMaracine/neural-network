#pragma once

#include <gui_base/gui_base.hpp>

#include "network.hpp"

namespace ui {
    void draw_network(const neuron::Network& network);
    void build_network(neuron::Network& network);
    void network_controls(neuron::Network& network);
}
