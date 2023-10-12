#pragma once

#include <gui_base/gui_base.hpp>

#include "network.hpp"

namespace ui {
    void draw_network(const neuron::Network& network);
    bool build_network(neuron::Network& network, double** inputs, std::size_t* n);
    void network_controls(neuron::Network& network, bool reset);
    void inputs_controls(double* inputs, std::size_t n);
}
