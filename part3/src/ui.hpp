#pragma once

#include <gui_base/gui_base.hpp>

#include "network.hpp"

namespace ui {
    bool build_network(neuron::Network& network, double** inputs, std::size_t* n);
    void learning_controls();
}
