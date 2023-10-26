#pragma once

#include <gui_base/gui_base.hpp>

#include "network.hpp"
#include "learn.hpp"

namespace ui {
    bool learning_setup(Learn& learn, neuron::Network& network);
    bool learning_process(const Learn& learn);
    void learning_graph(const Learn& learn);
}
