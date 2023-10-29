#pragma once

#include <functional>
#include <string>

#include <gui_base/gui_base.hpp>

#include "network.hpp"
#include "learn.hpp"

namespace ui {
    bool learning_setup(Learn& learn, neuron::Network<6, 1>& network);
    bool learning_process(const Learn& learn);
    void learning_graph(const Learn& learn);
    void training_set(TrainingSet& training_set);
    void open_file_browser();
    void file_browser(const std::function<void(const std::string&)>& callback);
}
