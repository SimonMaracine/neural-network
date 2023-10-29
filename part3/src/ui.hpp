#pragma once

#include <functional>
#include <string>

#include <gui_base/gui_base.hpp>

#include "network.hpp"
#include "learn.hpp"

namespace ui {
    bool learning_setup(Learn<6, 1>& learn, neuron::Network<6, 1>& network);
    int learning_process(const Learn<6, 1>& learn);
    void learning_graph(const Learn<6, 1>& learn);
    void training_set(TrainingSet& training_set);
    void open_file_browser();
    void file_browser(const std::function<void(const std::string&)>& callback);
}
