#pragma once

#include <functional>
#include <string>

#include <gui_base/gui_base.hpp>

#include "network.hpp"
#include "learn.hpp"

namespace ui {
    enum class Operation {
        None,
        Start,
        Stop,
        Reinitialize,
        Test,
        Execute,
    };

    bool learning_setup(Learn<18, 1>& learn, network::Network<18, 1>& network);
    Operation learning_process(const Learn<18, 1>& learn);
    void learning_graph(const Learn<18, 1>& learn);
    void training_set(TrainingSet& training_set);
    void open_file_browser();
    void file_browser(const std::function<void(const std::string&)>& callback);
    bool testing(const Learn<18, 1>& learn, const network::Network<18, 1>& network);
    bool executing(const network::Network<18, 1>& network);
}
