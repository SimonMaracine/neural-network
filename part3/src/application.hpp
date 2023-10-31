#pragma once

#include <cstddef>

#include <gui_base/gui_base.hpp>

#include "network.hpp"
#include "learn.hpp"

struct NnApplication : public gui_base::GuiApplication {
    NnApplication()
        : gui_base::GuiApplication(1280, 720, "Neural Network") {}

    virtual void start() override;
    virtual void update() override;
    virtual void dispose() override;

    network::Network<6, 1> network;

    Learn<6, 1> learn;

    enum class State {
        Setup,
        ReadyLearning,
        Learning,
        Testing
    } state = State::Setup;
};
