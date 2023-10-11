#pragma once

#include <cstddef>

#include <gui_base/gui_base.hpp>

#include "network.hpp"

struct NnApplication : public gui_base::GuiApplication {
    NnApplication()
        : gui_base::GuiApplication(1280, 720, "Neural Network") {}

    virtual void start() override;
    virtual void update() override;
    virtual void dispose() override;

    neuron::Network network;
    double* inputs = nullptr;
    std::size_t n = 0;
};
