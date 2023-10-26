#pragma once

#include "network.hpp"

struct Learn {
    double rate = 1.0f;
    double epsilon = 0.0f;
    unsigned long epoch_index = 0;
    unsigned long step_index = 0;

    double* outputs = nullptr;

    void setup(const neuron::Network& network);
    bool update(neuron::Network& network, double* inputs);
};
