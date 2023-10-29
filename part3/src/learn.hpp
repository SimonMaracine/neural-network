#pragma once

#include "network.hpp"
#include "helpers.hpp"

struct Learn {
    double rate = 1.0;
    double epsilon = 0.0;
    unsigned long max_epochs = 0;

    unsigned long epoch_index = 0;
    unsigned long step_index = 0;
    double current_error = 0.0;  // Epoch error

    TrainingSet training_set;

    double* inputs = nullptr;
    double* outputs = nullptr;

    void setup(const neuron::Network<6, 1>& network);
    bool update(neuron::Network<6, 1>& network);
};
