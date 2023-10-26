#include "network.hpp"
#include "learn.hpp"

void Learn::setup(const neuron::Network& network) {
    outputs = new double[network.output_layer.neurons.size()];
}

bool Learn::update(neuron::Network& network) {
    network.run(inputs, outputs);



    // TODO compute error and do backpropagation; return true when it should stop

    return false;
}
