#pragma once

#include <gui_base/gui_base.hpp>

#include "neuron.hpp"

struct NnApplication : public gui_base::GuiApplication {
    virtual void start() override;
    virtual void update() override;
    virtual void dispose() override;

    neuron::Neuron<float> neuron;

    neuron::InputFunction<float> input_function = neuron::input_function::sum<float>;
    neuron::ActivationFunction<float> activation_function = neuron::activation_function::heaviside<float>;
    neuron::OutputFunction<float> output_function = neuron::output_function::identity<float>;

    struct {
        float theta = 0.0f;
        float g = 1.0f;
        float a = 1.0f;
    } constants;
};
