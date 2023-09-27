#pragma once

#include <cstddef>

#include <gui_base/gui_base.hpp>

#include "neuron.hpp"

struct NnApplication : public gui_base::GuiApplication {
    NnApplication()
        : gui_base::GuiApplication(1024, 576, "Neural Network") {}

    virtual void start() override;
    virtual void update() override;
    virtual void dispose() override;

    void constants_control();
    void functions_control();
    void neuron_inputs_control();
    void neuron_output();
    void activation_function_plot();

    using Float = double;

    neuron::ActivationFunction<Float> get_activation_function();
    void reallocate_inputs(std::size_t new_size, std::size_t old_size);

    neuron::Neuron<Float> neuron;

    struct {
        Float theta = 0.0f;
        Float g = 1.0f;
        Float a = 1.0f;
    } constants;

    int activation_function_current = 0;

    Float* inputs = nullptr;
    std::size_t number_of_inputs = 0;
};
