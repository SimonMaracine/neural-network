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
    void inputs_control();
    void activation_function_plot();

    void update_activation_function();
    void reallocate_inputs(std::size_t new_size, std::size_t old_size);

    neuron::Neuron<float> neuron;

    neuron::InputFunction<float> input_function = neuron::input_function::sum<float>;
    neuron::ActivationFunction<float> activation_function;
    neuron::OutputFunction<float> output_function = neuron::output_function::identity<float>;

    struct {
        float theta = 0.0f;
        float g = 1.0f;
        float a = 1.0f;
    } constants;

    int activation_function_current = 0;
    float* inputs = nullptr;
    std::size_t number_of_inputs = 0;
};
