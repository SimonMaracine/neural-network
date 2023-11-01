#include <utility>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <string>

#include <gui_base/gui_base.hpp>
#include <implot.h>

#include "application.hpp"
#include "ui.hpp"

void NnApplication::start() {
    std::srand(std::time(nullptr));

    ImPlot::CreateContext();
}

void NnApplication::update() {
    switch (state) {
        case State::Setup:
            if (ui::learning_setup(learn, network)) {
                state = State::ReadyLearning;
            }

            if (learn.training_set.loaded) {
                ui::training_set(learn.training_set);
            }

            ui::file_browser([this](const std::string& file_path) {
                learn.training_set.load(file_path, 30.0f);
            });

            break;
        case State::ReadyLearning: {
            ui::learning_setup(learn, network);

            const auto result = ui::learning_process(learn);

            if (result == ui::Operation::Start) {
                learn.reset();
                learn.start(network);
                state = State::Learning;
            } else if (result == ui::Operation::Reinitialize) {
                network.initialize_neurons();
            } else if (result == ui::Operation::Test) {
                state = State::Testing;
            } else if (result == ui::Operation::Execute) {
                state = State::Executing;
            }

            ui::learning_graph(learn);

            break;
        }
        case State::Learning: {
            const auto result = ui::learning_process(learn);

            if (result == ui::Operation::Stop) {
                learn.stop();
                state = State::ReadyLearning;
            }

            ui::learning_graph(learn);

            if (!learn.is_running()) {
                state = State::ReadyLearning;
            }

            break;
        }
        case State::Testing:
            if (ui::testing(learn, network)) {
                state = State::ReadyLearning;
            }

            break;
        case State::Executing:
            if (ui::executing(network)) {
                state = State::ReadyLearning;
            }

            break;
    }

    std::string title {"Neural Network"};

    switch (state) {
        case State::Setup:
            title += " - Setup";
            break;
        case State::ReadyLearning:
            title += " - ReadyLearning";
            break;
        case State::Learning:
            title += " - Learning";
            break;
        case State::Testing:
            title += " - Testing";
            break;
        case State::Executing:
            title += " - Executing";
            break;
    }

    set_title(title.c_str());
}

void NnApplication::dispose() {
    learn.stop();
    ImPlot::DestroyContext();
}
