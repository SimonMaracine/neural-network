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
                if (learn.training_set.load(file_path)) {
                    learn.training_set.set_testing(20.0f);
                }
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
            break;
        case State::DoneTesting:
            break;
    }
}

void NnApplication::dispose() {
    learn.reset();
    ImPlot::DestroyContext();
}
