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
    ImGui::ShowDemoWindow();

    switch (state) {
        case State::Setup:
            if (ui::learning_setup(learn, network)) {
                learn.start(network);
                state = State::Learning;
            }

            if (learn.training_set.loaded) {
                ui::training_set(learn.training_set);
            }

            ui::file_browser([this](const std::string& file_path) {
                learn.training_set.load(file_path);  // TODO check return
            });

            break;
        case State::Learning: {
            const int result = ui::learning_process(learn);

            if (result < 0) {
                learn.stop();
                state = State::DoneLearning;
            }

            ui::learning_graph(learn);

            if (!learn.is_running()) {
                state = State::DoneLearning;
            }

            break;
        }
        case State::DoneLearning: {
            const int result = ui::learning_process(learn);

            if (result > 0) {
                learn.reset();
                learn.start(network);
                state = State::Learning;
            }

            ui::learning_graph(learn);

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
