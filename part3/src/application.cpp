#include <utility>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <string>

#include <gui_base/gui_base.hpp>

#include "application.hpp"
#include "ui.hpp"

void NnApplication::start() {
    std::srand(std::time(nullptr));
}

void NnApplication::update() {
    ImGui::ShowDemoWindow();

    switch (state) {
        case State::Setup:
            if (ui::learning_setup(learn, network)) {
                state = State::Learning;
            }

            if (learn.training_set.loaded) {
                ui::training_set(learn.training_set);
            }

            ui::file_browser([this](const std::string& file_path) {
                learn.training_set.load(file_path);
            });

            break;
        case State::Learning:
            if (ui::learning_process(learn)) {
                state = State::DoneLearning;
            }

            ui::learning_graph(learn);
            break;
        case State::DoneLearning:
            ui::learning_process(learn);
            ui::learning_graph(learn);
            break;
        case State::Testing:
            break;
        case State::DoneTesting:
            break;
    }

    if (state == State::Learning) {
        if (learn.update(network)) {
            state = State::DoneLearning;
        }
    }
}

void NnApplication::dispose() {

}
