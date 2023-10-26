#include <utility>
#include <cstddef>

#include <gui_base/gui_base.hpp>

#include "application.hpp"
#include "ui.hpp"

void NnApplication::start() {

}

void NnApplication::update() {
    switch (state) {
        case State::Setup:
            if (ui::learning_setup(learn, network)) {
                state = State::Learning;
            }

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
