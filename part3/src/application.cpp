#include <utility>
#include <cstddef>

#include <gui_base/gui_base.hpp>

#include "application.hpp"
#include "ui.hpp"

void NnApplication::start() {

}

void NnApplication::update() {
    if (learning) {
        learning = !learn.update(network, nullptr);  // TODO
    }
}

void NnApplication::dispose() {

}
