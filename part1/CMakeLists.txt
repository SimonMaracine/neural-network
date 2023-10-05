cmake_minimum_required(VERSION 3.20)

add_executable(nn
    "src/application.cpp"
    "src/application.hpp"
    "src/main.cpp"
    "src/neuron.hpp"
)

target_link_libraries(nn PRIVATE gui_base)

set_compile_options(nn)
