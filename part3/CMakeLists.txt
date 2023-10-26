cmake_minimum_required(VERSION 3.20)

add_executable(nn3
    "src/application.cpp"
    "src/application.hpp"
    "src/helpers.cpp"
    "src/helpers.hpp"
    "src/learn.cpp"
    "src/learn.hpp"
    "src/main.cpp"
    "src/network.cpp"
    "src/network.hpp"
    "src/ui.cpp"
    "src/ui.hpp"
)

target_link_libraries(nn3 PRIVATE gui_base)

set_compile_options(nn3)
