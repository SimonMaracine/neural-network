cmake_minimum_required(VERSION 3.20)

add_executable(nn3b
    "src/application.cpp"
    "src/application.hpp"
    "src/helpers.cpp"
    "src/helpers.hpp"
    "src/learn.hpp"
    "src/main.cpp"
    "src/network.hpp"
    "src/ui.cpp"
    "src/ui.hpp"
)

target_link_libraries(nn3b PRIVATE gui_base)

set_compile_options(nn3b)
