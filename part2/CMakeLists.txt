cmake_minimum_required(VERSION 3.20)

add_executable(nn2
    "src/application.cpp"
    "src/application.hpp"
    "src/main.cpp"
    "src/network.cpp"
    "src/network.hpp"
)

target_link_libraries(nn2 PRIVATE gui_base)

set_compile_options(nn2)
