cmake_minimum_required(VERSION 3.20)

project("neural_network" LANGUAGES CXX)

add_subdirectory(extern/tiny-gui-base)

add_executable(nn
    "src/application.cpp"
    "src/application.hpp"
    "src/main.cpp"
    "src/neuron.hpp"
)

target_link_libraries(nn PRIVATE gui_base)

if(UNIX)
    target_compile_options(nn PRIVATE "-Wall" "-Wextra" "-Wpedantic")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(nn PRIVATE "/W3")
else()
    message(WARNING "Compiler is not MSVC or platform is not supported, warnings are not enabled")
endif()

target_compile_features(nn PRIVATE cxx_std_20)
set_target_properties(nn PROPERTIES CXX_EXTENSIONS OFF)
