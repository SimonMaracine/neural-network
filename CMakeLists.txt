cmake_minimum_required(VERSION 3.20)

function(set_compile_options TARGET)
    if(UNIX)
        target_compile_options(${TARGET} PRIVATE "-Wall" "-Wextra" "-Wpedantic")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${TARGET} PRIVATE "/W3")
    else()
        message(WARNING "Compiler is not MSVC or platform is not supported, warnings are not enabled")
    endif()

    target_compile_features(${TARGET} PRIVATE cxx_std_20)
    set_target_properties(${TARGET} PROPERTIES CXX_EXTENSIONS OFF)
endfunction()

project("neural_network" LANGUAGES CXX)

add_subdirectory(extern/tiny-gui-base)

add_subdirectory(part1)
add_subdirectory(part2)
add_subdirectory(part3)
