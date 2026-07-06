cmake_minimum_required(VERSION 3.20)
project(grove_simt_mini_gpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra)
endif()

# Headless, testable engine core (the gem5/GPGPU-Sim pattern: C++ core, D-016).
add_library(simt_core
    src/isa.cpp
    src/assembler.cpp
    src/core.cpp)
target_include_directories(simt_core PUBLIC include)

# CLI front-end.
add_executable(simt src/main.cpp)
target_link_libraries(simt PRIVATE simt_core)

# Tests (CTest).
enable_testing()
add_executable(test_simt tests/test_simt.cpp)
target_link_libraries(test_simt PRIVATE simt_core)
add_test(NAME simt_tests COMMAND test_simt)
