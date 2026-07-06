cmake_minimum_required(VERSION 3.20)
project(grove_pim_datamovement CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra)
endif()

# Headless byte-accounting core (C++), Python analysis layer drives it (D-016).
add_library(pim_core src/model.cpp)
target_include_directories(pim_core PUBLIC include)

add_executable(pim src/main.cpp)
target_link_libraries(pim PRIVATE pim_core)

enable_testing()
add_executable(test_pim tests/test_pim.cpp)
target_link_libraries(test_pim PRIVATE pim_core)
add_test(NAME pim_tests COMMAND test_pim)
