cmake_minimum_required(VERSION 3.16)
project(neural_network)

set (CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(network src/main.cpp)

target_link_libraries(network OpenMP::OpenMP_CXX)

target_compile_options(network PRIVATE -O3 -fopenmp)

add_subdirectory(src)
