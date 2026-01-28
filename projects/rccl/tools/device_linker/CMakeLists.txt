cmake_minimum_required(VERSION 3.16)
project(device_linker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(device_linker device_linker.cpp)
target_link_libraries(device_linker PRIVATE pthread)

# Enable optimizations for release
target_compile_options(device_linker PRIVATE 
    $<$<CONFIG:Release>:-O3 -march=native>
)
