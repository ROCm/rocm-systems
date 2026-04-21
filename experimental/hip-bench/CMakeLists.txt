cmake_minimum_required(VERSION 3.16)
project(hip-bench LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find HIP. We need it because the .hip translation unit must be compiled
# with hipcc and linked against hip::host.
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip REQUIRED CONFIG)

enable_language(HIP)

add_executable(hipbench
    src/main.cpp
    src/kernels.hip
)

set_source_files_properties(src/kernels.hip src/main.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hipbench PRIVATE hip::host)
target_compile_options(hipbench PRIVATE -Wall -Wextra)
