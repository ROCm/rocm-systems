cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-causal-example LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)
if(NOT TARGET rocprofiler-systems::rocprofiler-systems-user-library)
    find_package(rocprofiler-systems REQUIRED COMPONENTS user)
endif()

add_library(causal-interface-library INTERFACE)
target_compile_options(
    causal-interface-library
    INTERFACE -g3 -gdwarf-3 -fno-omit-frame-pointer
)
target_link_libraries(
    causal-interface-library
    INTERFACE Threads::Threads ${CMAKE_DL_LIBS}
)

rocprofiler_systems_causal_example_executable(
    "causal-both"
    SOURCES causal.cpp impl.cpp
    LINK_LIBRARIES causal-interface-library
    DEFINITIONS USE_RNG=1 USE_CPU=1
)

rocprofiler_systems_causal_example_executable(
    "causal-rng"
    SOURCES causal.cpp impl.cpp
    LINK_LIBRARIES causal-interface-library
    DEFINITIONS USE_RNG=1 USE_CPU=0
)

rocprofiler_systems_causal_example_executable(
    "causal-cpu"
    SOURCES causal.cpp impl.cpp
    LINK_LIBRARIES causal-interface-library
    DEFINITIONS USE_RNG=0 USE_CPU=1
)
