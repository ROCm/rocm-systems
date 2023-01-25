cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-causal-example LANGUAGES CXX)

if(OMNITRACE_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(${PROJECT_NAME} IN_LIST OMNITRACE_DISABLE_EXAMPLES OR ${_DIR} IN_LIST
                                                             OMNITRACE_DISABLE_EXAMPLES)
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)
if(NOT TARGET omnitrace::omnitrace-user-library)
    find_package(omnitrace REQUIRED COMPONENTS user)
endif()

add_library(causal-interface-library INTERFACE)
target_compile_options(causal-interface-library INTERFACE -g3 -gdwarf-3
                                                          -fno-omit-frame-pointer)
target_link_libraries(causal-interface-library INTERFACE Threads::Threads
                                                         ${CMAKE_DL_LIBS})

omnitrace_causal_example_executable(
    "causal-both"
    SOURCES causal.cpp impl.cpp
    LINK_LIBRARIES causal-interface-library
    DEFINITIONS USE_RNG=1 USE_CPU=1)

omnitrace_causal_example_executable(
    "causal-rng"
    SOURCES causal.cpp impl.cpp
    LINK_LIBRARIES causal-interface-library
    DEFINITIONS USE_RNG=1 USE_CPU=0)

omnitrace_causal_example_executable(
    "causal-cpu"
    SOURCES causal.cpp impl.cpp
    LINK_LIBRARIES causal-interface-library
    DEFINITIONS USE_RNG=0 USE_CPU=1)
