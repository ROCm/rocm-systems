cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-parallel-overhead-example LANGUAGES CXX)

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

add_library(parallel-overhead-compile-options INTERFACE)
target_compile_options(parallel-overhead-compile-options INTERFACE -g)

add_executable(parallel-overhead parallel-overhead.cpp)
target_link_libraries(
    parallel-overhead
    PRIVATE Threads::Threads parallel-overhead-compile-options
)

add_executable(parallel-overhead-locks parallel-overhead.cpp)
target_link_libraries(
    parallel-overhead-locks
    PRIVATE Threads::Threads parallel-overhead-compile-options
)
target_compile_definitions(parallel-overhead-locks PRIVATE USE_LOCKS=1)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        TARGETS parallel-overhead parallel-overhead-locks
        DESTINATION bin
        COMPONENT rocprofiler-systems-examples
    )
endif()
