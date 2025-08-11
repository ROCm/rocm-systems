cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-trace-time-window-example LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Debug")

add_executable(trace-time-window trace-time-window.cpp)
target_compile_options(trace-time-window PRIVATE ${_FLAGS})

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        TARGETS trace-time-window
        DESTINATION bin
        COMPONENT rocprofiler-systems-examples
    )
endif()
