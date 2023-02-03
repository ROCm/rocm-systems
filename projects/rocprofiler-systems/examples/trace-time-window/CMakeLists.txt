cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(omnitrace-trace-time-window-example LANGUAGES CXX)

if(OMNITRACE_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(${PROJECT_NAME} IN_LIST OMNITRACE_DISABLE_EXAMPLES OR ${_DIR} IN_LIST
                                                             OMNITRACE_DISABLE_EXAMPLES)
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "Debug")

add_executable(trace-time-window trace-time-window.cpp)
target_compile_options(trace-time-window PRIVATE ${_FLAGS})

if(OMNITRACE_INSTALL_EXAMPLES)
    install(
        TARGETS trace-time-window
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()
