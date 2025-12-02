cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(rocprofiler-systems-rewrite-caller-example LANGUAGES CXX)

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

add_executable(rewrite-caller rewrite-caller.cpp)
target_compile_options(rewrite-caller PRIVATE ${_FLAGS})

set_target_properties(
    rewrite-caller
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rewrite-caller"
)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        TARGETS rewrite-caller
        DESTINATION share/rocprofiler-systems/samples/rewrite-caller
        COMPONENT rocprofiler-systems-samples
    )
endif()
