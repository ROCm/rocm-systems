if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(rccl-net-example LANGUAGES C)
    set(NET_EXAMPLE_STANDALONE ON)
else()
    set(NET_EXAMPLE_STANDALONE OFF)
endif()

set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
)

# Create shared library
add_library(nccl-net-example SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-net-example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
)

# Set output name to match Makefile
set_target_properties(nccl-net-example PROPERTIES
    OUTPUT_NAME "rccl-net-example"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
)

if(NOT NET_EXAMPLE_STANDALONE)
    set_target_properties(nccl-net-example PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
    )
endif()
