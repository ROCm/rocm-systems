# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(simdojo_headers INTERFACE)
target_include_directories(
    simdojo_headers
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(simdojo_headers INTERFACE util Threads::Threads)

add_library(
    simdojo
    OBJECT
    src/component.cpp
    src/simulation.cpp
    src/tag_array.cpp
    src/topology.cpp
)
set_target_properties(simdojo PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(simdojo PUBLIC simdojo_headers)
if(MSVC)
    target_compile_options(simdojo PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
        simdojo
        PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden
    )
endif()
