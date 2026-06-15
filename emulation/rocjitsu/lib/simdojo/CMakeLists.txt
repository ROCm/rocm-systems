# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    simdojo
    OBJECT
    src/component.cpp
    src/simulation.cpp
    src/topology.cpp
)
set_target_properties(simdojo PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(simdojo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simdojo PUBLIC util Threads::Threads)
if(MSVC)
    target_compile_options(simdojo PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # Default symbol visibility (no -fvisibility=hidden) so simdojo's
    # symbols export from the shared simulator library (librocjitsu.so),
    # which the KMD shim and plugin modules dynamically link against.
    target_compile_options(
        simdojo
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
endif()
