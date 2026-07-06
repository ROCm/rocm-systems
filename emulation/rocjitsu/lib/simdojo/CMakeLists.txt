# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# On Linux, simdojo is a standalone shared library (libsimdojo.so). The
# simulator core, the LD_PRELOAD interposer, the CLI, and the plugin modules
# all dynamically link it, so the simulation-engine symbols live in a single
# image exported from libsimdojo.so instead of being re-exported by
# librocjitsu.so. On other platforms it stays an OBJECT library aggregated
# into the single librocjitsu shared library (no MSVC dllexport annotations).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_simdojo_kind SHARED)
else()
    set(_simdojo_kind OBJECT)
endif()

add_library(
    simdojo
    ${_simdojo_kind}
    src/component.cpp
    src/simulation.cpp
    src/topology.cpp
)
unset(_simdojo_kind)
set_target_properties(
    simdojo
    PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME simdojo
)
target_include_directories(simdojo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simdojo PUBLIC util Threads::Threads)
if(MSVC)
    target_compile_options(simdojo PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # Default symbol visibility (no -fvisibility=hidden) so simdojo's
    # symbols export from libsimdojo.so, which the simulator core, KMD shim,
    # and plugin modules dynamically link against.
    target_compile_options(simdojo PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# libsimdojo.so sits next to its sibling libraries; resolve them via $ORIGIN.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(simdojo PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
