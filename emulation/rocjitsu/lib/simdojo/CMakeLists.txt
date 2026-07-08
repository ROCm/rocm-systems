# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    simdojo
    SHARED
    src/component.cpp
    src/simulation.cpp
    src/topology.cpp
)
set_target_properties(
    simdojo
    PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME simdojo
)
target_include_directories(simdojo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simdojo PUBLIC util Threads::Threads)
if(MSVC)
    # simdojo's public classes are annotated with __declspec(dllexport/dllimport)
    # via SIMDOJO_EXPORT (include/simdojo/export.h), so simdojo.dll exports only
    # its declared ABI rather than every symbol. C4251 fires because those
    # exported classes hold STL data members (std::vector, std::string, ...); it
    # is benign here since every consumer links the same toolchain's runtime
    # within this monorepo, so silence it to keep /WX green.
    target_compile_options(simdojo PRIVATE /W4 /WX /wd4251)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # Hide every symbol by default; only types annotated with SIMDOJO_EXPORT
    # (include/simdojo/export.h) are exported from libsimdojo.so. This keeps the
    # ABI surface minimal and mirrors the __declspec model used on Windows. The
    # simulator core, KMD shim, and plugin modules dynamically link
    # libsimdojo.so and resolve exactly those annotated symbols.
    target_compile_options(
        simdojo
        PRIVATE -Wall -Wextra -Wpedantic -Werror -fvisibility=hidden
    )
endif()

# libsimdojo.so sits next to its sibling libraries; resolve them via $ORIGIN.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(simdojo PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
