# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Define an OBJECT library with standard include dirs and flags for
# rocjitsu sub-components. ROCJITSU_INCLUDE_DIR and ROCJITSU_SRC_DIR
# must be set before including this module.
#
# Usage: rj_add_object_library(<name> <sources...>)
function(rj_add_object_library name)
    add_library(${name} OBJECT ${ARGN})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(
        ${name}
        PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR} ${HSA_INCLUDE_DIR}
    )
    target_link_libraries(${name} PRIVATE util simdojo)
    # NOTE: we intentionally do *not* set -fvisibility=hidden here. These
    # object libraries are aggregated into the single shared simulator
    # library (librocjitsu.so), and the KMD shim (librocjitsu_kmd.so) and
    # the plugin modules dynamically link against it instead of statically
    # bundling the simulator. For that to work librocjitsu.so must export
    # the simulator's C++ symbols, so the object code is compiled with the
    # default (visible) symbol visibility. Plugins still restrict their own
    # exported ABI to the three plugin entry points via a linker version
    # script (plugin_export.map).
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(
            ${name}
            PRIVATE -Wall -Wextra -Wpedantic -Werror
        )
    endif()
endfunction()
