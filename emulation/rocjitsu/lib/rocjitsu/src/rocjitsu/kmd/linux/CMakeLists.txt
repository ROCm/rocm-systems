# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

rj_add_object_library(rocjitsu_kmd
    events.cpp
    remote_driver.cpp
    simulated_driver.cpp
    sysfs.cpp
    transport.cpp
)
target_include_directories(rocjitsu_kmd PRIVATE ${GENERATED_DIR})

add_library(rocjitsu_kmd_shim SHARED interposer.cpp)
target_include_directories(
    rocjitsu_kmd_shim
    PRIVATE
        ${ROCJITSU_INCLUDE_DIR}
        ${ROCJITSU_SRC_DIR}
        ${HSA_INCLUDE_DIR}
        ${GENERATED_DIR}
        # librocjitsu.so keeps util/simdojo PRIVATE, so pull in their public
        # header paths here for the interposer's compile.
        $<TARGET_PROPERTY:util,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:simdojo,INTERFACE_INCLUDE_DIRECTORIES>
)
# The shim is a thin LD_PRELOAD interposer: it only provides interposed
# libc entry points (ioctl/open/...) and delegates all emulation to the
# shared simulator library. Rather than statically bundling the simulator
# (which produced a ~60MB shim with its own private copy of every
# singleton), it dynamically links librocjitsu.so so there is exactly one
# copy of the simulator in the process, shared with the plugin modules.
target_link_libraries(
    rocjitsu_kmd_shim
    PRIVATE rocjitsu_shared Threads::Threads dl
)
target_compile_definitions(rocjitsu_kmd_shim PRIVATE _GNU_SOURCE)
target_compile_options(
    rocjitsu_kmd_shim
    PRIVATE -Wall -Wextra -Wpedantic -Werror
)
set_target_properties(
    rocjitsu_kmd_shim
    PROPERTIES
        OUTPUT_NAME "rocjitsu_kmd"
        # Locate librocjitsu.so next to the shim when installed/extracted
        # (mirage co-locates both under <cache>/emulator/rocjitsu/). The
        # build tree gets its rpath automatically from CMake.
        INSTALL_RPATH "$ORIGIN"
)
