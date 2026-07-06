# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_executable(rocjitsu_bin main.cpp)

target_include_directories(
    rocjitsu_bin
    PRIVATE
        ${GENERATED_DIR}
        ${RJ_VERSION_INCLUDE_DIR}
        # librocjitsu.so links util/simdojo PRIVATE, so their public header
        # paths do not propagate to consumers. Pull them in directly for
        # main.cpp's compile (rj_vm_impl.h transitively includes simdojo).
        $<TARGET_PROPERTY:util,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:simdojo,INTERFACE_INCLUDE_DIRECTORIES>
)

# Dynamically link the shared simulator image (librocjitsu_core.so +
# libsimdojo.so) instead of statically bundling the simulator object libraries.
# This keeps a single simulator runtime image in the process: a plugin loaded
# in daemon mode (which links the same libraries) shares the CLI's simulator
# state. The CLI deliberately does not link librocjitsu.so (the LD_PRELOAD
# interposer), so it does not pull the interposer's libc overrides into itself.
target_link_libraries(
    rocjitsu_bin
    PRIVATE rocjitsu_core simdojo Threads::Threads dl
)

set_target_properties(rocjitsu_bin PROPERTIES OUTPUT_NAME rocjitsu)

# librocjitsu.so is now a runtime dependency of the CLI. CMake adds the build
# tree location to the RPATH automatically; for the installed binary, point at
# <prefix>/<libdir> where rocjitsu_shared is installed.
include(GNUInstallDirs)
set_target_properties(
    rocjitsu_bin
    PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)

rj_configure_target(rocjitsu_bin TOOL)

# Install the rocjitsu CLI so it is usable from a packaged dist (the default
# entrypoint for driving emulation when mirage is not present).
install(TARGETS rocjitsu_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
