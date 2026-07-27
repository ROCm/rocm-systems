# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_library(rocjitsu_cli_launch_preload STATIC launch_preload.cpp)
target_include_directories(
    rocjitsu_cli_launch_preload
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(rocjitsu_cli_launch_preload PRIVATE _GNU_SOURCE)
target_link_libraries(rocjitsu_cli_launch_preload PUBLIC ${CMAKE_DL_LIBS})
rj_configure_target(rocjitsu_cli_launch_preload TOOL)

add_executable(rocjitsu_bin main.cpp)

target_include_directories(
    rocjitsu_bin
    PRIVATE ${GENERATED_DIR} ${RJ_VERSION_INCLUDE_DIR}
)

target_link_libraries(
    rocjitsu_bin
    PRIVATE
        rocjitsu_kmd
        rocjitsu_analysis
        rocjitsu_code
        rocjitsu_isa
        rocjitsu_isa_registry
        simdojo
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        # Plugin loader: rj_vm_load_plugins() dlopen()s plugin modules in
        # daemon mode.
        rocjitsu_plugin_loader
        rocjitsu_vm_risc_v
        rocjitsu_config
        rocjitsu_dbt_config
        util
        flatbuffers
        Threads::Threads
        rocjitsu_cli_launch_preload
)

set_target_properties(rocjitsu_bin PROPERTIES OUTPUT_NAME rocjitsu)

rj_configure_target(rocjitsu_bin TOOL)

# Install the rocjitsu CLI so it is usable from a packaged dist (the default
# entrypoint for driving emulation when mirage is not present).
include(GNUInstallDirs)
install(TARGETS rocjitsu_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
