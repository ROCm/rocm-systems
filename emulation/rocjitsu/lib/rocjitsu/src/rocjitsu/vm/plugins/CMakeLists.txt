# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Plugin logic as object libraries. These are linked both into the runtime
# plugin shared objects (below) and into the unit-test binary, so they must
# NOT contain the ABI export symbols (rocjitsu_plugin_metadata /
# rocjitsu_plugin_create) — those live in the separate plugin_export.cpp
# sources compiled only into the shared objects.
rj_add_object_library(
    rocjitsu_plugin_logging
    logging/plugin.cpp
)

rj_add_object_library(
    rocjitsu_plugin_race_detector
    race_detector/plugin.cpp
    race_detector/core/race_detector.cpp
    race_detector/core/wave_race_state.cpp
)

# Host-side loader: discovers and dlopen()s plugin shared objects named in the
# config file. Linked into the interposer and main shared library.
rj_add_object_library(
    rocjitsu_plugin_loader
    plugin_loader.cpp
)
target_link_libraries(rocjitsu_plugin_loader PRIVATE flatbuffers)

# Runtime plugin shared objects are only meaningful on Linux (the interposer
# platform). Each plugin is a self-contained `librocjitsu_plugin_<name>.so`:
# host symbols are hidden, so the plugin statically bundles the simulator code
# it references and exports only the two ABI entry points.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Same object-library closure as the main shared library, minus the plugin
    # object libraries themselves (each plugin links only its own logic).
    set(RJ_PLUGIN_RUNTIME_LIBS
        rocjitsu_analysis
        rocjitsu_code
        rocjitsu_isa
        rocjitsu_isa_cdna1
        rocjitsu_isa_cdna2
        rocjitsu_isa_cdna3
        rocjitsu_isa_cdna4
        rocjitsu_isa_rdna1
        rocjitsu_isa_rdna2
        rocjitsu_isa_rdna3
        rocjitsu_isa_rdna3_5
        rocjitsu_isa_rdna4
        rocjitsu_isa_gfx1250
        rocjitsu_isa_risc_v
        simdojo
        rocjitsu_kmd
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        rocjitsu_vm_risc_v
        rocjitsu_config
        util
        flatbuffers
        Threads::Threads
    )

    # Build plugin shared objects next to the interposer so the standard
    # dynamic-linker search path finds them when the launcher adds that
    # directory to LD_LIBRARY_PATH.
    set(RJ_PLUGIN_OUTPUT_DIR
        ${PROJECT_BINARY_DIR}/lib/rocjitsu/src/rocjitsu/kmd/linux
    )

    # rj_add_plugin_so(<name> <object_lib> <export_source>)
    function(rj_add_plugin_so name object_lib export_src)
        set(target rocjitsu_plugin_${name}_so)
        add_library(${target} MODULE ${export_src})
        target_include_directories(
            ${target}
            PRIVATE ${ROCJITSU_INCLUDE_DIR} ${ROCJITSU_SRC_DIR} ${HSA_INCLUDE_DIR}
                    ${GENERATED_DIR}
        )
        target_link_libraries(
            ${target}
            PRIVATE ${object_lib} ${RJ_PLUGIN_RUNTIME_LIBS}
        )
        add_dependencies(${target} flatbuffers_schemas)
        set_target_properties(
            ${target}
            PROPERTIES
                OUTPUT_NAME rocjitsu_plugin_${name}
                LIBRARY_OUTPUT_DIRECTORY ${RJ_PLUGIN_OUTPUT_DIR}
        )
        # The interposer must exist alongside the plugins at runtime.
        add_dependencies(rocjitsu_kmd_shim ${target})
        install(TARGETS ${target} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endfunction()

    rj_add_plugin_so(logging rocjitsu_plugin_logging
        ${CMAKE_CURRENT_SOURCE_DIR}/logging/plugin_export.cpp
    )
    rj_add_plugin_so(race rocjitsu_plugin_race_detector
        ${CMAKE_CURRENT_SOURCE_DIR}/race_detector/plugin_export.cpp
    )
endif()
