# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Plugin logic as object libraries. These are linked both into the runtime
# plugin shared objects (below) and into the unit-test binary, so they must
# NOT contain the ABI export symbols (rocjitsu_plugin_metadata /
# rocjitsu_plugin_create / rocjitsu_plugin_destroy) — those live in the
# separate plugin_export.cpp sources compiled only into the shared objects.
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
    plugin_config_resolver.cpp
)
# plugin_loader.cpp resolves plugins with dlopen/dlsym/dlclose/dlerror (via
# util/dynamic_loader.h). On glibc < 2.34 those symbols live in a separate
# libdl, so every target that links this object library must link it too;
# PUBLIC propagates the dependency to all consumers. ${CMAKE_DL_LIBS} is `dl`
# there and empty on newer glibc (where dl* moved into libc), so it is portable.
target_link_libraries(
    rocjitsu_plugin_loader
    PRIVATE flatbuffers
    PUBLIC ${CMAKE_DL_LIBS}
)

# Runtime plugin shared objects are only meaningful on Linux (the interposer
# platform). Each plugin is a `librocjitsu_plugin_<name>.so` that carries only
# its own plugin logic plus the three ABI entry points, which a linker version
# script restricts to be the only exported symbols. It is fully self-contained
# (any simulator code it references is statically bundled) and needs nothing
# from librocjitsu.so (it is not linked against the interposer/simulator image).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Build plugin shared objects next to the interposer (librocjitsu.so is
    # emitted at the top of the build tree) so the standard dynamic-linker
    # search path finds them by soname when the launcher/daemon adds that
    # directory to LD_LIBRARY_PATH or resolves plugins from the interposer's
    # own directory. This mirrors the installed layout, where the plugins and
    # the interposer are co-located in <libdir>.
    set(RJ_PLUGIN_OUTPUT_DIR ${PROJECT_BINARY_DIR})

    # rj_add_plugin_so(<name> <object_lib> <export_source>)
    function(rj_add_plugin_so name object_lib export_src)
        set(target rocjitsu_plugin_${name}_so)
        add_library(${target} MODULE ${export_src})
        target_include_directories(
            ${target}
            PRIVATE
                ${ROCJITSU_INCLUDE_DIR}
                ${ROCJITSU_SRC_DIR}
                ${HSA_INCLUDE_DIR}
                ${GENERATED_DIR}
                # The plugin's object library links util/simdojo PRIVATE, so
                # pull in their public header paths for the plugin's compile.
                $<TARGET_PROPERTY:util,INTERFACE_INCLUDE_DIRECTORIES>
                $<TARGET_PROPERTY:simdojo,INTERFACE_INCLUDE_DIRECTORIES>
        )
        # Link the plugin's own logic. Any simulator code it references is
        # bundled statically through the object library; the plugin does not
        # link librocjitsu.so (the interposer/simulator image), so loading it
        # never pulls the interposer into the host process and it resolves
        # nothing from the simulator at load time.
        target_link_libraries(${target} PRIVATE ${object_lib})
        add_dependencies(${target} flatbuffers_schemas)
        # Hide everything except the ABI exports (which carry an explicit
        # default-visibility attribute via ROCJITSU_DEFINE_PLUGIN). This keeps
        # the plugin's own C++ symbols private to the plugin .so.
        set_target_properties(
            ${target}
            PROPERTIES
                OUTPUT_NAME rocjitsu_plugin_${name}
                LIBRARY_OUTPUT_DIRECTORY ${RJ_PLUGIN_OUTPUT_DIR}
                CXX_VISIBILITY_PRESET hidden
                C_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
        )
        # Restrict the dynamic symbol table to the three ABI entry points. The
        # plugin's own default-visibility C API/flatbuffers symbols that the
        # visibility preset cannot hide are made local by the version script.
        target_link_options(
            ${target}
            PRIVATE
                "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/plugin_export.map"
        )
        set_property(
            TARGET ${target}
            APPEND
            PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/plugin_export.map
        )
        install(TARGETS ${target} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endfunction()

    rj_add_plugin_so(logging rocjitsu_plugin_logging
        ${CMAKE_CURRENT_SOURCE_DIR}/logging/plugin_export.cpp
    )
    rj_add_plugin_so(race rocjitsu_plugin_race_detector
        ${CMAKE_CURRENT_SOURCE_DIR}/race_detector/plugin_export.cpp
    )
endif()
