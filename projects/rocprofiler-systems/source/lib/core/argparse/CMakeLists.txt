# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# TODO(seam): promote to `add_library(rocprofsys-argparse OBJECT ...)` so the
# timemory dependency is link-localised. Deferred: this layer's headers no
# longer leak tim:: (verified by validation gate), but argparse.cpp's
# add_group_arguments / add_extended_arguments still bind tim::vsettings
# directly. Until the vsettings-driven path is lifted via a
# settings_to_descriptor adapter (separate follow-up), an OBJECT library
# here can't actually enforce "no tim in core-library" — it would only
# guard the small descriptor subset already covered by the source-level
# detail/parser_engine.hpp boundary.

target_sources(
    rocprofiler-systems-core-library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/core_flags.cpp
        ${CMAKE_CURRENT_LIST_DIR}/core_flags.hpp
        ${CMAKE_CURRENT_LIST_DIR}/detail/parser_engine.hpp
        ${CMAKE_CURRENT_LIST_DIR}/flag_descriptor.hpp
        ${CMAKE_CURRENT_LIST_DIR}/interpreter.hpp
        ${CMAKE_CURRENT_LIST_DIR}/interpreter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/parsed_values.cpp
        ${CMAKE_CURRENT_LIST_DIR}/parsed_values.hpp
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
