# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-bin-common static library
#
# Shared functionality for bin tools (run, sample, etc.)
#
# ------------------------------------------------------------------------------#

set(PRESET_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/presets")
set(EMBEDDED_PRESET_ENTRIES "constexpr entry presets[] = {\n")

# GLOB is acceptable here: these are data files embedded at configure time,
# not compiled sources. Adding a new preset requires re-running cmake anyway.
file(GLOB PRESET_JSON_FILES "${PRESET_SOURCE_DIR}/*.json")
list(SORT PRESET_JSON_FILES)
foreach(PRESET_FILE ${PRESET_JSON_FILES})
    get_filename_component(PRESET_BASENAME "${PRESET_FILE}" NAME_WE)
    # Skip schema.json — it's for documentation/validation, not runtime
    if("${PRESET_BASENAME}" STREQUAL "schema")
        continue()
    endif()
    file(READ "${PRESET_FILE}" PRESET_CONTENT)
    # Escape for C string literal (same escaping as rocpd_schema)
    string(REPLACE "\\" "\\\\" PRESET_CONTENT "${PRESET_CONTENT}")
    string(REPLACE "\"" "\\\"" PRESET_CONTENT "${PRESET_CONTENT}")
    string(REPLACE "\n" "\\n\"\n\"" PRESET_CONTENT "${PRESET_CONTENT}")
    string(
        APPEND EMBEDDED_PRESET_ENTRIES
        "    {\"${PRESET_BASENAME}\", \"${PRESET_CONTENT}\"},\n"
    )
endforeach()

string(
    APPEND EMBEDDED_PRESET_ENTRIES
    "};\n\n"
    "constexpr auto num_presets = sizeof(presets) / sizeof(presets[0]);\n"
)

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/Templates/embedded_presets.in"
    "${CMAKE_CURRENT_BINARY_DIR}/embedded_presets.hpp"
    @ONLY
)

# ------------------------------------------------------------------------------#

add_library(
    rocprofiler-systems-bin-common
    STATIC
    json_config.cpp
    preset_registry.cpp
    common_utils.cpp
)

target_include_directories(
    rocprofiler-systems-bin-common
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(
    rocprofiler-systems-bin-common
    PUBLIC
        rocprofiler-systems::rocprofiler-systems-json
        rocprofiler-systems::rocprofiler-systems-common-library
)

# Create alias for consistent naming
add_library(
    rocprofiler-systems::rocprofiler-systems-bin-common
    ALIAS rocprofiler-systems-bin-common
)
