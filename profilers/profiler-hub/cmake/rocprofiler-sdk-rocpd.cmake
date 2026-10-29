# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# versions.yml is the index of schema SQL. Configure reads it, writes one header
# per listed file, and writes schema/embedded_schemas.hpp (the include list and
# k_embedded_schemas). The struct and find_embedded_schema() live in
# embedded_schema.hpp.

set(ROCPD_SCHEMA_FILE_KEYS
    rocpd_metadata
    rocpd_tables
    rocpd_views
    rocpd_indexes
    rocpd_data_views
    rocpd_summary_views
)

# versions.yml key -> symbol used inside each version namespace
set(rocpd_symbol_rocpd_metadata ROCPD_METADATA)
set(rocpd_symbol_rocpd_tables ROCPD_TABLES)
set(rocpd_symbol_rocpd_views ROCPD_VIEWS)
set(rocpd_symbol_rocpd_indexes ROCPD_INDEXES)
set(rocpd_symbol_rocpd_data_views DATA_VIEWS)
set(rocpd_symbol_rocpd_summary_views SUMMARY_VIEWS)

# ----------------------------------------------------------------------------------------#
#
# Parse versions.yml. The file is a flat list; comments (the schema history) are dropped
# before matching so they are not treated as entries.
#
# Sets, in the caller:
#   ROCPD_SCHEMA_VERSIONS                 list of "3.0.0;3.0.1;..."
#   ROCPD_SCHEMA_<key>_<maj>_<min>_<pat>  path relative to the schema root
#
# ----------------------------------------------------------------------------------------#
function(ROCPD_PARSE_VERSIONS_YML YML_FILE)
    file(READ "${YML_FILE}" _yml)
    string(REGEX REPLACE "(^|\n)[ \t]*#[^\n]*" "\\1" _yml "${_yml}")
    string(REPLACE "\n    - version:" "\n@@BLOCK@@- version:" _yml "${_yml}")
    string(REPLACE "@@BLOCK@@" ";" _blocks "${_yml}")

    set(_versions "")
    foreach(_block IN LISTS _blocks)
        if(NOT _block MATCHES "version: \"([0-9]+)\\.([0-9]+)\\.([0-9]+)\"")
            continue()
        endif()
        set(_major "${CMAKE_MATCH_1}")
        set(_minor "${CMAKE_MATCH_2}")
        set(_patch "${CMAKE_MATCH_3}")
        set(_ver "${_major}.${_minor}.${_patch}")
        set(_tag "${_major}_${_minor}_${_patch}")

        foreach(_key IN LISTS ROCPD_SCHEMA_FILE_KEYS)
            if(NOT _block MATCHES "${_key}:[ \t]+([^ \t\r\n]+)")
                message(
                    FATAL_ERROR
                    "versions.yml entry ${_ver} is missing ${_key} (${YML_FILE})"
                )
            endif()
            set(_rel "${CMAKE_MATCH_1}")
            set(ROCPD_SCHEMA_${_key}_${_tag} "${_rel}" PARENT_SCOPE)
        endforeach()
        list(APPEND _versions "${_ver}")
    endforeach()

    if(NOT _versions)
        message(FATAL_ERROR "No schema versions found in ${YML_FILE}")
    endif()
    set(ROCPD_SCHEMA_VERSIONS "${_versions}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------------------#
#
# Turn one .sql file into a header of string literals under schema/versions/<version>/.
#
# ----------------------------------------------------------------------------------------#
function(
    ROCPD_CONFIGURE_ONE_SCHEMA_FILE
    SCHEMA_ROOT
    REL_PATH
    SCHEMA_NS
    SCHEMA_VERSION
    SYMBOL_BASE
    OUT_DIR
)
    set(_sql_path "${SCHEMA_ROOT}/${REL_PATH}")
    if(NOT EXISTS "${_sql_path}")
        message(FATAL_ERROR "Schema file not found: ${_sql_path}")
    endif()

    file(READ "${_sql_path}" SQL_CONTENT)
    string(REPLACE "\\" "\\\\" SQL_CONTENT "${SQL_CONTENT}")
    string(REPLACE "\"" "\\\"" SQL_CONTENT "${SQL_CONTENT}")
    string(REPLACE "\n" "\\n\"\n\"" SQL_CONTENT "${SQL_CONTENT}")

    get_filename_component(_name "${REL_PATH}" NAME_WE)
    set(SCHEMA_FILE "${REL_PATH}")
    set(SCHEMA_NAME_UPPER "${SYMBOL_BASE}")
    set(TEMPLATE_FILE "${SQL_SCHEMA_DIR}/rocpd_schema.in")

    configure_file("${TEMPLATE_FILE}" "${OUT_DIR}/${_name}.hpp" @ONLY)
endfunction()

# ----------------------------------------------------------------------------------------#
#
# Standalone fallback. Not used when the monorepo tree or ROCPD_SCHEMA_ROOT is present.
#
# ----------------------------------------------------------------------------------------#
function(ROCPD_CLONE_ROCPD_SCHEMA_FILES OUTPUT_SCHEMA_DIR)
    find_package(Git REQUIRED)

    set(CLONE_DIR "${PROJECT_BINARY_DIR}/external/rocprofiler-sdk-rocpd")
    set(SCHEMA_DIR "${CLONE_DIR}/${ROCPD_SCHEMA_SDK_SUBDIR}")

    if(NOT EXISTS "${SCHEMA_DIR}/versions.yml")
        if(EXISTS "${CLONE_DIR}")
            file(REMOVE_RECURSE "${CLONE_DIR}")
        endif()

        message(
            STATUS
            "[profiler-hub] Cloning rocprofiler-sdk-rocpd schema from ${ROCPD_SCHEMA_GIT_URL} @ ${ROCPD_SCHEMA_GIT_BRANCH}"
        )

        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone --depth 1 --filter=blob:none --sparse
                --branch ${ROCPD_SCHEMA_GIT_BRANCH} ${ROCPD_SCHEMA_GIT_URL}
                ${CLONE_DIR}
            RESULT_VARIABLE RESULT
        )

        if(RESULT EQUAL 0)
            execute_process(
                COMMAND
                    ${GIT_EXECUTABLE} sparse-checkout set
                    ${ROCPD_SCHEMA_SDK_SUBDIR}
                WORKING_DIRECTORY ${CLONE_DIR}
                RESULT_VARIABLE RESULT
            )
        endif()

        if(NOT RESULT EQUAL 0 OR NOT EXISTS "${SCHEMA_DIR}/versions.yml")
            message(
                FATAL_ERROR
                "[profiler-hub] Failed to obtain RocPD schema files. "
                "Set ROCPD_SCHEMA_ROOT to the rocprofiler-sdk-rocpd directory "
                "(the one that contains versions.yml)."
            )
        endif()
    endif()

    message(
        STATUS
        "[profiler-hub] Using cloned rocprofiler-sdk-rocpd schema at ${SCHEMA_DIR}"
    )
    set(${OUTPUT_SCHEMA_DIR} "${SCHEMA_DIR}" PARENT_SCOPE)
endfunction()

set(ROCPD_SCHEMA_GIT_URL
    "https://github.com/ROCm/rocm-systems.git"
    CACHE STRING
    "Git repository to clone for rocprofiler-sdk-rocpd schema files (standalone builds only)"
)
set(ROCPD_SCHEMA_GIT_BRANCH
    "develop"
    CACHE STRING
    "Git branch/tag to clone for rocprofiler-sdk-rocpd schema files (standalone builds only)"
)
set(ROCPD_SCHEMA_SDK_SUBDIR
    "projects/rocprofiler-sdk/source/share/rocprofiler-sdk-rocpd"
    CACHE STRING
    "Path, within the cloned repo, of the directory that contains versions.yml"
)

# This file lives in profilers/profiler-hub/cmake, so the monorepo schema root is three
# levels up.
set(_ROCPD_IN_TREE_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../projects/rocprofiler-sdk/source/share/rocprofiler-sdk-rocpd"
)
cmake_path(ABSOLUTE_PATH _ROCPD_IN_TREE_ROOT NORMALIZE)

set(ROCPD_SCHEMA_ROOT
    ""
    CACHE PATH
    "RocPD schema directory containing versions.yml. Used only at configure time. Empty uses the in-tree SDK path, then a git clone for standalone builds."
)

if(ROCPD_SCHEMA_ROOT AND EXISTS "${ROCPD_SCHEMA_ROOT}/versions.yml")
    set(_ROCPD_SCHEMA_ROOT "${ROCPD_SCHEMA_ROOT}")
elseif(EXISTS "${_ROCPD_IN_TREE_ROOT}/versions.yml")
    set(_ROCPD_SCHEMA_ROOT "${_ROCPD_IN_TREE_ROOT}")
else()
    rocpd_clone_rocpd_schema_files(_ROCPD_SCHEMA_ROOT)
endif()

set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_ROCPD_SCHEMA_ROOT}/versions.yml"
)

rocpd_parse_versions_yml("${_ROCPD_SCHEMA_ROOT}/versions.yml")

set(ROCPD_REGISTRY_INCLUDES "")
set(ROCPD_REGISTRY_ENTRIES "")

# Previous configures wrote schema/v3_0_* or unversioned schema/*.hpp from a glob.
file(GLOB _ROCPD_OLD_NS_DIRS "${SQL_SCHEMA_BINARY_DIR}/schema/v[0-9]*")
if(_ROCPD_OLD_NS_DIRS)
    file(REMOVE_RECURSE ${_ROCPD_OLD_NS_DIRS})
endif()
foreach(
    _stale
    rocpd_metadata
    rocpd_tables
    rocpd_views
    rocpd_indexes
    data_views
    summary_views
    marker_views
)
    file(REMOVE "${SQL_SCHEMA_BINARY_DIR}/schema/${_stale}.hpp")
endforeach()

foreach(_ver IN LISTS ROCPD_SCHEMA_VERSIONS)
    if(NOT _ver MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
        message(FATAL_ERROR "Bad schema version '${_ver}'")
    endif()
    set(_major "${CMAKE_MATCH_1}")
    set(_minor "${CMAKE_MATCH_2}")
    set(_patch "${CMAKE_MATCH_3}")
    set(_tag "${_major}_${_minor}_${_patch}")
    set(_ns "v${_tag}")
    # Every version, including one whose SQL lives at the package root, is written
    # to schema/versions/<version>/. The path in versions.yml is only the read path.
    set(_out "${SQL_SCHEMA_BINARY_DIR}/schema/versions/${_ver}")
    file(MAKE_DIRECTORY "${_out}")

    foreach(_key IN LISTS ROCPD_SCHEMA_FILE_KEYS)
        set(_rel "${ROCPD_SCHEMA_${_key}_${_tag}}")
        cmake_path(GET _rel STEM _hdr)
        rocpd_configure_one_schema_file(
            "${_ROCPD_SCHEMA_ROOT}"
            "${_rel}"
            "${_ns}"
            "${_ver}"
            "${rocpd_symbol_${_key}}"
            "${_out}"
        )
        string(
            APPEND ROCPD_REGISTRY_INCLUDES
            "#include \"schema/versions/${_ver}/${_hdr}.hpp\"\n"
        )
    endforeach()

    string(
        APPEND ROCPD_REGISTRY_ENTRIES
        "    { \"${_ver}\", ${_major}, ${_minor}, ${_patch},\n"
        "      ${_ns}::ROCPD_TABLES_SQL, ${_ns}::ROCPD_VIEWS_SQL, ${_ns}::DATA_VIEWS_SQL,\n"
        "      ${_ns}::SUMMARY_VIEWS_SQL, ${_ns}::ROCPD_METADATA_SQL, ${_ns}::ROCPD_INDEXES_SQL },\n"
    )
endforeach()

configure_file(
    "${SQL_SCHEMA_DIR}/embedded_schemas.in"
    "${SQL_SCHEMA_BINARY_DIR}/schema/embedded_schemas.hpp"
    @ONLY
)

string(REPLACE ";" ", " _ROCPD_SCHEMA_VERSION_LIST "${ROCPD_SCHEMA_VERSIONS}")
message(
    STATUS
    "[profiler-hub] Generated RocPD schema headers for "
    "${_ROCPD_SCHEMA_VERSION_LIST}\n"
    "   Reading ${_ROCPD_SCHEMA_ROOT}/versions.yml\n"
    "   And generated schema headers in ${SQL_SCHEMA_BINARY_DIR}/schema/versions directory"
)
