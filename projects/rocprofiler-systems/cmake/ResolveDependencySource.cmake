# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ======================================================================================
# ResolveDependencySource.cmake
#
# Generic ordered fallback chain for locating a third-party dependency's source,
# shared by any TPL that needs to tolerate GNU-mirror-style download flakiness
# instead of failing the build outright.
#
# Usage:
#
#   rocprofiler_systems_resolve_dependency_source(
#       NAME           <LogicalName>        # e.g. LibIberty
#       CHECK_COMMAND  <macro-name>         # caller-defined; MUST be declared with
#                                           # macro(), not function() - it is invoked
#                                           # via cmake_language(CALL ...) and must set
#                                           # <NAME>_FOUND_SYSTEM (TRUE/FALSE) in this
#                                           # function's scope. A function() would get
#                                           # its own fresh scope and the variable would
#                                           # never propagate back, so CHECK_COMMAND
#                                           # would silently and permanently evaluate as
#                                           # "not found".
#       CACHE_FILENAME <filename>           # e.g. binutils-libiberty-src.tar.gz
#       VERSION        <value>              # substituted for the "VERSION" token
#                                           # in MIRROR_URLS
#       MIRROR_URLS    <url1> [<url2> ...]
#       OVERRIDE_VAR   <cmake-var-name>     # explicit escape hatch, checked first
#   )
#
# On return, sets in the caller's scope:
#
#   <NAME>_FOUND_SYSTEM     - TRUE if CHECK_COMMAND found a usable system install
#   <NAME>_RESOLVED_SOURCE  - URL/path list for ExternalProject_Add's URL argument,
#                             set only when a source build is actually needed
#
# Ordered fallback: explicit override -> system check -> local cache file ->
# network mirrors -> FATAL_ERROR (only if no mirrors were even provided).
# STERILE_BUILD skips the network tier outright and fails immediately instead.
# STERILE_BUILD is read as an ambient CMake variable set elsewhere in the project
# (not passed as an argument here), consistent with existing project convention
# (e.g. the original DyninstLibIberty.cmake reads it the same way).
# ======================================================================================

include_guard(GLOBAL)

set(ROCPROFSYS_TPL_STAGING_DIR
    "/opt/rocprofiler-systems-deps"
    CACHE PATH
    "Directory CI images pre-stage third-party source tarballs into"
)

function(rocprofiler_systems_resolve_dependency_source)
    set(_one_value NAME CHECK_COMMAND CACHE_FILENAME VERSION OVERRIDE_VAR)
    set(_multi_value MIRROR_URLS)
    cmake_parse_arguments(ARG "" "${_one_value}" "${_multi_value}" ${ARGN})

    if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR
            "rocprofiler_systems_resolve_dependency_source: unrecognized or "
            "incomplete arguments: ${ARG_UNPARSED_ARGUMENTS} ${ARG_KEYWORDS_MISSING_VALUES}"
        )
    endif()

    if(NOT ARG_NAME)
        message(FATAL_ERROR "rocprofiler_systems_resolve_dependency_source: NAME is required")
    endif()

    set(${ARG_NAME}_FOUND_SYSTEM FALSE PARENT_SCOPE)
    set(${ARG_NAME}_RESOLVED_SOURCE "" PARENT_SCOPE)

    # Tier 1: explicit override - caller/CI knows best, skip every other tier.
    if(ARG_OVERRIDE_VAR AND DEFINED "${ARG_OVERRIDE_VAR}" AND NOT "${${ARG_OVERRIDE_VAR}}" STREQUAL "")
        set(${ARG_NAME}_RESOLVED_SOURCE "${${ARG_OVERRIDE_VAR}}" PARENT_SCOPE)
        return()
    endif()

    # Tier 2: system check, provided by the caller (find_package, find_library, ...).
    if(ARG_CHECK_COMMAND)
        unset(${ARG_NAME}_FOUND_SYSTEM)
        cmake_language(CALL ${ARG_CHECK_COMMAND})
        if(NOT DEFINED ${ARG_NAME}_FOUND_SYSTEM)
            message(FATAL_ERROR
                "rocprofiler_systems_resolve_dependency_source: CHECK_COMMAND "
                "'${ARG_CHECK_COMMAND}' did not set ${ARG_NAME}_FOUND_SYSTEM. "
                "CHECK_COMMAND must be defined with macro(), not function(), so "
                "that the variable it sets is visible in this scope."
            )
        endif()
        if(${ARG_NAME}_FOUND_SYSTEM)
            set(${ARG_NAME}_FOUND_SYSTEM TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Tier 3: local cache file staged by CI (or a developer) at a conventional path.
    if(ARG_CACHE_FILENAME)
        set(_candidate "${ROCPROFSYS_TPL_STAGING_DIR}/${ARG_CACHE_FILENAME}")
        if(EXISTS "${_candidate}")
            set(${ARG_NAME}_RESOLVED_SOURCE "file://${_candidate}" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Tier 4: network mirrors, skipped entirely for sterile builds.
    if(STERILE_BUILD)
        if(ARG_CACHE_FILENAME)
            message(FATAL_ERROR
                "${ARG_NAME} was not found on the system or in the local TPL cache "
                "(${ROCPROFSYS_TPL_STAGING_DIR}), and this is a sterile build, so "
                "network mirrors cannot be used. Install ${ARG_NAME} on the system or "
                "pre-stage ${ARG_CACHE_FILENAME} in ${ROCPROFSYS_TPL_STAGING_DIR}."
            )
        else()
            message(FATAL_ERROR
                "${ARG_NAME} was not found on the system, and this is a sterile "
                "build, so network mirrors cannot be used. Install ${ARG_NAME} on "
                "the system."
            )
        endif()
    endif()

    if(NOT ARG_MIRROR_URLS)
        message(FATAL_ERROR
            "${ARG_NAME} was not found on the system or in the local TPL cache, "
            "and no MIRROR_URLS were provided to fall back to."
        )
    endif()

    if(NOT ARG_VERSION)
        message(FATAL_ERROR
            "rocprofiler_systems_resolve_dependency_source: MIRROR_URLS given but "
            "VERSION is empty"
        )
    endif()

    set(_resolved_urls "")
    foreach(_url ${ARG_MIRROR_URLS})
        string(REPLACE "VERSION" "${ARG_VERSION}" _url "${_url}")
        list(APPEND _resolved_urls "${_url}")
    endforeach()

    set(${ARG_NAME}_RESOLVED_SOURCE ${_resolved_urls} PARENT_SCOPE)
endfunction()
