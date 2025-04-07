# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

set(RCCL-Headers_INCLUDE_DIR_INTERNAL
    "${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys/library/tpls/rccl"
    CACHE PATH "Path to internal rccl.h")

# ----------------------------------------------------------------------------------------#

if(NOT ROCM_PATH AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
endif()

foreach(_DIR ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm /opt/rocm/rccl)
    if(EXISTS ${_DIR})
        get_filename_component(_ABS_DIR "${_DIR}" REALPATH)
        list(APPEND _RCCL_PATHS ${_ABS_DIR})
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#

find_package(
    rccl
    QUIET
    CONFIG
    HINTS
    ${_RCCL_PATHS}
    PATHS
    ${_RCCL_PATHS}
    PATH_SUFFIXES
    rccl/lib/cmake)

if(NOT rccl_FOUND)
    set(RCCL-Headers_INCLUDE_DIR
        "${RCCL-Headers_INCLUDE_DIR_INTERNAL}"
        CACHE PATH "Path to RCCL headers")
else()
    set(RCCL-Headers_INCLUDE_DIR
        "${rccl_INCLUDE_DIR}"
        CACHE PATH "Path to RCCL headers")
endif()

# because of the annoying warning starting with v5.2.0, we've got to do this crap
if(ROCmVersion_NUMERIC_VERSION)
    if(ROCmVersion_NUMERIC_VERSION LESS 50200)
        set(_RCCL-Headers_FILE "rccl.h")
        set(_RCCL-Headers_DIR "/rccl")
    else()
        set(_RCCL-Headers_FILE "rccl/rccl.h")
        set(_RCCL-Headers_DIR "")
    endif()
else()
    set(_RCCL-Headers_FILE "rccl/rccl.h")
    set(_RCCL-Headers_DIR "")
endif()

if(NOT EXISTS "${RCCL-Headers_INCLUDE_DIR}/${_RCCL-Headers_FILE}")
    rocprofiler_systems_message(
        AUTHOR_WARNING
        "RCCL header (${RCCL-Headers_INCLUDE_DIR}/${_RCCL-Headers_FILE}) does not exist! Setting RCCL-Headers_INCLUDE_DIR to internal RCCL include directory: ${RCCL-Headers_INCLUDE_DIR_INTERNAL}"
        )
    set(RCCL-Headers_INCLUDE_DIR
        "${RCCL-Headers_INCLUDE_DIR_INTERNAL}${_RCCL-Headers_DIR}"
        CACHE PATH "Path to RCCL headers" FORCE)
endif()

unset(_RCCL-Headers_FILE)
unset(_RCCL-Headers_DIR)

mark_as_advanced(RCCL-Headers_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

find_package_handle_standard_args(RCCL-Headers DEFAULT_MSG RCCL-Headers_INCLUDE_DIR)

# ------------------------------------------------------------------------------#

if(RCCL-Headers_FOUND)
    add_library(roc::rccl-headers INTERFACE IMPORTED)
    set(RCCL-Headers_INCLUDE_DIRS ${RCCL-Headers_INCLUDE_DIR})

    target_include_directories(roc::rccl-headers SYSTEM
                               INTERFACE ${RCCL-Headers_INCLUDE_DIR})

    add_library(RCCL-Headers::RCCL-Headers INTERFACE IMPORTED)
    target_link_libraries(RCCL-Headers::RCCL-Headers INTERFACE roc::rccl-headers)
endif()

# ------------------------------------------------------------------------------#
