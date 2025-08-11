# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

set(LIBVA_HEADERS_INCLUDE_DIR_INTERNAL
    "${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys/library/tpls"
    CACHE PATH
    "Path to internal va headers"
)

# ----------------------------------------------------------------------------------------#
find_path(
    LIBVA_HEADERS_INCLUDE_DIR
    NAMES va/va.h
    PATHS /opt/amdgpu/include
    NO_DEFAULT_PATH
)

if(NOT EXISTS "${LIBVA_HEADERS_INCLUDE_DIR}")
    rocprofiler_systems_message(
        AUTHOR_WARNING
        "VA API header does not exist! Setting LIBVA_HEADERS_INCLUDE_DIR to internal directory: ${LIBVA_HEADERS_INCLUDE_DIR}"
    )
    set(LIBVA_HEADERS_INCLUDE_DIR
        "${LIBVA_HEADERS_INCLUDE_DIR_INTERNAL}"
        CACHE PATH
        "Path to VA API headers"
        FORCE
    )
else()
    rocprofiler_systems_message(STATUS
                                "VA API header found: ${LIBVA_HEADERS_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(LIBVA_HEADERS_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

find_package_handle_standard_args(Libva-headers DEFAULT_MSG LIBVA_HEADERS_INCLUDE_DIR)

# ------------------------------------------------------------------------------#

if(Libva-headers_FOUND)
    add_library(roc::libva-headers INTERFACE IMPORTED)
    target_include_directories(
        roc::libva-headers
        SYSTEM
        INTERFACE ${LIBVA_HEADERS_INCLUDE_DIR}
    )
endif()

# ------------------------------------------------------------------------------#
