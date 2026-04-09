# ------------------------------------------------------------------------------#
#
# Finds headers for UCX (Unified Communication X)
#
# UCX is a high-performance communication framework used as a transport layer
# for MPI and other communication libraries. This module locates UCX headers
# (ucp.h, uct.h) for tracing and interception purposes.
#
# ------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

set(UCX_HEADERS_INCLUDE_DIR_INTERNAL
    "${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys/library/tpls/ucx"
    CACHE PATH
    "Path to internal UCX headers"
)

# ----------------------------------------------------------------------------------------#
# Find UCX headers (ucp.h and uct.h are under ucx/ subdirectory)
find_path(
    UCX_HEADERS_INCLUDE_DIR
    NAMES ucp/api/ucp.h
    PATHS /usr/include /usr/local/include /opt/ucx/include
)

if(NOT EXISTS "${UCX_HEADERS_INCLUDE_DIR}")
    rocprofiler_systems_message(
        AUTHOR_WARNING
        "UCX headers do not exist! Setting UCX_HEADERS_INCLUDE_DIR to internal directory: ${UCX_HEADERS_INCLUDE_DIR_INTERNAL}"
    )
    set(UCX_HEADERS_INCLUDE_DIR
        "${UCX_HEADERS_INCLUDE_DIR_INTERNAL}"
        CACHE PATH
        "Path to UCX headers"
        FORCE
    )
else()
    rocprofiler_systems_message(STATUS "UCX headers found: ${UCX_HEADERS_INCLUDE_DIR}")
endif()

mark_as_advanced(UCX_HEADERS_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

find_package_handle_standard_args(UCX DEFAULT_MSG UCX_HEADERS_INCLUDE_DIR)

# ------------------------------------------------------------------------------#

if(UCX_FOUND)
    add_library(roc::ucx-headers INTERFACE IMPORTED)
    target_include_directories(
        roc::ucx-headers
        SYSTEM
        INTERFACE ${UCX_HEADERS_INCLUDE_DIR}
    )
endif()

# ------------------------------------------------------------------------------#
