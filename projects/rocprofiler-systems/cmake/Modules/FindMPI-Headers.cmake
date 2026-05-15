# ------------------------------------------------------------------------------#
#
# Finds headers for MPI
#
# ------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)

set(MPI_HEADERS_VENDOR_INTERNAL
    "OpenMPI"
    CACHE STRING
    "Distribution type of internal mpi.h"
)
set(MPI_HEADERS_INCLUDE_DIR_INTERNAL
    "${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys/library/tpls/mpi"
    CACHE PATH
    "Path to internal ${MPI_HEADERS_VENDOR_INTERNAL} mpi.h"
)
mark_as_advanced(MPI_HEADERS_VENDOR_INTERNAL)
mark_as_advanced(MPI_HEADERS_INCLUDE_DIR_INTERNAL)

if(
    DEFINED _MPI_HEADERS_LAST_MPI_HEADERS_INCLUDE_DIR
    AND NOT _MPI_HEADERS_LAST_MPI_HEADERS_INCLUDE_DIR STREQUAL MPI_HEADERS_INCLUDE_DIR
)
    unset(MPI_HEADERS_VENDOR CACHE)
    # if skip mpicxx is on because of internal unset this value
    if(
        MPI_HEADERS_SKIP_MPICXX
        AND "${_MPI_HEADERS_LAST_MPI_HEADERS_INCLUDE_DIR}"
            STREQUAL
            "${MPI_HEADERS_INCLUDE_DIR_INTERNAL}"
    )
        unset(MPI_HEADERS_SKIP_MPICXX CACHE)
    endif()
endif()

# define the (OMPI|MPICH)_SKIP_MPICXX pp definition
option(MPI_HEADERS_SKIP_MPICXX "Skip MPI C++" ON)
mark_as_advanced(MPI_HEADERS_SKIP_MPICXX)

# ------------------------------------------------------------------------------#
#
# Try to find an openmpi header
#
# ------------------------------------------------------------------------------#

find_path(
    MPI_HEADERS_INCLUDE_DIR
    NAMES mpi.h
    PATH_SUFFIXES include/openmpi openmpi include
    HINTS ${MPI_ROOT_DIR}
    PATHS ${MPI_ROOT_DIR}
)

# ------------------------------------------------------------------------------#
#
# If direct find failed, try to find MPI and use MPI_C_INCLUDE_DIRS
#
# ------------------------------------------------------------------------------#

if(NOT MPI_HEADERS_INCLUDE_DIR)
    find_package(MPI QUIET)
    if(MPI_C_INCLUDE_DIRS)
        set(MPI_HEADERS_INCLUDE_DIR
            ${MPI_C_INCLUDE_DIRS}
            CACHE PATH
            "Include directory for MPI"
            FORCE
        )
    elseif(MPI_CXX_INCLUDE_DIRS)
        set(MPI_HEADERS_INCLUDE_DIR
            ${MPI_CXX_INCLUDE_DIRS}
            CACHE PATH
            "Include directory for MPI"
            FORCE
        )
    endif()
endif()

# ------------------------------------------------------------------------------#
#
# If found, try to determine the MPI vendor (i.e. distribution)
#
# ------------------------------------------------------------------------------#

if(MPI_HEADERS_INCLUDE_DIR)
    file(STRINGS ${MPI_HEADERS_INCLUDE_DIR}/mpi.h _MPI_H_LINES REGEX "#([ \t]+)define ")
    foreach(_LINE ${_MPI_H_LINES})
        if("${_LINE}" MATCHES "define([ \t]+)OMPI_")
            set(MPI_HEADERS_VENDOR
                "OpenMPI"
                CACHE STRING
                "MPI headers are from OpenMPI distribution"
            )
            break()
        elseif("${_LINE}" MATCHES "define([ \t]+)MPICH_")
            set(MPI_HEADERS_VENDOR
                "MPICH"
                CACHE STRING
                "MPI headers are from MPICH distribution"
            )
            break()
        elseif("${_LINE}" MATCHES "define([ \t]+)MVAPICH_")
            set(MPI_HEADERS_VENDOR
                "MVAPICH"
                CACHE STRING
                "MPI headers are from MVAPICH distribution"
            )
            break()
        endif()
    endforeach()
endif()

# ------------------------------------------------------------------------------#
#
# If not found, use internal version or if vendor is MPICH set to internal
#
# ------------------------------------------------------------------------------#

if(NOT MPI_HEADERS_INCLUDE_DIR)
    set(MPI_HEADERS_INCLUDE_DIR "${MPI_HEADERS_INCLUDE_DIR_INTERNAL}" CACHE PATH "" FORCE)
    set(MPI_HEADERS_VENDOR
        "${MPI_HEADERS_VENDOR_INTERNAL}"
        CACHE STRING
        "MPI headers are from OpenMPI distribution"
        FORCE
    )
    set(MPI_HEADERS_SKIP_MPICXX ON CACHE BOOL "" FORCE)
elseif("${MPI_HEADERS_VENDOR}" STREQUAL "MPICH")
    option(
        MPI_HEADERS_ALLOW_MPICH
        "Permit the use of MPI headers from MPICH instead of using internal OpenMPI header"
        OFF
    )
    mark_as_advanced(MPI_HEADERS_ALLOW_MPICH)
    if(NOT MPI_HEADERS_ALLOW_MPICH)
        set(_MESSAGE "\nFound MPI headers belonging to a MPICH distribution. ")
        set(_MESSAGE
            "${_MESSAGE}The data types for MPICH will cause segfaults when an application uses OpenMPI, "
        )
        set(_MESSAGE
            "${_MESSAGE}whereas the OpenMPI data types are compatible with both. "
        )
        set(_MESSAGE
            "${_MESSAGE}Forcing internal OpenMPI header... This can be disabled via MPI_HEADERS_ALLOW_MPICH=ON ...\n"
        )
        message(AUTHOR_WARNING "${_MESSAGE}")
        unset(_MESSAGE)
        set(MPI_HEADERS_INCLUDE_DIR
            "${MPI_HEADERS_INCLUDE_DIR_INTERNAL}"
            CACHE PATH
            ""
            FORCE
        )
        set(MPI_HEADERS_VENDOR
            "${MPI_HEADERS_VENDOR_INTERNAL}"
            CACHE STRING
            "MPI headers are from OpenMPI distribution"
            FORCE
        )
        set(MPI_HEADERS_SKIP_MPICXX ON CACHE BOOL "" FORCE)
    endif()
endif()

# set local variable
if(MPI_HEADERS_INCLUDE_DIR)
    set(MPI_HEADERS_INCLUDE_DIRS ${MPI_HEADERS_INCLUDE_DIR})
endif()

mark_as_advanced(MPI_HEADERS_INCLUDE_DIR)

# store value to detect changes
set(_MPI_HEADERS_LAST_MPI_HEADERS_INCLUDE_DIR
    "${MPI_HEADERS_INCLUDE_DIR}"
    CACHE INTERNAL
    "Last value of MPI_HEADERS_INCLUDE_DIR"
)

# handle find_package
find_package_handle_standard_args(MPI-Headers REQUIRED_VARS MPI_HEADERS_INCLUDE_DIR)

if(MPI-Headers_FOUND)
    add_library(MPI::MPI_HEADERS IMPORTED INTERFACE)
    if(MPI_HEADERS_SKIP_MPICXX)
        if(MPI_HEADERS_VENDOR STREQUAL "MPICH")
            target_compile_definitions(MPI::MPI_HEADERS INTERFACE MPICH_SKIP_MPICXX=1)
        else()
            target_compile_definitions(MPI::MPI_HEADERS INTERFACE OMPI_SKIP_MPICXX=1)
        endif()
    endif()
    target_include_directories(
        MPI::MPI_HEADERS
        INTERFACE
            $<$<COMPILE_LANGUAGE:C>:${MPI_HEADERS_INCLUDE_DIR}>
            $<$<COMPILE_LANGUAGE:CXX>:${MPI_HEADERS_INCLUDE_DIR}>
    )
endif()
