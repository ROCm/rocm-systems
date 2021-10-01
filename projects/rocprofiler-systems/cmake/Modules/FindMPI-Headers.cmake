# ------------------------------------------------------------------------------#
#
# Finds headers for MPI
#
# ------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)

find_path(
    MPI_HEADERS_INCLUDE_DIR
    NAMES mpi.h
    PATH_SUFFIXES include include/mpich include/openmpi mpich openmpi
    HINTS ${MPI_ROOT_DIR}
    PATHS ${MPI_ROOT_DIR})

if(NOT MPI_HEADERS_INCLUDE_DIR)
    find_package(MPI QUIET)
    if(MPI_C_INCLUDE_DIRS)
        set(MPI_HEADERS_INCLUDE_DIR
            ${MPI_C_INCLUDE_DIRS}
            CACHE PATH "Include directory for MPI" FORCE)
    elseif(MPI_CXX_INCLUDE_DIRS)
        set(MPI_HEADERS_INCLUDE_DIR
            ${MPI_CXX_INCLUDE_DIRS}
            CACHE PATH "Include directory for MPI" FORCE)
    endif()
endif()

if(MPI_HEADERS_INCLUDE_DIR)
    set(MPI_HEADERS_INCLUDE_DIRS ${MPI_HEADERS_INCLUDE_DIR})
endif()

mark_as_advanced(MPI_HEADERS_INCLUDE_DIR)
find_package_handle_standard_args(MPI-Headers REQUIRED_VARS MPI_HEADERS_INCLUDE_DIR)

if(MPI-Headers_FOUND)
    add_library(MPI::MPI_HEADERS IMPORTED INTERFACE)
    target_include_directories(
        MPI::MPI_HEADERS INTERFACE $<$<COMPILE_LANGUAGE:C>:${MPI_HEADERS_INCLUDE_DIR}>
                                   $<$<COMPILE_LANGUAGE:CXX>:${MPI_HEADERS_INCLUDE_DIR}>)
endif()
