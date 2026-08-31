# cmake/FindNUMA.cmake
find_path(NUMA_INCLUDE_DIR NAMES numa.h)
find_library(NUMA_LIBRARY NAMES numa)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARY NUMA_INCLUDE_DIR)

if(NUMA_FOUND)
    set(NUMA_LIBRARIES ${NUMA_LIBRARY})
    set(NUMA_INCLUDE_DIRS ${NUMA_INCLUDE_DIR})
    if(NOT TARGET NUMA::NUMA)
        add_library(NUMA::NUMA UNKNOWN IMPORTED)
        set_target_properties(
            NUMA::NUMA PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
                                  IMPORTED_LOCATION "${NUMA_LIBRARY}")
    endif()
endif()
