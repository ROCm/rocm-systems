# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

set(_ROCM_PATHS $ENV{ROCM_HOME} /opt/rocm /opt/rocm/roctracer)

# ----------------------------------------------------------------------------------------#

find_path(
    roctracer_ROOT_DIR
    NAMES include/roctracer.h
    HINTS ${_ROCM_PATHS}
    PATHS ${_ROCM_PATHS}
    PATH_SUFFIXES roctracer)

mark_as_advanced(roctracer_ROOT_DIR)

# ----------------------------------------------------------------------------------------#

find_path(
    roctracer_INCLUDE_DIR
    NAMES roctracer.h
    HINTS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATHS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATH_SUFFIXES roctracer/include include)

mark_as_advanced(roctracer_INCLUDE_DIR)

find_path(
    roctracer_hsa_INCLUDE_DIR
    NAMES hsa.h
    HINTS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATHS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATH_SUFFIXES include include/hsa)

mark_as_advanced(roctracer_hsa_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

find_library(
    roctracer_LIBRARY
    NAMES roctracer64 roctracer
    HINTS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATHS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATH_SUFFIXES lib lib64)

find_library(
    roctracer_roctx_LIBRARY
    NAMES roctx64 roctx
    HINTS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATHS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATH_SUFFIXES lib lib64)

find_library(
    roctracer_kfdwrapper_LIBRARY
    NAMES kfdwrapper64 kfdwrapper
    HINTS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATHS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATH_SUFFIXES lib lib64)

find_library(
    roctracer_hsakmt_LIBRARY
    NAMES hsakmt
    HINTS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATHS ${roctracer_ROOT_DIR} ${_ROCM_PATHS}
    PATH_SUFFIXES lib lib64)

if(roctracer_LIBRARY)
    get_filename_component(roctracer_LIBRARY_DIR "${roctracer_LIBRARY}" PATH CACHE)
endif()

mark_as_advanced(roctracer_LIBRARY roctracer_roctx_LIBRARY)

# ----------------------------------------------------------------------------------------#

find_package_handle_standard_args(
    roctracer DEFAULT_MSG roctracer_ROOT_DIR roctracer_INCLUDE_DIR
    roctracer_hsa_INCLUDE_DIR roctracer_LIBRARY roctracer_roctx_LIBRARY)

# ------------------------------------------------------------------------------#

if(roctracer_FOUND)
    add_library(roctracer::roctracer INTERFACE IMPORTED)
    add_library(roctracer::roctx INTERFACE IMPORTED)
    set(roctracer_INCLUDE_DIRS ${roctracer_INCLUDE_DIR} ${roctracer_hsa_INCLUDE_DIR})
    set(roctracer_LIBRARIES ${roctracer_LIBRARY} ${roctracer_roctx_LIBRARY})
    set(roctracer_LIBRARY_DIRS ${roctracer_LIBRARY_DIR})

    target_include_directories(
        roctracer::roctracer INTERFACE ${roctracer_INCLUDE_DIR}
                                       ${roctracer_hsa_INCLUDE_DIR})
    target_include_directories(roctracer::roctx INTERFACE ${roctracer_INCLUDE_DIR}
                                                          ${roctracer_hsa_INCLUDE_DIR})

    target_link_libraries(roctracer::roctracer INTERFACE ${roctracer_LIBRARY})
    target_link_libraries(roctracer::roctx INTERFACE ${roctracer_roctx_LIBRARY})

    if(roctracer_kfdwrapper_LIBRARY)
        list(APPEND roctracer_LIBRARIES ${roctracer_kfdwrapper_LIBRARY})
        target_link_libraries(roctracer::roctracer
                              INTERFACE ${roctracer_kfdwrapper_LIBRARY})
        target_link_libraries(roctracer::roctx INTERFACE ${roctracer_kfdwrapper_LIBRARY})
    endif()

    if(roctracer_hsakmt_LIBRARY)
        list(APPEND roctracer_LIBRARIES ${roctracer_hsakmt_LIBRARY})
        target_link_libraries(roctracer::roctracer INTERFACE ${roctracer_hsakmt_LIBRARY})
        target_link_libraries(roctracer::roctx INTERFACE ${roctracer_hsakmt_LIBRARY})
    endif()

endif()

# ------------------------------------------------------------------------------#

unset(_ROCM_PATHS)

# ------------------------------------------------------------------------------#
