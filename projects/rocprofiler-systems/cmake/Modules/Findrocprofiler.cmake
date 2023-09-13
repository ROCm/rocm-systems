# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying file
# Copyright.txt or https://cmake.org/licensing for details.

include(FindPackageHandleStandardArgs)

# ----------------------------------------------------------------------------------------#

if(NOT ROCM_PATH AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
endif()

foreach(_DIR ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm /opt/rocm/rocprofiler)
    if(EXISTS ${_DIR})
        get_filename_component(_ABS_DIR "${_DIR}" REALPATH)
        list(APPEND _ROCM_ROCPROFILER_PATHS ${_ABS_DIR})
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#

find_path(
    rocprofiler_ROOT_DIR
    NAMES include/rocprofiler/rocprofiler.h include/rocprofiler.h
    HINTS ${_ROCM_ROCPROFILER_PATHS}
    PATHS ${_ROCM_ROCPROFILER_PATHS}
    PATH_SUFFIXES rocprofiler)

mark_as_advanced(rocprofiler_ROOT_DIR)

# ----------------------------------------------------------------------------------------#

find_path(
    rocprofiler_INCLUDE_DIR
    NAMES rocprofiler.h
    HINTS ${rocprofiler_ROOT_DIR} ${_ROCM_ROCPROFILER_PATHS}
    PATHS ${rocprofiler_ROOT_DIR} ${_ROCM_ROCPROFILER_PATHS}
    PATH_SUFFIXES include include/rocprofiler rocprofiler/include)

mark_as_advanced(rocprofiler_INCLUDE_DIR)

find_path(
    rocprofiler_hsa_INCLUDE_DIR
    NAMES hsa.h
    HINTS ${rocprofiler_ROOT_DIR} ${_ROCM_ROCPROFILER_PATHS}
    PATHS ${rocprofiler_ROOT_DIR} ${_ROCM_ROCPROFILER_PATHS}
    PATH_SUFFIXES include include/hsa)

mark_as_advanced(rocprofiler_hsa_INCLUDE_DIR)

# ----------------------------------------------------------------------------------------#

find_library(
    rocprofiler_LIBRARY
    NAMES ${CMAKE_SHARED_LIBRARY_PREFIX}rocprofiler64${CMAKE_SHARED_LIBRARY_SUFFIX}.1
          rocprofiler64 rocprofiler
    HINTS ${rocprofiler_ROOT_DIR}/rocprofiler ${rocprofiler_ROOT_DIR}
          ${_ROCM_ROCPROFILER_PATHS}
    PATHS ${rocprofiler_ROOT_DIR}/rocprofiler ${rocprofiler_ROOT_DIR}
          ${_ROCM_ROCPROFILER_PATHS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)

find_library(
    rocprofiler_hsa-runtime_LIBRARY
    NAMES hsa-runtime64 hsa-runtime
    HINTS ${rocprofiler_ROOT_DIR} ${_ROCM_ROCPROFILER_PATHS}
    PATHS ${rocprofiler_ROOT_DIR} ${_ROCM_ROCPROFILER_PATHS}
    PATH_SUFFIXES lib lib64)

if(rocprofiler_LIBRARY)
    get_filename_component(rocprofiler_LIBRARY_DIR "${rocprofiler_LIBRARY}" PATH CACHE)
endif()

mark_as_advanced(rocprofiler_LIBRARY rocprofiler_hsa-runtime_LIBRARY)
unset(_ROCM_ROCPROFILER_PATHS)

if(ROCmVersion_NUMERIC_VERSION EQUAL 50500)
    find_library(
        rocprofiler_pciaccess_LIBRARY
        NAMES pciaccess
        PATH_SUFFIXES lib lib64)
    mark_as_advanced(rocprofiler_pciaccess_LIBRARY)
endif()

# ----------------------------------------------------------------------------------------#

find_package_handle_standard_args(
    rocprofiler DEFAULT_MSG rocprofiler_ROOT_DIR rocprofiler_INCLUDE_DIR
    rocprofiler_hsa_INCLUDE_DIR rocprofiler_LIBRARY rocprofiler_hsa-runtime_LIBRARY)

# ----------------------------------------------------------------------------------------#

if(rocprofiler_FOUND)
    add_library(rocprofiler::rocprofiler INTERFACE IMPORTED)
    add_library(rocprofiler::roctx INTERFACE IMPORTED)
    set(rocprofiler_INCLUDE_DIRS ${rocprofiler_INCLUDE_DIR}
                                 ${rocprofiler_hsa_INCLUDE_DIR})
    set(rocprofiler_LIBRARY_DIRS ${rocprofiler_LIBRARY_DIR})
    set(rocprofiler_LIBRARIES ${rocprofiler_LIBRARY} ${rocprofiler_hsa-runtime_LIBRARY})
    if(rocprofiler_pciaccess_LIBRARY)
        list(APPEND rocprofiler_LIBRARIES ${rocprofiler_pciaccess_LIBRARY})
    endif()

    target_include_directories(
        rocprofiler::rocprofiler INTERFACE ${rocprofiler_INCLUDE_DIR}
                                           ${rocprofiler_hsa_INCLUDE_DIR})

    target_link_libraries(rocprofiler::rocprofiler INTERFACE ${rocprofiler_LIBRARIES})
endif()
