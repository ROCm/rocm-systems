# include guard
include_guard(DIRECTORY)

# ########################################################################################
#
# External Packages are found here
#
# ########################################################################################

target_include_directories(
    rocprofiler-headers
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(
    rocprofiler-headers INTERFACE $<BUILD_INTERFACE:AMD_INTERNAL_BUILD=1>
                                  $<BUILD_INTERFACE:__HIP_PLATFORM_AMD__=1>)

# ensure the env overrides the appending /opt/rocm later
string(REPLACE ":" ";" CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")

set(ROCPROFILER_DEFAULT_ROCM_PATH
    /opt/rocm
    CACHE PATH "Default search path for ROCM")
if(EXISTS ${ROCPROFILER_DEFAULT_ROCM_PATH})
    get_filename_component(_ROCPROFILER_DEFAULT_ROCM_PATH
                           "${ROCPROFILER_DEFAULT_ROCM_PATH}" REALPATH)

    if(NOT "${_ROCPROFILER_DEFAULT_ROCM_PATH}" STREQUAL
       "${ROCPROFILER_DEFAULT_ROCM_PATH}")
        set(ROCPROFILER_DEFAULT_ROCM_PATH
            "${_ROCPROFILER_DEFAULT_ROCM_PATH}"
            CACHE PATH "Default search path for ROCM" FORCE)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# Threading
#
# ----------------------------------------------------------------------------------------#

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG OFF)

find_library(pthread_LIBRARY NAMES pthread pthreads)
find_package_handle_standard_args(pthread-library REQUIRED_VARS pthread_LIBRARY)

find_library(pthread_LIBRARY NAMES pthread pthreads)
find_package_handle_standard_args(pthread-library REQUIRED_VARS pthread_LIBRARY)

if(pthread_LIBRARY)
    target_link_libraries(rocprofiler-threading INTERFACE ${pthread_LIBRARY})
else()
    find_package(Threads ${rocprofiler_FIND_QUIETLY} ${rocprofiler_FIND_REQUIREMENT})
    if(Threads_FOUND)
        target_link_libraries(rocprofiler-threading INTERFACE Threads::Threads)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# dynamic linking (dl) and runtime (rt) libraries
#
# ----------------------------------------------------------------------------------------#

foreach(_LIB dl rt)
    find_library(${_LIB}_LIBRARY NAMES ${_LIB})
    find_package_handle_standard_args(${_LIB}-library REQUIRED_VARS ${_LIB}_LIBRARY)
    if(${_LIB}_LIBRARY)
        target_link_libraries(rocprofiler-threading INTERFACE ${${_LIB}_LIBRARY})
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#
#
# stdc++fs (filesystem) library
#
# ----------------------------------------------------------------------------------------#

find_library(stdcxxfs_LIBRARY NAMES stdc++fs)
find_package_handle_standard_args(stdcxxfs-library REQUIRED_VARS stdcxxfs_LIBRARY)

if(stdcxxfs_LIBRARY)
    target_link_libraries(rocprofiler-stdcxxfs INTERFACE ${stdcxxfs_LIBRARY})
else()
    target_link_libraries(rocprofiler-stdcxxfs INTERFACE stdc++fs)
endif()

# ----------------------------------------------------------------------------------------#
#
# HIP
#
# ----------------------------------------------------------------------------------------#

find_package(rocm_version REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "${rocm_version_DIR}" "${rocm_version_DIR}/llvm")
list(APPEND CMAKE_MODULE_PATH "${rocm_version_DIR}/hip/cmake"
     "${rocm_version_DIR}/lib/cmake")
find_package(hip REQUIRED CONFIG)
target_link_libraries(rocprofiler-hip INTERFACE hip::host)

# ----------------------------------------------------------------------------------------#
#
# HSA runtime
#
# ----------------------------------------------------------------------------------------#

find_package(
    hsa-runtime64
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH})

target_link_libraries(rocprofiler-hsa-runtime INTERFACE hsa-runtime64::hsa-runtime64)

# ----------------------------------------------------------------------------------------#
#
# amd comgr
#
# ----------------------------------------------------------------------------------------#

find_package(
    amd_comgr
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATH_SUFFIXES
    lib/cmake/amd_comgr)

target_link_libraries(rocprofiler-amd-comgr INTERFACE amd_comgr)
