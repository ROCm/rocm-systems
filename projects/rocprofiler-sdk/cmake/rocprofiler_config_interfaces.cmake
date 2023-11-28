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
# filesystem library
#
# ----------------------------------------------------------------------------------------#

if(NOT ROCPROFILER_BUILD_GHC_FS)
    if(CMAKE_CXX_COMPILER_IS_GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
        target_link_libraries(rocprofiler-cxx-filesystem INTERFACE stdc++fs)
    elseif(CMAKE_CXX_COMPILER_IS_CLANG AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(rocprofiler-cxx-filesystem INTERFACE c++fs)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# HIP
#
# ----------------------------------------------------------------------------------------#
find_package(rocm_version)

if(rocm_version_FOUND)
    list(APPEND CMAKE_PREFIX_PATH "${rocm_version_DIR}" "${rocm_version_DIR}/llvm")
    list(APPEND CMAKE_MODULE_PATH "${rocm_version_DIR}/hip/cmake"
         "${rocm_version_DIR}/lib/cmake")
endif()

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

string(REPLACE "." ";" HSA_RUNTIME_VERSION "${hsa-runtime64_VERSION}")

# the following values are encoded into version.h
list(GET HSA_RUNTIME_VERSION 0 HSA_RUNTIME_VERSION_MAJOR)
list(GET HSA_RUNTIME_VERSION 1 HSA_RUNTIME_VERSION_MINOR)

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

# ----------------------------------------------------------------------------------------#
#
# PTL (Parallel Tasking Library)
#
# ----------------------------------------------------------------------------------------#
target_link_libraries(rocprofiler-ptl INTERFACE PTL::ptl-static)

# ----------------------------------------------------------------------------------------#
#
# amd aql
#
# ----------------------------------------------------------------------------------------#
find_library(
    hsa-amd-aqlprofile64_library
    NAMES hsa-amd-aqlprofile64 hsa-amd-aqlprofile
    HINTS ${rocm_version_DIR} ${ROCM_PATH}
    PATHS ${rocm_version_DIR} ${ROCM_PATH})

target_link_libraries(rocprofiler-hsa-aql INTERFACE ${hsa-amd-aqlprofile64_library})

# ----------------------------------------------------------------------------------------#
#
# drm
#
# ----------------------------------------------------------------------------------------#
find_path(
    drm_INCLUDE_DIR
    NAMES drm.h
    HINTS ${rocm_version_DIR} ${ROCM_PATH} /opt/amdgpu
    PATHS ${rocm_version_DIR} ${ROCM_PATH} /opt/amdgpu
    PATH_SUFFIXES include/drm include/libdrm include REQUIRED)

find_path(
    xf86drm_INCLUDE_DIR
    NAMES xf86drm.h
    HINTS ${rocm_version_DIR} ${ROCM_PATH} /opt/amdgpu
    PATHS ${rocm_version_DIR} ${ROCM_PATH} /opt/amdgpu
    PATH_SUFFIXES include/drm include/libdrm include REQUIRED)

find_library(
    drm_LIBRARY
    NAMES drm
    HINTS ${rocm_version_DIR} ${ROCM_PATH} /opt/amdgpu
    PATHS ${rocm_version_DIR} ${ROCM_PATH} /opt/amdgpu REQUIRED)

find_library(
    drm_amdgpu_LIBRARY
    NAMES drm_amdgpu
    HINTS ${rocm_version_DIR} ${ROCM_PATH} /opt/amdgpu
    PATHS ${rocm_version_DIR} ${ROCM_PATH} /opt/amdgpu REQUIRED)

target_include_directories(rocprofiler-drm SYSTEM INTERFACE ${drm_INCLUDE_DIR}
                                                            ${xf86drm_INCLUDE_DIR})
target_link_libraries(rocprofiler-drm INTERFACE ${drm_LIBRARY} ${drm_amdgpu_LIBRARY})
