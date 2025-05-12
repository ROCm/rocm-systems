# include guard
include_guard(DIRECTORY)

include(rocprofiler_config_nolink_target)

# ########################################################################################
#
# External Packages are found here
#
# ########################################################################################

target_include_directories(
    rocprofiler-sdk-headers
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_definitions(
    rocprofiler-sdk-headers INTERFACE $<BUILD_INTERFACE:AMD_INTERNAL_BUILD=1>
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
    target_link_libraries(rocprofiler-sdk-threading INTERFACE ${pthread_LIBRARY})
else()
    find_package(Threads ${rocprofiler_FIND_QUIETLY} ${rocprofiler_FIND_REQUIREMENT})

    if(Threads_FOUND)
        target_link_libraries(rocprofiler-sdk-threading INTERFACE Threads::Threads)
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
        target_link_libraries(rocprofiler-sdk-threading INTERFACE ${${_LIB}_LIBRARY})
    endif()
endforeach()

# ----------------------------------------------------------------------------------------#
#
# atomic library
#
# ----------------------------------------------------------------------------------------#

target_link_libraries(rocprofiler-sdk-atomic INTERFACE atomic)

# ----------------------------------------------------------------------------------------#
#
# filesystem library
#
# ----------------------------------------------------------------------------------------#

if(NOT ROCPROFILER_BUILD_GHC_FS)
    if(CMAKE_CXX_COMPILER_IS_GNU AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
        target_link_libraries(rocprofiler-sdk-cxx-filesystem INTERFACE stdc++fs)
    elseif(CMAKE_CXX_COMPILER_IS_CLANG AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(rocprofiler-sdk-cxx-filesystem INTERFACE c++fs)
    endif()
endif()

# ----------------------------------------------------------------------------------------#
#
# HIP
#
# ----------------------------------------------------------------------------------------#

find_package(rocm_version 6.2)

if(rocm_version_FOUND)
    list(APPEND CMAKE_PREFIX_PATH "${rocm_version_DIR}" "${rocm_version_DIR}/llvm")
    list(APPEND CMAKE_MODULE_PATH "${rocm_version_DIR}/hip/cmake"
         "${rocm_version_DIR}/lib/cmake")
endif()

find_package(
    hip
    6.2
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH})
target_link_libraries(rocprofiler-sdk-hip INTERFACE hip::host)
# TODO: As of 2024/2/12, the hip::host target does not advertise its
# include directory but amdhip64 does. This ordinarily wouldn't be an issue
# because most folks just get it transitively, but here this is doing direct
# property copying to get usage requirements.
# The proper fix is for hip to export a hip::headers target with only usage
# requirements and depend on that.
rocprofiler_config_nolink_target(rocprofiler-sdk-hip-nolink hip::host)
rocprofiler_config_nolink_target(rocprofiler-sdk-hip-nolink hip::amdhip64)

# ----------------------------------------------------------------------------------------#
#
# HSA runtime
#
# ----------------------------------------------------------------------------------------#

find_package(
    hsa-runtime64
    1.14
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

target_link_libraries(rocprofiler-sdk-hsa-runtime INTERFACE hsa-runtime64::hsa-runtime64)
rocprofiler_config_nolink_target(rocprofiler-sdk-hsa-runtime-nolink
                                 hsa-runtime64::hsa-runtime64)

rocprofiler_parse_hsa_api_table_versions(rocprofiler-sdk-hsa-runtime-nolink)

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

target_link_libraries(rocprofiler-sdk-amd-comgr INTERFACE amd_comgr)

# ----------------------------------------------------------------------------------------#
#
# PTL (Parallel Tasking Library)
#
# ----------------------------------------------------------------------------------------#

target_link_libraries(rocprofiler-sdk-ptl INTERFACE PTL::ptl-static)

# ----------------------------------------------------------------------------------------#
#
# libelf
#
# ----------------------------------------------------------------------------------------#

find_package(libelf REQUIRED)
target_link_libraries(rocprofiler-sdk-elf INTERFACE libelf::libelf)

# ----------------------------------------------------------------------------------------#
#
# libdw
#
# ----------------------------------------------------------------------------------------#

find_package(libdw REQUIRED)
target_link_libraries(rocprofiler-sdk-dw INTERFACE libdw::libdw)

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

if(hsa-amd-aqlprofile64_library)
    target_link_libraries(rocprofiler-sdk-hsa-aql INTERFACE ${hsa-amd-aqlprofile64_library})
endif()

# ----------------------------------------------------------------------------------------#
#
# HSAKMT
#
# ----------------------------------------------------------------------------------------#

find_package(
    hsakmt
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATH_SUFFIXES
    lib/cmake/hsakmt)

target_link_libraries(rocprofiler-sdk-hsakmt INTERFACE hsakmt::hsakmt)
rocprofiler_config_nolink_target(rocprofiler-sdk-hsakmt-nolink hsakmt::hsakmt)

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

target_include_directories(rocprofiler-sdk-drm SYSTEM INTERFACE ${drm_INCLUDE_DIR}
                                                                ${xf86drm_INCLUDE_DIR})
target_link_libraries(rocprofiler-sdk-drm INTERFACE ${drm_LIBRARY} ${drm_amdgpu_LIBRARY})

# ----------------------------------------------------------------------------------------#
#
# ELFIO library
#
# ----------------------------------------------------------------------------------------#

# get_target_property(ELFIO_INCLUDE_DIR elfio::elfio INTERFACE_INCLUDE_DIRECTORIES)
# target_include_directories(rocprofiler-sdk-elfio SYSTEM INTERFACE ${ELFIO_INCLUDE_DIR})
target_link_libraries(rocprofiler-sdk-elfio INTERFACE elfio::elfio)

# ----------------------------------------------------------------------------------------#
#
# OTF2
#
# ----------------------------------------------------------------------------------------#

target_link_libraries(rocprofiler-sdk-otf2 INTERFACE otf2::otf2)

# ----------------------------------------------------------------------------------------#
#
# RCCL
#
# ----------------------------------------------------------------------------------------#

find_package(
    rccl
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATH_SUFFIXES
    lib/cmake/rccl)

if(rccl_FOUND
   AND rccl_INCLUDE_DIR
   AND EXISTS "${rccl_INCLUDE_DIR}/rccl/amd_detail/api_trace.h")
    set(rccl_API_TRACE_FOUND ON)
    rocprofiler_config_nolink_target(rocprofiler-sdk-rccl-nolink rccl::rccl)
else()
    set(rccl_API_TRACE_FOUND OFF)
    target_compile_definitions(rocprofiler-sdk-rccl-nolink
                               INTERFACE ROCPROFILER_SDK_USE_SYSTEM_RCCL=0)

endif()

# ----------------------------------------------------------------------------------------#
#
# ROCDecode
#
# ----------------------------------------------------------------------------------------#

find_package(rocDecode)

if(rocDecode_FOUND
   AND rocDecode_INCLUDE_DIR
   AND EXISTS "${rocDecode_INCLUDE_DIR}/rocdecode/amd_detail/rocdecode_api_trace.h")
    rocprofiler_config_nolink_target(
        rocprofiler-sdk-rocdecode-nolink rocdecode::rocdecode INTERFACE
        ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE=1)
else()
    target_compile_definitions(rocprofiler-sdk-rocdecode-nolink
                               INTERFACE ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE=0)

endif()

# ----------------------------------------------------------------------------------------#
#
# rocJPEG
#
# ----------------------------------------------------------------------------------------#

find_package(rocJPEG)

if(rocJPEG_FOUND
   AND rocJPEG_INCLUDE_DIR
   AND EXISTS "${rocJPEG_INCLUDE_DIR}/rocjpeg/amd_detail/rocjpeg_api_trace.h")
    rocprofiler_config_nolink_target(rocprofiler-sdk-rocjpeg-nolink rocjpeg::rocjpeg
                                     INTERFACE ROCPROFILER_SDK_USE_SYSTEM_ROCJPEG=1)
else()
    target_compile_definitions(rocprofiler-sdk-rocjpeg-nolink
                               INTERFACE ROCPROFILER_SDK_USE_SYSTEM_ROCJPEG=0)

endif()
