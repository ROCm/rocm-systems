#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-samples-spm LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

if(ROCPROFILER_MEMCHECK STREQUAL "ThreadSanitizer")
    set(IS_THREAD_SANITIZER ON)
else()
    set(IS_THREAD_SANITIZER OFF)
endif()

set(IS_DISABLED False)

rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
if(IS_SPM_DISABLED OR IS_THREAD_SANITIZER)
    set(IS_DISABLED True)
endif()

set(IS_DISABLED True)

find_package(rocprofiler-sdk REQUIRED)

add_library(spm-device-counting-client SHARED)
target_sources(spm-device-counting-client PRIVATE spm_device_counting_client.cpp
                                                  client.hpp)
target_link_libraries(
    spm-device-counting-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(spm-device-counting)
target_sources(spm-device-counting PRIVATE main.cpp)
target_link_libraries(spm-device-counting PRIVATE spm-device-counting-client
                                                  Threads::Threads)

rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)
rocprofiler_samples_get_preload_env(PRELOAD_ENV spm-device-counting-client)

set(spm-device-counting-env "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}"
                            "ROCPROFILER_SPM_BETA_ENABLED=True")

add_test(NAME spm-device-counting COMMAND $<TARGET_FILE:spm-device-counting>)

set_tests_properties(
    spm-device-counting
    PROPERTIES TIMEOUT
               120
               LABELS
               "samples"
               ENVIRONMENT
               "${spm-device-counting-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${IS_DISABLED}")
