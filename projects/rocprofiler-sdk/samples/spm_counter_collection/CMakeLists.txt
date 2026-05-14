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

project(rocprofiler-sdk-samples-spm-counter-collection LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

if(NOT ROCPROFILER_MEMCHECK STREQUAL "")
    set(SANITIZER True)
else()
    set(SANITIZER False)
endif()

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_sdk_spm_disabled(IS_SPM_DISABLED)
if(${SANITIZER})
    set(IS_SPM_DISABLED True)
endif()
set(IS_DISABLED True)

add_library(spm-counter-collection-buffer-client SHARED)
target_sources(spm-counter-collection-buffer-client PRIVATE buffered_client.cpp
                                                            common.cpp common.hpp)
target_link_libraries(
    spm-counter-collection-buffer-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(spm-counter-collection-buffer)
target_sources(spm-counter-collection-buffer PRIVATE main.cpp)
target_link_libraries(spm-counter-collection-buffer
                      PRIVATE spm-counter-collection-buffer-client Threads::Threads)

rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)
rocprofiler_samples_get_preload_env(PRELOAD_ENV spm-counter-collection-buffer-client)

set(SPM_ENV ROCPROFILER_SPM_BETA_ENABLED=True)
set(spm-counter-collection-buffer-env "${SPM_ENV}" "${PRELOAD_ENV}" "${LIBRARY_PATH_ENV}")

add_test(NAME spm-counter-collection-buffer
         COMMAND $<TARGET_FILE:spm-counter-collection-buffer>)

set_tests_properties(
    spm-counter-collection-buffer
    PROPERTIES TIMEOUT
               120
               LABELS
               "samples"
               ENVIRONMENT
               "${spm-counter-collection-buffer-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${IS_DISABLED})

add_library(spm-counter-collection-callback-client SHARED)
target_sources(spm-counter-collection-callback-client PRIVATE callback_client.cpp
                                                              common.cpp common.hpp)
target_link_libraries(
    spm-counter-collection-callback-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(spm-counter-collection-callback)
target_sources(spm-counter-collection-callback PRIVATE main.cpp)
target_link_libraries(spm-counter-collection-callback
                      PRIVATE spm-counter-collection-callback-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV spm-counter-collection-callback-client)

set(spm-counter-collection-callback-env "${SPM_ENV}" "${PRELOAD_ENV}"
                                        "${LIBRARY_PATH_ENV}")

add_test(NAME spm-counter-collection-callback
         COMMAND $<TARGET_FILE:spm-counter-collection-callback>)

set_tests_properties(
    spm-counter-collection-callback
    PROPERTIES TIMEOUT
               120
               LABELS
               "samples"
               ENVIRONMENT
               "${spm-counter-collection-callback-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${IS_DISABLED})

add_library(spm-counter-collection-stop-inflight-client SHARED)
target_sources(spm-counter-collection-stop-inflight-client
               PRIVATE stop_inflight_client.cpp common.cpp common.hpp)
target_link_libraries(
    spm-counter-collection-stop-inflight-client
    PUBLIC rocprofiler-sdk::samples-build-flags
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-common-library)

set_source_files_properties(stop_inflight_main.cpp PROPERTIES LANGUAGE HIP)
add_executable(spm-counter-collection-stop-inflight)
target_sources(spm-counter-collection-stop-inflight PRIVATE stop_inflight_main.cpp)
target_link_libraries(
    spm-counter-collection-stop-inflight
    PRIVATE spm-counter-collection-stop-inflight-client Threads::Threads)

rocprofiler_samples_get_preload_env(PRELOAD_ENV
                                    spm-counter-collection-stop-inflight-client)

set(spm-counter-collection-stop-inflight-env "${SPM_ENV}" "${PRELOAD_ENV}"
                                             "${LIBRARY_PATH_ENV}")

add_test(NAME spm-counter-collection-stop-inflight
         COMMAND $<TARGET_FILE:spm-counter-collection-stop-inflight>)

set_tests_properties(
    spm-counter-collection-stop-inflight
    PROPERTIES TIMEOUT
               30
               LABELS
               "samples"
               ENVIRONMENT
               "${spm-counter-collection-stop-inflight-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${IS_DISABLED})
