include(CheckCSourceCompiles)
# ########################################################################################
# ########################################################################################
# General Requirements
# ########################################################################################
# ########################################################################################
get_property(
    HSA_RUNTIME_INCLUDE_DIRECTORIES
    TARGET hsa-runtime64::hsa-runtime64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(
    HSA_H hsa.h
    PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES hsa
    NO_DEFAULT_PATH REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)
include_directories(${HSA_RUNTIME_INC_PATH})

# Set the HIP language runtime link flags as FindHIP does not set them.
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ROCM_PATH}/lib/cmake/hip")
set(CMAKE_HIP_ARCHITECTURES OFF)
if(DEFINED ROCM_PATH)
    set(HIP_ROOT_DIR "${ROCM_PATH}/bin")
endif()
find_package(HIP REQUIRED MODULE)

find_package(
    Clang REQUIRED CONFIG
    PATHS "${ROCM_PATH}"
    PATH_SUFFIXES "llvm/lib/cmake/clang")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules"
                      "${ROCM_PATH}/lib/cmake/hip")
find_package(LibElf REQUIRED)
find_package(LibDw REQUIRED)

# Add a custom targets to build and run all the tests
add_custom_target(samples ALL)
add_dependencies(samples rocprofiler-v2)
add_custom_target(
    run-samples
    COMMAND ${PROJECT_BINARY_DIR}/samples/run_samples.sh
    DEPENDS samples)

file(GLOB ROCPROFILER_UTIL_SRC_FILES ${PROJECT_SOURCE_DIR}/src/utils/helper.cpp)
# ########################################################################################

function(rocprofiler_sample_add_test _TARGET _ARGS)
    if(TARGET ${_TARGET})
        if(NOT TEST ${_TARGET})
            add_test(
                NAME ${_TARGET}
                COMMAND $<TARGET_FILE:${_TARGET}> ${_ARGS}
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()
        set_tests_properties(
            ${_TARGET}
            PROPERTIES
                LABELS
                "samples"
                ENVIRONMENT
                "ROCPROFILER_METRICS_PATH=${PROJECT_BINARY_DIR}/libexec/rocprofiler/counters/derived_counters.xml;${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
                RUN_SERIAL
                TRUE
                ${ARGN})
    endif()
endfunction()

function(rocprofiler_sample_add_executable _TARGET)
    hip_add_executable(${_TARGET} ${ARGN})
    rocprofiler_sample_add_test(${_TARGET} "")
endfunction()

# ########################################################################################
# ########################################################################################
# Samples Build & Run Script
# ########################################################################################
# ########################################################################################

# ########################################################################################
# Profiler Samples
# ########################################################################################

# Build Kernel No Replay Sample
set_source_files_properties(profiler/kernel_profiling_no_replay_sample.cpp
                            PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_sample_add_executable(
    profiler_kernel_no_replay profiler/kernel_profiling_no_replay_sample.cpp
    ${ROCPROFILER_UTIL_SRC_FILES})
target_include_directories(
    profiler_kernel_no_replay PRIVATE ${PROJECT_SOURCE_DIR}
                                      ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(profiler_kernel_no_replay PRIVATE rocprofiler-v2 amd_comgr)
target_link_options(profiler_kernel_no_replay PRIVATE "-Wl,--build-id=md5")
add_dependencies(samples profiler_kernel_no_replay)
install(TARGETS profiler_kernel_no_replay
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples
                COMPONENT samples)

# Build Device Profiling Sample
set_source_files_properties(profiler/device_profiling_sample.cpp
                            PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_sample_add_executable(
    profiler_device_profiling profiler/device_profiling_sample.cpp
    ${ROCPROFILER_UTIL_SRC_FILES})
target_include_directories(
    profiler_device_profiling PRIVATE ${PROJECT_SOURCE_DIR}
                                      ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(profiler_device_profiling PRIVATE rocprofiler-v2 amd_comgr)
target_link_options(profiler_device_profiling PRIVATE "-Wl,--build-id=md5")
add_dependencies(samples profiler_device_profiling)
install(TARGETS profiler_device_profiling
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples
                COMPONENT samples)
set_tests_properties(profiler_device_profiling PROPERTIES DISABLED TRUE)

# Build Counters Sampling example
set_source_files_properties(counters_sampler/pcie_counters_example.cpp
                            PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_sample_add_executable(
    pcie_counters_sampler counters_sampler/pcie_counters_example.cpp
    ${ROCPROFILER_UTIL_SRC_FILES})
target_include_directories(
    pcie_counters_sampler PRIVATE ${PROJECT_SOURCE_DIR}
                                  ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(pcie_counters_sampler PRIVATE rocprofiler-v2 amd_comgr)
target_link_options(pcie_counters_sampler PRIVATE "-Wl,--build-id=md5")
add_dependencies(samples pcie_counters_sampler)
install(TARGETS pcie_counters_sampler
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples
                COMPONENT samples)

# Build XGMI Counters Sampling example
set_source_files_properties(counters_sampler/xgmi_counters_sampler_example.cpp
                            PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_sample_add_executable(
    xgmi_counters_sampler counters_sampler/xgmi_counters_sampler_example.cpp
    ${ROCPROFILER_UTIL_SRC_FILES})
target_include_directories(
    xgmi_counters_sampler PRIVATE ${PROJECT_SOURCE_DIR}
                                  ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(xgmi_counters_sampler PRIVATE rocprofiler-v2 amd_comgr)
target_link_options(xgmi_counters_sampler PRIVATE "-Wl,--build-id=md5")
add_dependencies(samples xgmi_counters_sampler)
install(TARGETS xgmi_counters_sampler
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples
                COMPONENT samples)
set_tests_properties(xgmi_counters_sampler PROPERTIES DISABLED TRUE)

# ########################################################################################

# ########################################################################################
# Tracer Samples
# ########################################################################################

# Build HIP/HSA Trace Sample
set_source_files_properties(tracer/sample.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_sample_add_executable(tracer_hip_hsa tracer/sample.cpp
                                  ${ROCPROFILER_UTIL_SRC_FILES})
target_include_directories(tracer_hip_hsa PRIVATE ${PROJECT_SOURCE_DIR}
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(tracer_hip_hsa PRIVATE rocprofiler-v2 amd_comgr)
target_link_options(tracer_hip_hsa PRIVATE "-Wl,--build-id=md5")
add_dependencies(samples tracer_hip_hsa)
install(TARGETS tracer_hip_hsa
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples
                COMPONENT samples)

# Build HIP/HSA Trace with async output api trace data Sample
set_source_files_properties(tracer/sample_async.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT
                                                               1)
rocprofiler_sample_add_executable(tracer_hip_hsa_async tracer/sample_async.cpp
                                  ${ROCPROFILER_UTIL_SRC_FILES})
target_include_directories(
    tracer_hip_hsa_async PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(tracer_hip_hsa_async PRIVATE rocprofiler-v2 amd_comgr)
target_link_options(tracer_hip_hsa_async PRIVATE "-Wl,--build-id=md5")
add_dependencies(samples tracer_hip_hsa_async)
install(TARGETS tracer_hip_hsa_async
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/samples
                COMPONENT samples)

# ########################################################################################
# Scripts to run samples
# ########################################################################################

# Copy run_samples script to samples folder
configure_file(run_samples.sh ${PROJECT_BINARY_DIR}/samples COPYONLY)

# ########################################################################################
