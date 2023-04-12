# ##############################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ##############################################################################

find_library(PCIACCESS_LIBRARIES pciaccess REQUIRED)

enable_testing()
find_package(GTest REQUIRED)

# Getting Source files for ROCProfiler, Hardware, HSA, Memory, Session, Counters, Utils
set(CORE_MEMORY_DIR ${PROJECT_SOURCE_DIR}/src/core/memory)
file(GLOB CORE_MEMORY_SRC_FILES ${CORE_MEMORY_DIR}/*.cpp)

set(CORE_SESSION_DIR ${PROJECT_SOURCE_DIR}/src/core/session)
file(GLOB CORE_SESSION_SRC_FILES ${CORE_SESSION_DIR}/session.cpp)
file(GLOB CORE_FILTER_SRC_FILES ${CORE_SESSION_DIR}/filter.cpp)
file(GLOB CORE_DEVICE_PROFILING_SRC_FILES ${CORE_SESSION_DIR}/device_profiling.cpp)
file(GLOB CORE_COUNTERS_SAMPLER_SRC_FILES ${CORE_SESSION_DIR}/counters_sampler.cpp)

set(CORE_HW_DIR ${PROJECT_SOURCE_DIR}/src/core/hardware)
file(GLOB CORE_HW_SRC_FILES ${CORE_HW_DIR}/hsa_info.cpp)

set(CORE_HW_DIR ${PROJECT_SOURCE_DIR}/src/core/hardware)
file(GLOB CORE_HW_SRC_FILES ${CORE_HW_DIR}/hsa_info.cpp)

set(CORE_UTILS_DIR ${PROJECT_SOURCE_DIR}/src/utils)
file(GLOB CORE_UTILS_SRC_FILES ${CORE_UTILS_DIR}/*.cpp)

set(CORE_HSA_PACKETS_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa/packets)
file(GLOB CORE_HSA_PACKETS_SRC_FILES ${CORE_HSA_PACKETS_DIR}/packets_generator.cpp)

file(GLOB CORE_COUNTERS_SRC_FILES ${PROJECT_BINARY_DIR}/src/api/*_counter.cpp)

file(GLOB ROCPROFILER_SRC_PROFILER_FILES
     ${PROJECT_SOURCE_DIR}/src/core/session/profiler/profiler.cpp)
file(GLOB ROCPROFILER_TRACER_SRC_FILES
     ${PROJECT_SOURCE_DIR}/src/core/session/tracer/*.cpp)
file(GLOB ROCPROFILER_ROCTRACER_SRC_FILES
     ${PROJECT_SOURCE_DIR}/src/core/session/tracer/src/*.cpp)
     file(GLOB ROCPROFILER_ATT_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/att/*.cpp)
     file(GLOB ROCPROFILER_SRC_CLASS_FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler_singleton.cpp)
     file(GLOB ROCPROFILER_ISA_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/isa_capture/*.cpp)
     file(GLOB ROCPROFILER_SPM_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/spm/spm.cpp)
     file(GLOB ROCPROFILER_SRC_API_FILES ${PROJECT_SOURCE_DIR}/src/api/*.cpp)
     set(ROCPROFILER_SRC_FILES ${ROCPROFILER_SRC_API_FILES} ${ROCPROFILER_ATT_SRC_FILES}
          ${ROCPROFILER_ISA_SRC_FILES} ${ROCPROFILER_SRC_PROFILER_FILES} ${ROCPROFILER_ATT_SRC_FILES})

set(CORE_HSA_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa)
file(GLOB CORE_HSA_SRC_FILES ${CORE_HSA_DIR}/*.cpp)

set(CORE_HSA_QUEUES_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa/queues)
file(GLOB CORE_HSA_QUEUES_SRC_FILES ${CORE_HSA_QUEUES_DIR}/*.cpp)

set(CORE_PC_SAMPLING_DIR ${PROJECT_SOURCE_DIR}/src/pcsampler)
file(GLOB CORE_PC_SAMPLING_FILES ${CORE_PC_SAMPLING_DIR}/core/*.cpp
     ${CORE_PC_SAMPLING_DIR}/gfxip/*.cpp ${CORE_PC_SAMPLING_DIR}/session/*.cpp)

# Compiling gtests
file(GLOB ROCPROFILER_TOOL_SRC_FILES ${PROJECT_SOURCE_DIR}/src/tools/tool.cpp)

file(GLOB CORE_COUNTERS_PARENT_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/counters/*.cpp)
file(GLOB CORE_COUNTERS_METRICS_SRC_FILES
     ${PROJECT_SOURCE_DIR}/src/core/counters/metrics/*.cpp)
file(GLOB CORE_COUNTERS_MMIO_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/counters/mmio/*.cpp)
set(GTEST_MAIN_DIR ${PROJECT_SOURCE_DIR}/tests-v2/unittests/core)
file(GLOB GTEST_MAIN_SRC_FILE ${GTEST_MAIN_DIR}/gtests_main.cpp)
add_executable(
    runCoreUnitTests
    ${CORE_MEMORY_SRC_FILES}
    ${CORE_SESSION_SRC_FILES}
    ${CORE_FILTER_SRC_FILES}
    ${CORE_DEVICE_PROFILING_SRC_FILES}
    ${CORE_COUNTERS_SAMPLER_SRC_FILES}
    ${CORE_HW_SRC_FILES}
    ${CORE_UTILS_SRC_FILES}
    ${ROCPROFILER_SPM_SRC_FILES}
    ${ROCPROFILER_SRC_FILES}
    ${CORE_HSA_SRC_FILES}
    ${CORE_HSA_PACKETS_SRC_FILES}
    ${CORE_COUNTERS_SRC_FILES}
    ${CORE_HSA_QUEUES_SRC_FILES}
    ${ROCPROFILER_TRACER_SRC_FILES}
    ${ROCPROFILER_ROCTRACER_SRC_FILES}
    ${CORE_COUNTERS_METRICS_SRC_FILES}
    ${CORE_COUNTERS_MMIO_SRC_FILES}
    ${CORE_COUNTERS_PARENT_SRC_FILES}
    ${CORE_PC_SAMPLING_FILES}
    ${GTEST_MAIN_SRC_FILE}
    ${CMAKE_CURRENT_SOURCE_DIR}/ROCProfiler_Singleton/ROCProfiler_Singleton_unittests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HSASingleton/HSASingleton_unittests.cpp
   )

target_include_directories(
    runCoreUnitTests
    PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/inc
            ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR}
            ${PROJECT_BINARY_DIR}/rocprofiler)

target_compile_definitions(
    runCoreUnitTests
    PUBLIC AMD_INTERNAL_BUILD
    PRIVATE PROF_API_IMPL HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)


target_link_libraries(
    runCoreUnitTests PRIVATE  rocprofiler_tool test_hsatool_library ${AQLPROFILE_LIB} hsa-runtime64::hsa-runtime64
                         GTest::gtest GTest::gtest_main stdc++fs ${PCIACCESS_LIBRARIES})


add_dependencies(tests runCoreUnitTests)
install(TARGETS runCoreUnitTests
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests
                COMPONENT tests)
add_test(AllTests runCoreUnitTests)