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

# Setup unit testing env

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
file(GLOB CORE_PROFILING_LOCK_FILES ${PROJECT_SOURCE_DIR}/src/core/profiling_lock.cpp)

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
file(GLOB HSASingleton_TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/HSASingleton/*.cpp)
file(GLOB ROCProfiler_Singleton_TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ROCProfiler_Singleton/*.cpp)
file(GLOB GTEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Getting HSA Include Directory
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
# ########################################################################################
# ROCProfiler Tracer HIP/HSA Parsing
# ########################################################################################
get_property(
    HIP_INCLUDE_DIRECTORIES
    TARGET hip::amdhip64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(
    HIP_RUNTIME_API_H hip_runtime_api.h
    PATHS ${HIP_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES hip
    NO_DEFAULT_PATH REQUIRED)

# # Generate the HSA wrapper functions header
add_custom_command(
    OUTPUT hsa_prof_str.h hsa_prof_str.inline.h
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/hsaap.py
            ${CMAKE_CURRENT_BINARY_DIR} "${HSA_RUNTIME_INC_PATH}" > /dev/null
    DEPENDS ${PROJECT_SOURCE_DIR}/script/hsaap.py
            "${HSA_RUNTIME_INC_PATH}/hsa.h"
            "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
            "${HSA_RUNTIME_INC_PATH}/hsa_ext_image.h"
            "${HSA_RUNTIME_INC_PATH}/hsa_api_trace.h"
    COMMENT "Generating hsa_prof_str.h,hsa_prof_str.inline.h...")

# # Generate the HSA pretty printers
add_custom_command(
    OUTPUT hsa_ostream_ops.h
    COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa.h" -o hsa.h.i
    COMMAND ${CMAKE_C_COMPILER} -E "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h" -o
            hsa_ext_amd.h.i
    BYPRODUCTS hsa.h.i hsa_ext_amd.h.i
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py -in
            hsa.h.i,hsa_ext_amd.h.i -out hsa_ostream_ops.h > /dev/null
    DEPENDS ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py
            "${HSA_RUNTIME_INC_PATH}/hsa.h" "${HSA_RUNTIME_INC_PATH}/hsa_ext_amd.h"
    COMMENT "Generating hsa_ostream_ops.h...")

get_property(
    HIP_INCLUDE_DIRECTORIES
    TARGET hip::amdhip64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(
    HIP_RUNTIME_API_H hip_runtime_api.h
    PATHS ${HIP_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES hip
    NO_DEFAULT_PATH REQUIRED)

# Generate the HIP pretty printers
add_custom_command(
    OUTPUT hip_ostream_ops.h
    COMMAND
        ${CMAKE_C_COMPILER}
        "$<$<BOOL:${HIP_INCLUDE_DIRECTORIES}>:-I$<JOIN:${HIP_INCLUDE_DIRECTORIES},$<SEMICOLON>-I>>"
        -E "${PROJECT_SOURCE_DIR}/src/core/session/tracer/src/hip_full_api.h" -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -o
        hip_runtime_api.h.i
    BYPRODUCTS hip_runtime_api.h.i
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py -in
            hip_runtime_api.h.i -out hip_ostream_ops.h > /dev/null
    DEPENDS ${PROJECT_SOURCE_DIR}/script/gen_ostream_ops.py "${PROJECT_SOURCE_DIR}/src/core/session/tracer/src/hip_full_api.h"
    COMMENT "Generating hip_ostream_ops.h..."
    COMMAND_EXPAND_LISTS)

set(GENERATED_SOURCES hip_ostream_ops.h hsa_prof_str.h hsa_ostream_ops.h
                      hsa_prof_str.inline.h)

set(runCoreUnitTests_SOURCES
    ${GTEST_SRC_FILES}
    ${GENERATED_SOURCES}
    ${HSASingleton_TEST_SRC_FILES}
    ${ROCProfiler_Singleton_TEST_SRC_FILES}
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
    ${CORE_PROFILING_LOCK_FILES})


add_executable(runCoreUnitTests ${runCoreUnitTests_SOURCES})
target_include_directories(
     runCoreUnitTests
     PUBLIC ${HIP_INCLUDE_DIRECTORIES}
           ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/rocprofiler/v2>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE ${LIB_DIR} ${ROOT_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${PROJECT_SOURCE_DIR})


target_compile_definitions(
     runCoreUnitTests
     PUBLIC AMD_INTERNAL_BUILD
     PRIVATE PROF_API_IMPL HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)
target_link_libraries(runCoreUnitTests PRIVATE rocprofiler::build-flags)

set_target_properties(runCoreUnitTests PROPERTIES
 INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests")
target_link_libraries(
    runCoreUnitTests
    PRIVATE ${AQLPROFILE_LIB}
            test_hsatool_library
            hsa-runtime64::hsa-runtime64
            Threads::Threads
            GTest::gtest GTest::gtest_main
            atomic
            numa
            dl
            c
            stdc++
            stdc++fs
            amd_comgr
            dw
            elf
            ${PCIACCESS_LIBRARIES}
            rocprofiler::memcheck)

add_dependencies(tests runCoreUnitTests)
install(TARGETS runCoreUnitTests
     RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests
     COMPONENT tests)


include(GoogleTest)

set(GTEST_DISCOVER_TESTS_TARGET runCoreUnitTests)
set(GTEST_DISCOVER_TESTS_LABELS "v2" "unittests")
set(GTEST_DISCOVER_TESTS_ENVIRONMENT ${ROCPROFILER_MEMCHECK_PRELOAD_ENV})
configure_file(
     ${PROJECT_SOURCE_DIR}/cmake_modules/Templates/gtest_discover_tests_properties.cmake.in
     ${CMAKE_CURRENT_BINARY_DIR}/runUnitTests_TestProperties.cmake @ONLY)

if(NOT ROCPROFILER_MEMCHECK MATCHES "(Thread|Address)Sanitizer")
     gtest_discover_tests(runCoreUnitTests)
     set_property(
          DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
          APPEND
          PROPERTY TEST_INCLUDE_FILES
          ${CMAKE_CURRENT_BINARY_DIR}/runUnitTests_TestProperties.cmake)
else()
     gtest_add_tests(
          TARGET runCoreUnitTests
          SOURCES "${runUnitTests_SOURCES}"
          TEST_LIST runUnitTests_TESTS)
     include(${CMAKE_CURRENT_BINARY_DIR}/runUnitTests_TestProperties.cmake)
endif()

# for the *_FilePlugin tests
if(NOT EXISTS "${PROJECT_BINARY_DIR}/test-output")
     file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/test-output")
endif()