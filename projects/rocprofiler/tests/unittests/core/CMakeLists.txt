set (OLD_LIB_SRC
  ${LIB_DIR}/core/rocprofiler.cpp
  ${LIB_DIR}/core/gpu_command.cpp
  ${LIB_DIR}/core/proxy_queue.cpp
  ${LIB_DIR}/core/simple_proxy_queue.cpp
  ${LIB_DIR}/core/intercept_queue.cpp
  ${LIB_DIR}/core/metrics.cpp
  ${LIB_DIR}/core/activity.cpp
  ${LIB_DIR}/util/hsa_rsrc_factory.cpp
)

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

set(CORE_HW_DIR ${PROJECT_SOURCE_DIR}/src/core/hardware)
file(GLOB CORE_HW_SRC_FILES ${CORE_HW_DIR}/hsa_info.cpp)

set(CORE_HW_DIR ${PROJECT_SOURCE_DIR}/src/core/hardware)
file(GLOB CORE_HW_SRC_FILES ${CORE_HW_DIR}/hsa_info.cpp)

set(CORE_UTILS_DIR ${PROJECT_SOURCE_DIR}/src/utils)
file(GLOB CORE_UTILS_SRC_FILES ${CORE_UTILS_DIR}/*.cpp)

set(CORE_HSA_PACKETS_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa/packets)
file(GLOB CORE_HSA_PACKETS_SRC_FILES ${CORE_HSA_PACKETS_DIR}/packets_generator.cpp)

file(GLOB CORE_COUNTERS_SRC_FILES ${PROJECT_BINARY_DIR}/src/api/*_counter.cpp)

file(GLOB ROCPROFILER_SRC_API_FILES ${PROJECT_SOURCE_DIR}/src/api/*.cpp)

file(GLOB ROCPROFILER_SRC_PROFILER_FILES ${PROJECT_SOURCE_DIR}/src/core/session/profiler/profiler.cpp)
file(GLOB ROCPROFILER_TRACER_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/tracer/*.cpp)
file(GLOB ROCPROFILER_ROCTRACER_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/tracer/src/*.cpp)
file(GLOB ROCPROFILER_ATT_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/att/att.cpp)
file(GLOB ROCPROFILER_SRC_CLASS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/rocmtool.cpp)
file(GLOB ROCPROFILER_SPM_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/session/spm/spm.cpp)

set(ROCPROFILER_SRC_FILES ${ROCPROFILER_SRC_API_FILES} ${ROCPROFILER_SRC_CLASS_FILES} ${ROCPROFILER_SRC_PROFILER_FILES} ${ROCPROFILER_ATT_SRC_FILES})

set(CORE_HSA_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa)
file(GLOB CORE_HSA_SRC_FILES ${CORE_HSA_DIR}/*.cpp)

set(CORE_HSA_QUEUES_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa/queues)
file(GLOB CORE_HSA_QUEUES_SRC_FILES ${CORE_HSA_QUEUES_DIR}/*.cpp)

file(GLOB CORE_COUNTERS_PARENT_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/counters/*.cpp)
file(GLOB CORE_COUNTERS_METRICS_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/counters/metrics/*.cpp)
file(GLOB CORE_COUNTERS_MMIO_SRC_FILES ${PROJECT_SOURCE_DIR}/src/core/counters/mmio/*.cpp)

set(CORE_PC_SAMPLING_DIR ${PROJECT_SOURCE_DIR}/src/pcsampler)
file(GLOB CORE_PC_SAMPLING_FILES ${CORE_PC_SAMPLING_DIR}/core/*.cpp ${CORE_PC_SAMPLING_DIR}/gfxip/*.cpp ${CORE_PC_SAMPLING_DIR}/session/*.cpp)

# Compiling gtests
add_executable(runCoreUnitTests ${CMAKE_CURRENT_SOURCE_DIR}/gtests_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/session/session_gtest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory/memory_gtest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hardware/hsa_info_gtest.cpp
  ${CORE_MEMORY_SRC_FILES}
  ${CORE_SESSION_SRC_FILES}
  ${CORE_FILTER_SRC_FILES}
  ${CORE_DEVICE_PROFILING_SRC_FILES}
  ${CORE_COUNTERS_SAMPLER_SRC_FILES}
  ${CORE_HW_SRC_FILES}
  ${CORE_UTILS_SRC_FILES}
  ${ROCPROFILER_SRC_FILES}
  ${CORE_HSA_SRC_FILES}
  ${ROCPROFILER_SPM_SRC_FILES}
  ${CORE_HSA_PACKETS_SRC_FILES}
  ${CORE_COUNTERS_SRC_FILES}
  ${CORE_HSA_QUEUES_SRC_FILES}
  ${ROCPROFILER_TRACER_SRC_FILES}
  ${ROCPROFILER_ROCTRACER_SRC_FILES}
  ${CORE_COUNTERS_METRICS_SRC_FILES}
  ${CORE_COUNTERS_MMIO_SRC_FILES}
  ${CORE_COUNTERS_PARENT_SRC_FILES}
  ${CORE_PC_SAMPLING_FILES}
  ${OLD_LIB_SRC})

target_include_directories(runCoreUnitTests PRIVATE ${PROJECT_SOURCE_DIR}
  ${LIB_DIR} ${ROOT_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/inc
  ${PROJECT_SOURCE_DIR}/tests/unittests/profiler
  ${PROJECT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}/rocprofiler)

target_compile_definitions(runCoreUnitTests
  PUBLIC AMD_INTERNAL_BUILD
  PRIVATE PROF_API_IMPL HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)

# Link test executable against gtest & gtest_main
target_link_libraries(runCoreUnitTests PRIVATE ${ROCPROFILER_TARGET} ${AQLPROFILE_LIB}
  hsa-runtime64::hsa-runtime64 c stdc++
  GTest::gtest GTest::gtest_main stdc++fs dl ${PCIACCESS_LIBRARIES})

add_dependencies(tests runCoreUnitTests)
install(TARGETS runCoreUnitTests RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests COMPONENT tests)
add_test(AllTests runCoreUnitTests)
