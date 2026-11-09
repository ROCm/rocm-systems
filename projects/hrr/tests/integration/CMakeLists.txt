# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# hrr_api_matrix_expectations.h is generated from api_classes.json and api_matrix.yaml in ../../tools/api-matrix.
# The build never runs the generators. After a classification or api_matrix.yaml change, regenerate there with:
#   ./derive_manifest.py
#   ./check_matrix.py --emit-cxx

# These sources use __global__ kernels, device built-ins and hipLaunchKernelGGL, so they need the HIP device toolchain.
set(HRR_GPU_WORKLOAD_SOURCES
  hrr_workload_test.cc
  hrr_repro_workload_test.cc
  hrr_spt_workload_test.cc
  hrr_regions_test.cc
  hrr_api_matrix_workload_test.cc
  hrr_kernel_args_test.cc
  hrr_kernel_signature_test.cc
  hrr_pinned_host_test.cc
  hrr_va_placement_test.cc
  hrr_disk_space_test.cc
  hrr_host_alloc_budget_test.cc
)

if(NOT DEFINED HIP_PLATFORM)
  set(HIP_PLATFORM "amd")
endif()

if(HIP_PLATFORM STREQUAL "amd" OR HIP_PLATFORM STREQUAL "nvidia")
  enable_language(HIP)
  set_source_files_properties(${HRR_GPU_WORKLOAD_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

add_executable(hrr-integration-tests
  ${HRR_GPU_WORKLOAD_SOURCES}
  hrr_pitched_d2h_test.cc
  hrr_roundtrip_test.cc
  hrr_api_matrix_test.cc
  hrr_dispatch_swap_test.cc
  hrr_capture_access_test.cc
  hrr_fatbin_header_test.cc
  hrr_capture_capabilities_test.cc
  hrr_secure_exec_test.cc
  hrr_clock_hook.cc
)
set_target_properties(hrr-integration-tests PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED ON)

# hrr_spt_workload_test.cc calls the plain hipMemset* names and relies on the per-thread-default-stream option
# to redirect them to the _spt entry points. Keep the option per source so other workloads use the plain ones.
if(HIP_PLATFORM STREQUAL "amd")
  set_source_files_properties(hrr_spt_workload_test.cc PROPERTIES
                              COMPILE_OPTIONS "-fgpu-default-stream=per-thread")
elseif(HIP_PLATFORM STREQUAL "nvidia")
  set_source_files_properties(hrr_spt_workload_test.cc PROPERTIES
                              COMPILE_OPTIONS "--default-stream=per-thread")
endif()

target_include_directories(hrr-integration-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../playback
)

find_package(hiprtc CONFIG REQUIRED)

target_link_libraries(hrr-integration-tests PRIVATE
  hrr::reader
  hip::host
  hiprtc::hiprtc
  Catch2::Catch2WithMain)

# hrr_repro_workload_test.cc zeroes a device allocation with hsa_amd_memory_fill so the fill is never recorded.
# hip::host does not carry ROCr in its link interface, so link it directly. Windows has no ROCr and compiles the call out.
if(NOT WIN32)
  find_package(hsa-runtime64 CONFIG REQUIRED)
  target_link_libraries(hrr-integration-tests PRIVATE hsa-runtime64::hsa-runtime64)
  # Define the macro next to the dependency so the two cannot drift; without it the HSA cases vanish from the binary silently.
  # tests/config/integration.yaml is the backstop.
  target_compile_definitions(hrr-integration-tests PRIVATE HRR_HAVE_HSA)

  # hrr_disk_space_test.cc defines these for the HIP runtime to bind to. An executable exports only symbols its
  # libraries reference, and the runtime may use either statvfs name, so export all five.
  set(_hrr_disk_space_exports "${CMAKE_CURRENT_BINARY_DIR}/hrr_disk_space_exports.list")
  file(WRITE "${_hrr_disk_space_exports}" "{ statvfs; statvfs64; fstatvfs; fstatvfs64; fsync; };\n")
  target_link_options(hrr-integration-tests PRIVATE
                      "LINKER:--dynamic-list=${_hrr_disk_space_exports}")
  target_link_libraries(hrr-integration-tests PRIVATE ${CMAKE_DL_LIBS})
endif()

# hip_raw_trace is optional, used only by Unit_HRR_MultiThreadRoundtrip.
# When it is missing, HRR_RAW_TRACE_EXE is empty and that test skips at run time.
if(NOT DEFINED HRR_RAW_TRACE_EXE)
  find_program(_hrr_raw_trace_found NAMES hip_raw_trace hip_raw_trace.exe
    HINTS
      "${CMAKE_INSTALL_PREFIX}/bin"
      "${ROCM_PATH}/bin"
  )
  if(_hrr_raw_trace_found)
    set(HRR_RAW_TRACE_EXE "${_hrr_raw_trace_found}")
  else()
    set(HRR_RAW_TRACE_EXE "")
  endif()
endif()

# Capture honours HIP_HRR_DEBUG_ARGS only in a CLR built without NDEBUG (Debug).
# This option tells the tests which kind of runtime they run against.
option(HRR_TEST_CLR_DEBUG_FLAGS "The HIP runtime under test is a Debug build" OFF)
if(HRR_TEST_CLR_DEBUG_FLAGS)
  target_compile_definitions(hrr-integration-tests PRIVATE HRR_TEST_CLR_DEBUG_FLAGS)
endif()

target_compile_definitions(hrr-integration-tests PRIVATE
  HRR_API_ARGS_IMPLEMENTATION
  __HIP_PLATFORM_AMD__
  HRR_TEST_EXE="$<TARGET_FILE:hrr-integration-tests>"
  HRR_PLAYBACK_EXE="$<TARGET_FILE:hrr-playback>"
  HRR_RAW_TRACE_EXE="${HRR_RAW_TRACE_EXE}"
  ROCM_BIN_PATH="${ROCM_PATH}/bin"
)

add_dependencies(hrr-integration-tests hrr-playback)

# Tool library that hrr_dispatch_swap_test.cc loads into its workloads. libamdhip64 registers with
# rocprofiler-register on Linux only, and not in every build, so tests check at run time and skip.
if(NOT WIN32)
  add_library(hrr-test-dispatch-tool MODULE hrr_dispatch_tool.cc)
  set_target_properties(hrr-test-dispatch-tool PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden)
  target_link_libraries(hrr-test-dispatch-tool PRIVATE ${CMAKE_DL_LIBS})
  find_package(Threads REQUIRED)
  target_link_libraries(hrr-integration-tests PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
  target_compile_definitions(hrr-integration-tests PRIVATE
    HRR_DISPATCH_TOOL="$<TARGET_FILE:hrr-test-dispatch-tool>")
  add_dependencies(hrr-integration-tests hrr-test-dispatch-tool)
endif()

add_test(NAME hrr-integration-tests COMMAND hrr-integration-tests)
