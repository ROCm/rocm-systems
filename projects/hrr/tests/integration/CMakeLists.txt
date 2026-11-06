# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# hrr_api_matrix_expectations.h is GENERATED from the pair api_classes.json +
# api_matrix.yaml, both in ../../tools/api-matrix.  The build never runs the
# generators.  Regenerate after an upstream classification change or an
# api_matrix.yaml edit, from ../../tools/api-matrix:
#   ./derive_manifest.py
#   ./check_matrix.py --emit-cxx

# The workload translation units carry __global__ kernels, device built-ins
# (blockIdx/blockDim/threadIdx), hipLaunchKernelGGL, and HIP_SYMBOL, so they must
# be compiled with the HIP device toolchain rather than a plain C++ compiler.
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

# hrr_spt_workload_test.cc calls the ordinary hipMemset* names and relies on the
# per-thread-default-stream compile option to redirect them to the _spt entry
# points under test. This must stay per source so the other workloads keep
# resolving to the plain entry points.
if(HIP_PLATFORM STREQUAL "amd")
  set_source_files_properties(hrr_spt_workload_test.cc PROPERTIES
                              COMPILE_OPTIONS "-fgpu-default-stream=per-thread")
elseif(HIP_PLATFORM STREQUAL "nvidia")
  set_source_files_properties(hrr_spt_workload_test.cc PROPERTIES
                              COMPILE_OPTIONS "--default-stream=per-thread")
endif()

# ../include -> local harness headers.
# ../../playback -> hrr_reader.h / hip_playback.h (also propagated by hrr::reader).
target_include_directories(hrr-integration-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../playback
)

# hiprtc::hiprtc (hipRTC) is a separate ROCm CMake package from hip::host.
find_package(hiprtc CONFIG REQUIRED)

target_link_libraries(hrr-integration-tests PRIVATE
  hrr::reader
  hip::host
  hiprtc::hiprtc
  Catch2::Catch2WithMain)

# hrr_repro_workload_test.cc zeroes a device allocation with hsa_amd_memory_fill
# so the fill itself is never recorded. hip::host resolves ROCr for its own
# needs but does not carry it in its link interface, so a direct call from our
# objects needs the runtime named here. Windows has no ROCr and the call site is
# compiled out there.
if(NOT WIN32)
  find_package(hsa-runtime64 CONFIG REQUIRED)
  target_link_libraries(hrr-integration-tests PRIVATE hsa-runtime64::hsa-runtime64)
  # hsa-runtime64 is REQUIRED and linked just above, so the cases that call HSA
  # directly can compile here. Define the macro they are guarded by next to the
  # dependency so the two cannot drift apart; without it those cases vanish from
  # the binary with no signal. tests/config/integration.yaml is the backstop.
  target_compile_definitions(hrr-integration-tests PRIVATE HRR_HAVE_HSA)

  # hrr_disk_space_test.cc defines these for the HIP runtime to bind to. An
  # executable exports only what its link-time libraries reference, and the
  # runtime may name statvfs or fstatvfs by either name, so export all five.
  set(_hrr_disk_space_exports "${CMAKE_CURRENT_BINARY_DIR}/hrr_disk_space_exports.list")
  file(WRITE "${_hrr_disk_space_exports}" "{ statvfs; statvfs64; fstatvfs; fstatvfs64; fsync; };\n")
  target_link_options(hrr-integration-tests PRIVATE
                      "LINKER:--dynamic-list=${_hrr_disk_space_exports}")
  target_link_libraries(hrr-integration-tests PRIVATE ${CMAKE_DL_LIBS})
endif()

# Optional: hip_raw_trace, a genuine multi-threaded capture workload used only by
# Unit_HRR_MultiThreadRoundtrip. If it is not found, HRR_RAW_TRACE_EXE is defined
# empty and that test skips itself at run time.
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

# Capture honours HIP_HRR_DEBUG_ARGS only in a CLR built without NDEBUG
# (CMAKE_BUILD_TYPE=Debug). Tell the tests which kind of runtime they run against.
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

# The rocprofiler-register tool library hrr_dispatch_swap_test.cc loads into its
# workloads. libamdhip64 registers with rocprofiler-register on Linux only, and
# not in every Linux build: a test that needs the table checks at run time and
# skips when the runtime did not register.
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
