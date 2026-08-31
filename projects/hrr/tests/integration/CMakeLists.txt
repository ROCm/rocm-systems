# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/HrrCaptureLib.cmake")

# HRR GPU capture/replay integration tests.
#
# hrr_roundtrip_test.cc is the driver: it sets HIP_HRR_CAPTURE_OUTPUT, spawns the
# hidden [.][hrr-direct] workloads from this same binary as subprocesses, then
# runs hrr-playback to validate replayed device-to-host buffers byte-for-byte.
# The workload translation units (hrr_workload / hrr_repro_workload / hrr_spt_workload)
# provide those hidden _Direct cases and must be linked into the same executable.

# The workload translation units carry __global__ kernels, device built-ins
# (blockIdx/blockDim/threadIdx), hipLaunchKernelGGL, and HIP_SYMBOL, so they must
# be compiled with the HIP device toolchain rather than a plain C++ compiler.
# hrr_roundtrip_test.cc is a host-only driver (subprocess orchestration + hipRTC
# host calls) and stays a plain C++ translation unit.
set(HRR_GPU_WORKLOAD_SOURCES
  hrr_workload_test.cc
  hrr_repro_workload_test.cc
  hrr_spt_workload_test.cc
)

if(NOT DEFINED HIP_PLATFORM)
  set(HIP_PLATFORM "amd")
endif()

if(HIP_PLATFORM STREQUAL "amd")
  enable_language(HIP)
  set_source_files_properties(${HRR_GPU_WORKLOAD_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

add_executable(hrr-integration-tests
  ${HRR_GPU_WORKLOAD_SOURCES}
  hrr_roundtrip_test.cc
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

# ${CMAKE_CURRENT_SOURCE_DIR}/.. -> local harness headers.
# ../../playback -> hrr_reader.h / hip_playback.h (also propagated by hrr::reader).
target_include_directories(hrr-integration-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../playback
)

# hiprtc::hiprtc (hipRTC) is a separate ROCm CMake package from hip::host.
find_package(hiprtc CONFIG REQUIRED)

target_link_libraries(hrr-integration-tests PRIVATE
  hrr::reader
  hip::host
  hiprtc::hiprtc
  Catch2::Catch2WithMain)

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

# Capture-enabled libamdhip64 must match this branch's wire format. Fail at
# configure time rather than at replay with opaque payload-size errors.
hrr_resolve_capture_lib(HRR_CAPTURE_LIB_DIR)
if(NOT HRR_CAPTURE_LIB_DIR)
  message(FATAL_ERROR
    "HRR integration tests require capture-enabled libamdhip64 from the same tree.\n"
    "Build amdhip64, then reconfigure with -DHRR_CLR_LIB=<clr-build>/hipamd/lib:\n"
    "  cmake -S projects/clr -B build/clr -GNinja "
    "-DHIP_COMMON_DIR=projects/hip -DROCM_PATH=\$ROCM_PATH "
    "-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd\n"
    "  ninja -C build/clr amdhip64\n"
    "  cmake ... -DHRR_CLR_LIB=\$PWD/build/clr/hipamd/lib")
endif()
message(STATUS "HRR capture lib: ${HRR_CAPTURE_LIB_DIR}")

target_compile_definitions(hrr-integration-tests PRIVATE
  HRR_API_ARGS_IMPLEMENTATION
  __HIP_PLATFORM_AMD__
  HRR_TEST_EXE="$<TARGET_FILE:hrr-integration-tests>"
  HRR_PLAYBACK_EXE="$<TARGET_FILE:hrr-playback>"
  HRR_RAW_TRACE_EXE="${HRR_RAW_TRACE_EXE}"
  ROCM_BIN_PATH="${ROCM_PATH}/bin"
  HRR_CLR_LIB="${HRR_CAPTURE_LIB_DIR}"
)

hrr_apply_capture_lib_rpath("${HRR_CAPTURE_LIB_DIR}" hrr-integration-tests hrr-playback)

# The integration tests spawn hrr-playback; make sure it is built first.
add_dependencies(hrr-integration-tests hrr-playback)

add_test(NAME hrr-integration-tests COMMAND hrr-integration-tests)
if(CMAKE_HOST_UNIX)
  set_tests_properties(hrr-integration-tests PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${HRR_CAPTURE_LIB_DIR}")
endif()
