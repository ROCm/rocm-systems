# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# HRR (HIP Record & Replay) tests
#
# Three source files compiled into HrrTest:
#
#   hrr_format.cc        — CPU-only: archive format I/O unit tests (no GPU).
#   hrr_workload.cc      — Hidden ([.][hrr-direct]) GPU test cases invoked as
#                          subprocesses by hrr_roundtrip.cc; provides clean
#                          Windows process context for HIP + capture layer.
#   hrr_roundtrip.cc     — Round-trip: sets HIP_HRR_CAPTURE_OUTPUT, spawns
#                          Unit_HRR_GpuWorkload_Direct / Unit_HRR_GraphWorkload_Direct
#                          as subprocesses, then runs hrr-playback to validate
#                          replayed D2H buffers byte-for-byte against captured blobs.
#
# hrr_reader.cpp is compiled directly into the test executable so the format
# tests can exercise the reader without a separate install step.

if(NOT DEFINED HRR_CLR_DIR)
  # Default: clr and hip-tests are siblings under projects/
  get_filename_component(HRR_CLR_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../clr/hipamd/src/hrr"
    ABSOLUTE)
endif()
if(NOT IS_DIRECTORY "${HRR_CLR_DIR}")
  message(FATAL_ERROR "HRR_CLR_DIR not found: ${HRR_CLR_DIR}\n"
                      "Pass -DHRR_CLR_DIR=<path> to point to clr/hipamd/src/hrr")
endif()

set(TEST_SRC
    hrr_format.cc
    hrr_workload.cc
    hrr_roundtrip.cc
    ${HRR_CLR_DIR}/playback/hrr_reader.cpp
)

hip_add_exe_to_target(NAME HrrTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc::hiprtc
                      PROPERTY CXX_STANDARD 17)

target_include_directories(HrrTest PRIVATE
    ${HRR_CLR_DIR}
    ${HRR_CLR_DIR}/playback
)

# hrr_api_args.h uses a compile-time guard to emit the hrr_api_names[] string
# table in exactly one translation unit.
target_compile_definitions(HrrTest PRIVATE
    HRR_API_ARGS_IMPLEMENTATION
    __HIP_PLATFORM_AMD__
)

# ---------------------------------------------------------------------------
# hrr_roundtrip.cc needs:
#   HRR_TEST_EXE     — absolute path to HrrTest.exe (the capture subprocess)
#   HRR_PLAYBACK_EXE — absolute path to hrr-playback.exe
#   ROCM_BIN_PATH    — prepended to PATH in the child so it finds amdhip64_7.dll
#                      (CTest/bash PATH does not include the ROCm bin directory)
# ---------------------------------------------------------------------------
if(HIP_PLATFORM STREQUAL "amd")
  if(NOT DEFINED HRR_PLAYBACK_EXE)
    find_program(HRR_PLAYBACK_EXE NAMES hrr-playback hrr-playback.exe
      HINTS
        "${CMAKE_INSTALL_PREFIX}/bin"
        "${ROCM_PATH}/bin"
    )
  endif()
  if(NOT HRR_PLAYBACK_EXE)
    message(FATAL_ERROR "hrr-playback not found.\n"
                        "Pass -DHRR_PLAYBACK_EXE=<path> to cmake.")
  endif()
  message(STATUS "HRR playback: ${HRR_PLAYBACK_EXE}")

  # Optional: hip_raw_trace.exe for Unit_HRR_MultiThreadRoundtrip.
  # A genuine multi-threaded capture workload (4 threads + graph + pinned + hostReg).
  # If not found, the MT roundtrip test is skipped at runtime via HRR_RAW_TRACE_EXE="".
  if(NOT DEFINED HRR_RAW_TRACE_EXE)
    find_program(HRR_RAW_TRACE_EXE NAMES hip_raw_trace hip_raw_trace.exe
      HINTS
        "${CMAKE_INSTALL_PREFIX}/bin"
        "${ROCM_PATH}/bin"
    )
    if(NOT HRR_RAW_TRACE_EXE)
      set(HRR_RAW_TRACE_EXE "")
    endif()
  endif()
  message(STATUS "HRR raw trace: ${HRR_RAW_TRACE_EXE}")

  target_compile_definitions(HrrTest PRIVATE
    HRR_TEST_EXE="$<TARGET_FILE:HrrTest>"
    HRR_PLAYBACK_EXE="${HRR_PLAYBACK_EXE}"
    HRR_RAW_TRACE_EXE="${HRR_RAW_TRACE_EXE}"
    ROCM_BIN_PATH="${ROCM_PATH}/bin"
  )
endif()
