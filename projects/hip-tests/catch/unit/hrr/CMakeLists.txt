# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# HRR (HIP Record & Replay) tests
#
# Source files compiled into HrrTest:
#
#   hrr_format_test.cc   — CPU-only: archive format I/O unit tests (no GPU).
#   hrr_recovery_test.cc — CPU-only: crash-truncation recovery unit tests for
#                          the archive reader (no GPU).
#   hrr_workload_test.cc — Hidden ([.][hrr-direct]) GPU test cases invoked as
#                          subprocesses by hrr_roundtrip_test.cc; provides clean
#                          Windows process context for HIP + capture layer.
#   hrr_repro_workload_test.cc — Hidden ([.][hrr-direct]) regression micro-kernels
#                          (first-hipMalloc capture, embedded-pointer struct
#                          args, uncaptured-host-write divergence, null optional
#                          output pointer 0x20000 fault class, replay zero-init).
#   hrr_roundtrip_test.cc — Round-trip: sets HIP_HRR_CAPTURE_OUTPUT, spawns
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
    hrr_format_test.cc
    hrr_recovery_test.cc
    hrr_workload_test.cc
    hrr_repro_workload_test.cc
    hrr_roundtrip_test.cc
    ${HRR_CLR_DIR}/playback/hrr_reader.cpp
)

hip_add_exe_to_target(NAME HrrTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc::hiprtc
                      PROPERTY CXX_STANDARD 17)

# On Linux with ROCm clang/lld, the system C++ stdlib is not automatically
# linked. Add it explicitly so that std::filesystem, std::ostream etc. resolve.
# Use plain (non-keyword) signature to match hip_add_exe_to_target's usage.
if(NOT WIN32)
  target_link_libraries(HrrTest stdc++ stdc++fs)
endif()

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
# hrr_roundtrip_test.cc needs:
#   HRR_TEST_EXE     — bare filename; SpawnProc resolves it from currentPath()
#                      (= CTest WORKING_DIRECTORY = HrrTest's own directory).
#   HRR_PLAYBACK_EXE — bare filename; hrr-playback is copied next to HrrTest
#                      via POST_BUILD so it is always colocated.
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

  # Copy hrr-playback next to HrrTest so SpawnProc finds it via the bare name
  # "hrr-playback" (resolved from currentPath() = CTest WORKING_DIRECTORY).
  #
  # Two steps are needed:
  #   1. POST_BUILD copy into the build-tree output dir (for local ctest runs).
  #   2. install(PROGRAMS ...) into share/hip/catch_tests/ (for packaged runs).
  #      packaging/CMakeLists.txt installs all G_INSTALL_EXE_TARGETS there;
  #      we install hrr-playback to the same destination so it is colocated.
  add_custom_command(TARGET HrrTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${HRR_PLAYBACK_EXE}"
      "$<TARGET_FILE_DIR:HrrTest>/"
    COMMENT "Copying hrr-playback next to HrrTest"
  )
  include(GNUInstallDirs)
  install(PROGRAMS "${HRR_PLAYBACK_EXE}"
          DESTINATION "${CMAKE_INSTALL_DATADIR}/hip/${CATCH_BUILD_DIR}")

  # Optional: hip_raw_trace.exe for Unit_HRR_MultiThreadRoundtrip.
  # A genuine multi-threaded capture workload (4 threads + graph + pinned + hostReg).
  # If not found, the MT roundtrip test is skipped at runtime via HRR_RAW_TRACE_EXE="".
  if(NOT DEFINED HRR_RAW_TRACE_EXE)
    # Use a local variable so find_program's cache sentinel (-NOTFOUND) does
    # not leak into the compile definition.
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
  message(STATUS "HRR raw trace: ${HRR_RAW_TRACE_EXE}")

  target_compile_definitions(HrrTest PRIVATE
    # Use the exe filename only (no directory).  SpawnProc resolves relative
    # paths via currentPath() which equals the CTest WORKING_DIRECTORY —
    # the directory containing HrrTest.exe.  This works both from the build
    # tree and from an installed/packaged test run where the build tree is
    # no longer present.
    HRR_TEST_EXE="HrrTest"
    HRR_PLAYBACK_EXE="hrr-playback"
    HRR_RAW_TRACE_EXE="${HRR_RAW_TRACE_EXE}"
    ROCM_BIN_PATH="${ROCM_PATH}/bin"
  )
endif()
