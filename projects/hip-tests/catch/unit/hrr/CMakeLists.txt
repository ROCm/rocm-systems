# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Minimal HRR capture sentinel for hip-tests.
# Full HRR unit and integration tests live under projects/hrr/tests.

include("${CMAKE_CURRENT_SOURCE_DIR}/../../../../hrr/cmake/HrrCaptureLib.cmake")

set(TEST_SRC
    hrr_capture_smoke_test.cc
)

hip_add_exe_to_target(NAME HrrTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      PROPERTY CXX_STANDARD 17)

# hrr_capture_smoke_test.cc needs:
#   HRR_TEST_EXE  — bare filename; SpawnProc resolves it from currentPath()
#                   (= CTest WORKING_DIRECTORY = HrrTest's own directory).
#   ROCM_BIN_PATH — prepended to PATH in the child so it finds amdhip64_7.dll
#                   on Windows and the ROCm bin directory in packaged runs.
if(HIP_PLATFORM STREQUAL "amd")
  hrr_resolve_capture_lib(HRR_CAPTURE_LIB_DIR)
  if(HRR_CAPTURE_LIB_DIR)
    message(STATUS "HRR capture lib (smoke test): ${HRR_CAPTURE_LIB_DIR}")
    hrr_apply_capture_lib_rpath("${HRR_CAPTURE_LIB_DIR}" HrrTest)
  else()
    message(WARNING
      "HRR capture smoke test: HRR_CLR_LIB not set — subprocess may load a stale "
      "SDK libamdhip64. Build amdhip64 and pass -DHRR_CLR_LIB=<clr-build>/hipamd/lib")
  endif()

  target_compile_definitions(HrrTest PRIVATE
    HRR_TEST_EXE="HrrTest"
    ROCM_BIN_PATH="${ROCM_PATH}/bin"
    HRR_CLR_LIB="${HRR_CAPTURE_LIB_DIR}"
  )
endif()
