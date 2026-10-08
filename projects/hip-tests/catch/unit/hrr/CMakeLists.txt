# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Minimal HRR capture sentinel for hip-tests.
# Full HRR unit and integration tests live under projects/hrr/tests.

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
  target_compile_definitions(HrrTest PRIVATE
    HRR_TEST_EXE="HrrTest"
    ROCM_BIN_PATH="${ROCM_PATH}/bin"
  )
endif()
