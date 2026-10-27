# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    __syncthreads.cc
    __syncthreads_count.cc
    __syncthreads_and.cc
    __syncthreads_or.cc
)

# the last argument linker libraries is required for this test but optional to the function
if(HIP_PLATFORM MATCHES "nvidia")
hip_add_exe_to_target(NAME SyncthreadsTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
hip_add_exe_to_target(NAME SyncthreadsTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc::hiprtc)
endif()
