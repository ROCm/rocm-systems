# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(TEST_SRC
    test_hip_capture_to_graph_contract.cc
)

# cudaStreamBeginRecaptureToGraph needs CUDA 13.3; enable on NVIDIA once CI has it
if(HIP_PLATFORM MATCHES "amd")
  list(APPEND TEST_SRC test_hip_recapture_to_graph_contract.cc)
endif()

hip_add_exe_to_target(NAME ContractCaptureToGraphTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests)
