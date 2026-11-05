# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(TEST_SRC
    test_hip_peer_copy_contract.cc
    test_rocp2p_direction_contract.cc
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../../clr/rocclr/device/rocm)

hip_add_exe_to_target(NAME ContractPeerCopyTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests)
