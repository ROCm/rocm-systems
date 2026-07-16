# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    test_hip_library_file_contract.cc
)

if(HIP_PLATFORM MATCHES "amd")
  set(RTCLIB "hiprtc::hiprtc")
else()
  set(RTCLIB "nvrtc")
endif()

hip_add_exe_to_target(NAME ContractLibraryFileTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME contract_tests
                      LINKER_LIBS ${RTCLIB})
