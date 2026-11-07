# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
  hipLocalityDomainAttributes.cc
  hipLocalityDomainGreenCtx.cc
  hipLocalityDomainMemPool.cc
)

hip_add_exe_to_target(
  NAME LocalityDomain
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests
)

set_target_properties(LocalityDomain PROPERTIES LINKER_LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(LocalityDomain ${GPGPU_LINKER_LIBRARIES})
