// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_UNIT_FIXTURES_H_
#define TESTS_AMD_SMI_TEST_UNIT_FIXTURES_H_

#include <gtest/gtest.h>

// Fixtures for the unit tier: no device, no amdsmi_init, no ROCm SMI internals.
// Kept out of api_test_framework.h so a unit test does not compile the device
// stack that header pulls in for the integration and functional tiers.
class GpuUnit : public ::testing::Test {};
class SystemUnit : public ::testing::Test {};

#endif  // TESTS_AMD_SMI_TEST_UNIT_FIXTURES_H_
