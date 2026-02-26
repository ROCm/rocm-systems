// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "util/detect_virtualization.hpp"

// ---------------------------------------------------------------------------
// is_sriov_virtual_function
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, SriovVfReturnsBool) {
    bool result = false;
    EXPECT_NO_THROW(result = is_sriov_virtual_function());
    (void)result;
}

// ---------------------------------------------------------------------------
// is_running_in_vm
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, VmDetectionReturnsBool) {
    bool result = false;
    EXPECT_NO_THROW(result = is_running_in_vm());
    (void)result;
}

// ---------------------------------------------------------------------------
// is_running_in_container
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, ContainerDetectionReturnsBool) {
    bool result = false;
    EXPECT_NO_THROW(result = is_running_in_container());
    (void)result;
}

// ---------------------------------------------------------------------------
// is_virtualization_enabled - consistency with sub-checks
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, MasterCheckConsistentWithSubChecks) {
    // Call each sub-function once, then verify the master result agrees.
    bool sriov     = is_sriov_virtual_function();
    bool vm        = is_running_in_vm();

    EXPECT_EQ(is_virtualization_enabled(), sriov || vm);
}


// ---------------------------------------------------------------------------
// Expect false on bare-metal / non-VF environment
// ---------------------------------------------------------------------------
TEST(DetectVirtualization, SriovVfReturnsFalseOnBareMetal) {
    EXPECT_FALSE(is_sriov_virtual_function());
}

TEST(DetectVirtualization, VmDetectionReturnsFalseOnBareMetal) {
    EXPECT_FALSE(is_running_in_vm());
}

TEST(DetectVirtualization, ContainerDetectionReturnsFalseOnBareMetal) {
    EXPECT_FALSE(is_running_in_container());
}

TEST(DetectVirtualization, VirtualizationEnabledReturnsFalseOnBareMetal) {
    EXPECT_FALSE(is_virtualization_enabled());
}