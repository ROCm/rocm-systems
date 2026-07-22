/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the centralized ROCm/HIP version-gating macros in rocmwrap.h.
// These macros are constant expressions evaluated against both HIP_VERSION
// (compile time) and the runtime driver version, so the support windows are
// asserted here against representative version codes to prevent silent drift.

#include "rocmwrap.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

// Milestone constants must not drift (call sites and the predicates below
// depend on these exact encodings: MAJOR*10000000 + MINOR*100000 + PATCH).
static_assert(ROCM_VER_7_0_2_2     == 70051831, "7.0.2.x backport lower bound changed");
static_assert(ROCM_VER_7_0_3_0     == 70060000, "7.0.2.x backport upper bound changed");
static_assert(ROCM_VER_7_12_0      == 71200000, "7.12 native min changed");
static_assert(ROCM_VER_7_12_60540  == 71260540, "cuMem/DMA-BUF native min changed");
static_assert(NCCL_CUMEM_NATIVE_MIN_VERSION == ROCM_VER_7_12_60540, "native-min alias drifted");

// Generic range predicates.
static_assert(NCCL_VER_GE(ROCM_VER_7_12_60540, ROCM_VER_7_12_60540), "GE boundary");
static_assert(!NCCL_VER_GE(ROCM_VER_7_12_60540 - 1, ROCM_VER_7_12_60540), "GE below boundary");
static_assert(NCCL_VER_IN(ROCM_VER_7_0_2_2, ROCM_VER_7_0_2_2, ROCM_VER_7_0_3_0), "IN lower-inclusive");
static_assert(!NCCL_VER_IN(ROCM_VER_7_0_3_0, ROCM_VER_7_0_2_2, ROCM_VER_7_0_3_0), "IN upper-exclusive");

TEST(VersionGateTests, CuMemVersionSupported)
{
    // Native 7.12+ window (open-ended).
    EXPECT_TRUE (NCCL_CUMEM_VERSION_SUPPORTED(ROCM_VER_7_12_60540));
    EXPECT_TRUE (NCCL_CUMEM_VERSION_SUPPORTED(ROCM_VER_7_12_60540 + 1));
    EXPECT_FALSE(NCCL_CUMEM_VERSION_SUPPORTED(ROCM_VER_7_12_60540 - 1));

    // 7.0.2.x backport window [70051831, 70060000).
    EXPECT_TRUE (NCCL_CUMEM_VERSION_SUPPORTED(ROCM_VER_7_0_2_2));
    EXPECT_TRUE (NCCL_CUMEM_VERSION_SUPPORTED(ROCM_VER_7_0_3_0 - 1));
    EXPECT_FALSE(NCCL_CUMEM_VERSION_SUPPORTED(ROCM_VER_7_0_2_2 - 1));
    EXPECT_FALSE(NCCL_CUMEM_VERSION_SUPPORTED(ROCM_VER_7_0_3_0));

    // The gap between the backport upper bound and the native min is unsupported.
    EXPECT_FALSE(NCCL_CUMEM_VERSION_SUPPORTED(ROCM_VER_7_12_0));
}

TEST(VersionGateTests, CuMemHostIsNativeOnly)
{
    EXPECT_TRUE (NCCL_CUMEM_HOST_VERSION_SUPPORTED(ROCM_VER_7_12_60540));
    EXPECT_FALSE(NCCL_CUMEM_HOST_VERSION_SUPPORTED(ROCM_VER_7_12_60540 - 1));

    // Host allocations are deliberately NOT part of the 7.0.2.x backport
    // (they rely on hipDeviceAttributeHostNumaId, absent there).
    EXPECT_FALSE(NCCL_CUMEM_HOST_VERSION_SUPPORTED(ROCM_VER_7_0_2_2));
    EXPECT_FALSE(NCCL_CUMEM_HOST_VERSION_SUPPORTED(ROCM_VER_7_0_3_0 - 1));
}

TEST(VersionGateTests, CeBatchAsyncWindow)
{
    // Native 7.12+ (note: 7.12.0, lower than the cuMem milestone).
    EXPECT_TRUE (NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(ROCM_VER_7_12_0));
    EXPECT_TRUE (NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(ROCM_VER_7_12_60540));
    EXPECT_FALSE(NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(ROCM_VER_7_12_0 - 1));

    // 7.0.2.x backport window also enables the batch API.
    EXPECT_TRUE (NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(ROCM_VER_7_0_2_2));
    EXPECT_TRUE (NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(ROCM_VER_7_0_3_0 - 1));
    EXPECT_FALSE(NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(ROCM_VER_7_0_2_2 - 1));
    EXPECT_FALSE(NCCL_CE_BATCH_ASYNC_VERSION_SUPPORTED(ROCM_VER_7_0_3_0));
}

}  // namespace RcclUnitTesting
