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

// The DMA-BUF export gate is a conjunction: the CMake symbol probe alone must not
// turn it on, because hipMemGetHandleForAddressRange is declared in headers on
// builds outside the supported version window too. All four input combinations,
// using a version inside the window and one below it.
static_assert(NCCL_CUMEM_DMABUF_EXPORT_GATE_FOR(1, ROCM_VER_7_12_60540), "probe + supported version must gate on");
static_assert(!NCCL_CUMEM_DMABUF_EXPORT_GATE_FOR(1, ROCM_VER_7_12_0),
              "probe must not override an unsupported version");
static_assert(!NCCL_CUMEM_DMABUF_EXPORT_GATE_FOR(0, ROCM_VER_7_12_60540),
              "a supported version must not override a failed probe");
static_assert(!NCCL_CUMEM_DMABUF_EXPORT_GATE_FOR(0, ROCM_VER_7_12_0), "neither input holds");

// The gate this build compiled with must still be the parameterized form instantiated
// with this build's own inputs, or the combinations asserted above stop describing it.
static_assert(NCCL_CUMEM_DMABUF_EXPORT_GATE ==
                  NCCL_CUMEM_DMABUF_EXPORT_GATE_FOR(NCCL_CUMEM_DMABUF_EXPORT_PROBE, HIP_VERSION),
              "the build's gate is no longer the parameterized gate for (probe, HIP_VERSION)");

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

// Kernel floor for cuMem auto-detect. Encoding and the override rule must not
// silently drift: NCCL_CUMEM_ENABLE>0 bypasses <6.8; auto (-2) and explicit 0 do not.
static_assert(NCCL_CUMEM_MIN_KERNEL_VERSION == NCCL_KERNEL_VERSION_CODE(6, 8), "cuMem kernel floor drifted");
static_assert(NCCL_CUMEM_KERNEL_GATE_FOR(NCCL_KERNEL_VERSION_CODE(6, 8), -2), "auto on 6.8 must pass");
static_assert(NCCL_CUMEM_KERNEL_GATE_FOR(NCCL_KERNEL_VERSION_CODE(6, 8), 0), "explicit-off on 6.8 must pass kernel gate");
static_assert(!NCCL_CUMEM_KERNEL_GATE_FOR(NCCL_KERNEL_VERSION_CODE(6, 7), -2), "auto on 6.7 must fail");
static_assert(!NCCL_CUMEM_KERNEL_GATE_FOR(NCCL_KERNEL_VERSION_CODE(5, 14), -2), "auto on 5.14 must fail");
static_assert(!NCCL_CUMEM_KERNEL_GATE_FOR(NCCL_KERNEL_VERSION_CODE(5, 14), 0), "explicit-off on 5.14 must fail");
static_assert(NCCL_CUMEM_KERNEL_GATE_FOR(NCCL_KERNEL_VERSION_CODE(5, 14), 1), "force-enable on 5.14 must pass");
static_assert(NCCL_CUMEM_KERNEL_GATE_FOR(NCCL_KERNEL_VERSION_CODE(6, 7), 1), "force-enable on 6.7 must pass");

TEST(VersionGateTests, CuMemKernelGate)
{
    const int k68 = NCCL_KERNEL_VERSION_CODE(6, 8);
    const int k67 = NCCL_KERNEL_VERSION_CODE(6, 7);
    const int k514 = NCCL_KERNEL_VERSION_CODE(5, 14);

    // Kernel >= 6.8: gate passes for auto, explicit off, and force-enable.
    EXPECT_TRUE(NCCL_CUMEM_KERNEL_GATE_FOR(k68, -2));
    EXPECT_TRUE(NCCL_CUMEM_KERNEL_GATE_FOR(k68, 0));
    EXPECT_TRUE(NCCL_CUMEM_KERNEL_GATE_FOR(k68, 1));
    EXPECT_TRUE(NCCL_CUMEM_KERNEL_GATE_FOR(NCCL_KERNEL_VERSION_CODE(6, 9), -2));

    // Kernel < 6.8: only an explicit NCCL_CUMEM_ENABLE>0 override passes.
    EXPECT_FALSE(NCCL_CUMEM_KERNEL_GATE_FOR(k67, -2));
    EXPECT_FALSE(NCCL_CUMEM_KERNEL_GATE_FOR(k514, -2));
    EXPECT_FALSE(NCCL_CUMEM_KERNEL_GATE_FOR(k514, 0));
    EXPECT_TRUE(NCCL_CUMEM_KERNEL_GATE_FOR(k514, 1));
    EXPECT_TRUE(NCCL_CUMEM_KERNEL_GATE_FOR(k67, 2));
}

}  // namespace RcclUnitTesting
