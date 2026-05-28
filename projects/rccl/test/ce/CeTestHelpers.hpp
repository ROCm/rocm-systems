/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared CE test utilities – include in CeMPITests.cpp and CeInternalMPITests.cpp.

#ifndef CE_TEST_HELPERS_HPP
#define CE_TEST_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"

// Driver-version check
// Returns true when the HIP driver supports CE: ROCm ≥ 7.12 or the 7.0.2.x
// backport range [7.0.2.2, 7.0.3.0).
inline bool isCeDriverSupported()
{
    int driverVer = 0;
    if(hipDriverGetVersion(&driverVer) != hipSuccess)
        return false;
    return (driverVer >= 71200000) ||
           (driverVer >= 70051831 && driverVer < 70060000);
}

// Full CE dispatch gate

// Returns true when all four prerequisites for CE dispatch are satisfied:
//   1. Driver version supports CE (ROCm ≥ 7.12 or 7.0.2.2).
//   2. NCCL_CTA_POLICY=2     (CE dispatch mode).
//   3. NCCL_LOCAL_REGISTER=0  (explicit buffer registration only).
//   4. NCCL_CUMEM_ENABLE=1    (VMM-backed symmetric memory; default is 0).
//      Without this, comm->symmetricSupport is false and CE is never dispatched.
//
// Use this single check wherever a test needs to decide between asserting the
// CE path or the SM fallback path.
inline bool isCeDispatchConfigured()
{
    // Required values
    constexpr int kCtaPolicyCE     = 2; // NCCL_CTA_POLICY=2      → CE dispatch
    constexpr int kLocalRegisterCE = 0; // NCCL_LOCAL_REGISTER=0  → explicit registration
    constexpr int kCuMemEnable     = 1; // NCCL_CUMEM_ENABLE=1    → symmetric VMM memory
    // NCCL compiled-in defaults (returned when the env var is unset)
    constexpr int kCtaPolicyDefault     = 0; // SM path
    constexpr int kLocalRegisterDefault = 1; // implicit registration enabled
    constexpr int kCuMemEnableDefault   = 0; // VMM/symmetric memory disabled
    return isCeDriverSupported() &&
           MPIHelpers::getEnvParam("NCCL_CTA_POLICY",    kCtaPolicyDefault)     == kCtaPolicyCE &&
           MPIHelpers::getEnvParam("NCCL_LOCAL_REGISTER", kLocalRegisterDefault) == kLocalRegisterCE &&
           MPIHelpers::getEnvParam("NCCL_CUMEM_ENABLE",   kCuMemEnableDefault)   == kCuMemEnable;
}

// Batch path prediction helpers

// Mirrors the thresholds from ce_coll.cc (CE_COLL_INTRA_BATCH_SYNC_FREQ /
// CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD).  Kept here so the source file does
// not need to expose internal constants via its header.
static constexpr uint32_t kCeIntraBatchSyncFreq          = 8;
static constexpr uint64_t kCeIntraBatchSyncMsgThreshold  = 512ULL * 1024 * 1024;

// Returns true when the CE batch path will use intraBatchSync, given the
// actual numOps and chunkBytes for a collective.  Mirrors the condition in
// ncclCeLaunchBatchOps() so tests can assert the correct path sub-variant.
inline bool ceExpectIntraBatchSync(size_t numOps, size_t chunkBytes)
{
    return numOps > kCeIntraBatchSyncFreq &&
           chunkBytes * numOps >= kCeIntraBatchSyncMsgThreshold;
}

// Shared buffer helpers

// Fill buf[0..nElem) with float(rank + 1) on the GPU.
// Returns hipSuccess on success; caller should ASSERT_EQ on the result.
inline hipError_t ceFillRankScalarFloat(void* buf, size_t nElem, int rank)
{
    return RCCLTestHelpers::initializeBufferWithPattern<float>(
        buf, nElem, [rank](size_t) { return static_cast<float>(rank + 1); });
}

// Verify AllGather / AlltoAll output: element at index i must equal
// float(i / elemsPerRank + 1), i.e. each rank's block contains float(rank+1).
// Returns true when all elements match.
inline bool ceVerifyBlockPatternFloat(const void* buf, size_t totalElem, size_t elemsPerRank)
{
    return RCCLTestHelpers::verifyBufferData<float>(
        buf, totalElem,
        [elemsPerRank](size_t i) { return static_cast<float>(i / elemsPerRank + 1); });
}

#endif // MPI_TESTS_ENABLED

#endif // CE_TEST_HELPERS_HPP
