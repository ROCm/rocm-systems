/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Smoke tests for the env vars added in NCCL 2.29.2 / 2.29.7:
//
//   * NCCL_SOCKET_POLL_TIMEOUT_MSEC (2.29.2)   - waits instead of spinning
//                                                during bootstrap to reduce
//                                                CPU usage
//   * NCCL_NO_CACHE                 (2.29.7)   - forces NCCL to re-read
//                                                selected env vars rather
//                                                than caching their value
//   * NCCL_NETDEVS_POLICY                       - now also obeyed by
//                                                all2all/send/recv since
//                                                2.29.2
//
// Each test runs in an isolated process so the env-var value is parsed
// fresh on every run. The assertion shape is intentionally minimal: a
// healthy single-rank comm must initialize, run a collective, and tear
// down cleanly under each setting. Deeper behavioral assertions (CPU
// utilization, re-read effects, per-device routing) are out of scope for
// a unit-level smoke test and belong in higher-level integration tests.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{

static bool hasGpuAvailable() {
    int numDevices = 0;
    hipError_t err = hipGetDeviceCount(&numDevices);
    return (err == hipSuccess && numDevices >= 1);
}

#define SKIP_IF_NO_GPU()                                                  \
    do {                                                                  \
        if (!hasGpuAvailable()) {                                         \
            GTEST_SKIP() << "This test requires at least 1 GPU device."; \
            return;                                                       \
        }                                                                 \
    } while (0)

static ncclResult_t initSingleRankComm(ncclComm_t* comm) {
    ncclUniqueId id;
    ncclResult_t res = ncclGetUniqueId(&id);
    if (res != ncclSuccess) return res;
    return ncclCommInitRank(comm, 1, id, 0);
}

static void runTrivialAllReduce(ncclComm_t comm) {
    const size_t numElements = 1024;
    float* devBuf = nullptr;
    HIPCALL(hipMalloc(&devBuf, numElements * sizeof(float)));
    HIPCALL(hipMemset(devBuf, 0, numElements * sizeof(float)));

    hipStream_t stream;
    HIPCALL(hipStreamCreate(&stream));
    NCCLCHECK(ncclAllReduce(devBuf, devBuf, numElements, ncclFloat32,
                            ncclSum, comm, stream));
    HIPCALL(hipStreamSynchronize(stream));
    HIPCALL(hipStreamDestroy(stream));
    HIPCALL(hipFree(devBuf));
}

static void runHealthyLifecycle() {
    SKIP_IF_NO_GPU();
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm = nullptr;
    NCCLCHECK(initSingleRankComm(&comm));
    runTrivialAllReduce(comm);
    NCCLCHECK(ncclCommDestroy(comm));
}

TEST(EnvVarSmoke, ProcessIsolatedSuite)
{
    using TC = ProcessIsolatedTestRunner::TestConfig;
    RUN_ISOLATED_TESTS(
        // -------------------------------------------------------------------
        // NCCL_SOCKET_POLL_TIMEOUT_MSEC: 0 (legacy spin), 1 (wait 1ms),
        // 100 (wait 100ms). All three are valid and must not block init.
        // -------------------------------------------------------------------
        TC("SocketPollTimeout_Zero",      runHealthyLifecycle)
            .withEnvironment({{"NCCL_SOCKET_POLL_TIMEOUT_MSEC", "0"}}),
        TC("SocketPollTimeout_One",       runHealthyLifecycle)
            .withEnvironment({{"NCCL_SOCKET_POLL_TIMEOUT_MSEC", "1"}}),
        TC("SocketPollTimeout_Hundred",   runHealthyLifecycle)
            .withEnvironment({{"NCCL_SOCKET_POLL_TIMEOUT_MSEC", "100"}}),

        // -------------------------------------------------------------------
        // NCCL_NO_CACHE: 0 (default cached behavior), 1 (force re-read)
        // -------------------------------------------------------------------
        TC("NoCache_Disabled",            runHealthyLifecycle)
            .withEnvironment({{"NCCL_NO_CACHE", "0"}}),
        TC("NoCache_Enabled",             runHealthyLifecycle)
            .withEnvironment({{"NCCL_NO_CACHE", "1"}}),

        // -------------------------------------------------------------------
        // NCCL_NETDEVS_POLICY: documented policy names.
        // We exercise each documented value to catch parse regressions; the
        // implementation logs a warning and falls back to AUTO for unknown
        // values, so BOGUS must not abort init.
        // -------------------------------------------------------------------
        TC("NetdevsPolicy_AUTO",          runHealthyLifecycle)
            .withEnvironment({{"NCCL_NETDEVS_POLICY", "AUTO"}}),
        TC("NetdevsPolicy_ALL",           runHealthyLifecycle)
            .withEnvironment({{"NCCL_NETDEVS_POLICY", "ALL"}}),
        TC("NetdevsPolicy_GPU",           runHealthyLifecycle)
            .withEnvironment({{"NCCL_NETDEVS_POLICY", "GPU"}}),
        TC("NetdevsPolicy_Bogus_Fallback", runHealthyLifecycle)
            .withEnvironment({{"NCCL_NETDEVS_POLICY", "DEFINITELY_NOT_A_POLICY"}})
    );
}

} // namespace RcclUnitTesting
