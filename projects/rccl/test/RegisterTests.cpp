/*************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <atomic>
#include <new>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{

// Helper to check GPU availability
static bool hasGpuAvailable() {
    int numDevices = 0;
    hipError_t err = hipGetDeviceCount(&numDevices);
    return (err == hipSuccess && numDevices >= 1);
}

// The GH#1859 user-buffer IPC registration path that this test exercises only
// behaves as intended on the CDNA datacenter GPUs (gfx942 / gfx950 families).
// On other architectures (e.g. the RDNA gfx11xx consumer parts) the IPC /
// collective registration machinery differs and the test cannot meaningfully
// reproduce the defect, so we skip rather than report a misleading pass/fail.
static bool isArchSupportedForP2pCollTest(const char* gcnArchName) {
    // gcnArchName looks like "gfx942:sramecc+:xnack-"; compare the gfxNNN prefix.
    static const char* kSupportedArchs[] = {"gfx942", "gfx950"};
    for (const char* arch : kSupportedArchs) {
        if (strncmp(gcnArchName, arch, strlen(arch)) == 0) {
            return true;
        }
    }
    return false;
}

// Queries every visible device and returns true only if all of them are a
// supported architecture for the P2P+collective registration test.
static bool allDevicesSupportP2pCollTest(int numDevices, char* unsupportedArchOut,
                                         size_t outLen) {
    for (int dev = 0; dev < numDevices; dev++) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
            return false;
        }
        if (!isArchSupportedForP2pCollTest(prop.gcnArchName)) {
            if (unsupportedArchOut && outLen > 0) {
                strncpy(unsupportedArchOut, prop.gcnArchName, outLen - 1);
                unsupportedArchOut[outLen - 1] = '\0';
            }
            return false;
        }
    }
    return true;
}

// Macro to skip test if no GPU is available
#define SKIP_IF_NO_GPU() \
    do { \
        if (!hasGpuAvailable()) { \
            GTEST_SKIP() << "This test requires at least 1 GPU device."; \
            return; \
        } \
    } while(0)

// Helper to initialize a single-rank communicator
static ncclResult_t initSingleRankComm(ncclComm_t* comm) {
    ncclUniqueId id;
    ncclResult_t res = ncclGetUniqueId(&id);
    if (res != ncclSuccess) return res;
    return ncclCommInitRank(comm, 1, id, 0);
}

//==============================================================================
// Test implementation functions - parameterized by registration expectation
//==============================================================================

/**
 * @brief Test basic register/deregister of a single buffer
 * @param expectNonNull If true, expect non-NULL handle (registration enabled)
 */
static void testCommRegisterDeregister(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Create buffer on device
    const size_t bufferSize = 1024 * 1024; // 1 MB
    void* deviceBuffer = nullptr;
    HIPCALL(hipMalloc(&deviceBuffer, bufferSize));
    ASSERT_NE(deviceBuffer, nullptr) << "Failed to allocate device buffer";

    // Register buffer with ncclCommRegister
    void* regHandle = nullptr;
    NCCLCHECK(ncclCommRegister(comm, deviceBuffer, bufferSize, &regHandle));

    // Verify handle based on expected behavior
    if (expectNonNull) {
        EXPECT_NE(regHandle, nullptr)
            << "Buffer registration failed: regHandle is NULL even though NCCL_LOCAL_REGISTER=1";
    } else {
        EXPECT_EQ(regHandle, nullptr)
            << "Expected NULL handle when NCCL_LOCAL_REGISTER is disabled";
    }

    // Deregister and clean up
    NCCLCHECK(ncclCommDeregister(comm, regHandle));
    HIPCALL(hipFree(deviceBuffer));
    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test registering multiple buffers simultaneously
 * @param expectNonNull If true, expect non-NULL handles and verify uniqueness
 */
static void testMultipleBufferRegistration(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Create and register multiple buffers
    const int numBuffers = 4;
    const size_t bufferSize = 64 * 1024; // 64 KB each
    void* deviceBuffers[numBuffers] = {nullptr};
    void* regHandles[numBuffers] = {nullptr};

    for (int i = 0; i < numBuffers; i++) {
        HIPCALL(hipMalloc(&deviceBuffers[i], bufferSize));
        ASSERT_NE(deviceBuffers[i], nullptr) << "Failed to allocate buffer " << i;

        NCCLCHECK(ncclCommRegister(comm, deviceBuffers[i], bufferSize, &regHandles[i]));

        if (expectNonNull) {
            EXPECT_NE(regHandles[i], nullptr) << "Registration failed for buffer " << i;
        } else {
            EXPECT_EQ(regHandles[i], nullptr) << "Expected NULL handle for buffer " << i;
        }
    }

    // Verify all handles are unique (only when registration is enabled)
    if (expectNonNull) {
        for (int i = 0; i < numBuffers; i++) {
            for (int j = i + 1; j < numBuffers; j++) {
                if (regHandles[i] != nullptr && regHandles[j] != nullptr) {
                    EXPECT_NE(regHandles[i], regHandles[j])
                        << "Buffers " << i << " and " << j << " have the same registration handle";
                }
            }
        }
    }

    // Deregister and clean up
    for (int i = 0; i < numBuffers; i++) {
        NCCLCHECK(ncclCommDeregister(comm, regHandles[i]));
        HIPCALL(hipFree(deviceBuffers[i]));
    }
    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test registering buffers of various sizes
 * @param expectNonNull If true, expect non-NULL handles for all sizes
 */
static void testVariableSizeBuffers(bool expectNonNull) {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Test various buffer sizes: 4KB, 64KB, 1MB, 4MB
    const size_t sizes[] = {4096, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < numSizes; i++) {
        void* deviceBuffer = nullptr;
        void* regHandle = nullptr;

        HIPCALL(hipMalloc(&deviceBuffer, sizes[i]));
        ASSERT_NE(deviceBuffer, nullptr) << "Failed to allocate buffer of size " << sizes[i];

        NCCLCHECK(ncclCommRegister(comm, deviceBuffer, sizes[i], &regHandle));

        if (expectNonNull) {
            EXPECT_NE(regHandle, nullptr)
                << "Registration failed for buffer size " << sizes[i] << " bytes";
        } else {
            EXPECT_EQ(regHandle, nullptr)
                << "Expected NULL handle for buffer size " << sizes[i] << " bytes";
        }

        NCCLCHECK(ncclCommDeregister(comm, regHandle));
        HIPCALL(hipFree(deviceBuffer));
    }

    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Regression test for NCCL GH#1859 / v2.29.2-1 fix:
 *        "Fixes crash that can happen when calling p2p and then collectives
 *         while using the same user buffer."
 *
 * Root cause: in p2p.cc the NCCL_IPC_COLLECTIVE branch allocates
 * devPeerRmtAddrs but only copies hostPeerRmtAddrs → devPeerRmtAddrs when
 * needUpdate==true.  If P2P ran first on the same regRecord, needUpdate is
 * already false by the time the collective resolves addresses, so
 * devPeerRmtAddrs is left zeroed and the GPU kernel dereferences a null
 * remote pointer → illegal memory access / crash.
 *
 * Sequence that triggers the bug (requires ≥ 2 intra-node GPUs), run with
 * one forked process per rank (see below):
 *   1. ncclCommInitRank in each rank's own process (one GPU per rank), using
 *      a shared ncclUniqueId.
 *   2. ncclCommRegister the same device buffer on each rank
 *      (NCCL_LOCAL_REGISTER=1 ensures real IPC registration).
 *   3. ncclSend / ncclRecv group call on the registered buffer → sync.
 *      (populates hostPeerRmtAddrs; leaves devPeerRmtAddrs NULL for the
 *       collective branch because only the P2P branch ran.)
 *   4. ncclAllReduce group call on the same registered buffer → sync.
 *      (buggy: allocates devPeerRmtAddrs, skips memcpy, crashes;
 *       fixed:  allocates devPeerRmtAddrs and always copies on first alloc.)
 *   5. Verify ncclSuccess and correct AllReduce output.
 *
 * Skipped when fewer than 2 GPUs are visible (intra-node IPC path requires
 * at least 2 ranks on the same node) or when the two GPUs cannot do direct
 * GPU-to-GPU P2P.  The GH#1859 fix lives in ipcRegisterBuffer, which is only
 * reached when the collective takes the direct-P2P/IPC transport.  On systems
 * where the GPUs are not P2P-capable (e.g. PCIE-only consumer cards) RCCL
 * falls back to the SHM transport, ipcRegisterBuffer is never called, and the
 * test would pass without exercising the fixed code path at all.
 *
 * --- Why fork() + ncclCommInitRank instead of ncclCommInitAll ---
 *
 * comm->directMode is set true by ncclTransportCheckP2pType whenever any two
 * local ranks share the same pid.  ipcRegisterBuffer's legacy IPC path then
 * hits:
 *     if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
 * and bypasses the fixed code path entirely.  ncclCommInitAll runs all ranks
 * in one process, so directMode is always true and the GH#1859 fix is never
 * exercised.  We therefore fork() one child process per rank and use
 * ncclCommInitRank with a shared ncclUniqueId; each rank is its own process,
 * directMode stays false, and ipcRegisterBuffer is reachable.
 *
 * The ProcessIsolatedTestRunner re-execs each test into its own process but
 * cannot split a single test across one-process-per-rank, so the fork() is
 * done here inside the test body.
 */

// Shared-memory bootstrap written by rank-0's child before the others proceed.
namespace {
// Bootstrap state published by rank 0's child to the other ranks.
enum class P2pCollState : int {
    NotReady = 0,  // rank 0 has not yet published the unique ID
    Ready    = 1,  // unique ID is valid and ready to be read
    Skip     = -1, // rank 0 hit a skip/fatal precondition; others should skip
};

struct P2pCollShared {
    ncclUniqueId id;
    std::atomic<P2pCollState> state;
};

// Child-process error handling: print and return a non-zero code rather than
// using GTest assertions (which do not propagate across fork()).
#define CHILD_HC(x)                                                          \
    do {                                                                     \
        hipError_t _e = (x);                                                 \
        if (_e != hipSuccess) {                                              \
            printf("[rank %d] HIP error %d (%s) @ %s:%d\n", rank, _e,        \
                   hipGetErrorString(_e), __FILE__, __LINE__);               \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define CHILD_NC(x)                                                          \
    do {                                                                     \
        ncclResult_t _e = (x);                                               \
        if (_e != ncclSuccess) {                                             \
            printf("[rank %d] NCCL error %d (%s) @ %s:%d\n", rank, _e,       \
                   ncclGetErrorString(_e), __FILE__, __LINE__);              \
            return 1;                                                        \
        }                                                                    \
    } while (0)

// Child exit codes communicated back to the parent test body.
enum P2pCollChildExit {
    CHILD_OK   = 0,  // ran the full sequence, result correct
    CHILD_FAIL = 1,  // a HIP/NCCL error or a wrong AllReduce result
    CHILD_SKIP = 2,  // preconditions not met (too few GPUs / no direct P2P)
};

// Runs entirely inside a forked child process — first HIP/NCCL call is here.
static int p2pCollRunRank(int rank, int nranks, P2pCollShared* shared)
{
    // Must exceed nChannels * NCCL_P2P_LL_THRESHOLD (default 8192 bytes) so
    // that addP2pToPlan selects NCCL_PROTO_SIMPLE. Only with SIMPLE does it
    // enter the ncclRegisterP2pIpcBuffer branch that leads to ipcRegisterBuffer
    // (lines 1188-1222, where the GH#1859 fix lives). 16k floats = 65536 bytes
    // is above the threshold for any plausible channel count.
    const size_t numElements = 16 * 1024;                  // 64 KB of float
    const size_t bufBytes    = numElements * sizeof(float);

    if (rank == 0) {
        int numDevices = 0;
        CHILD_HC(hipGetDeviceCount(&numDevices));
        if (numDevices < nranks) {
            printf("Requires %d GPUs (detected %d).\n", nranks, numDevices);
            shared->state.store(P2pCollState::Skip, std::memory_order_release);
            return CHILD_SKIP;
        }

        // The GH#1859 defect only reproduces on the CDNA datacenter GPUs
        // (gfx942 / gfx950).  On other architectures (e.g. RDNA gfx11xx) the
        // registration path differs and the test is meaningless, so skip.
        char unsupportedArch[256] = {0};
        if (!allDevicesSupportP2pCollTest(nranks, unsupportedArch, sizeof(unsupportedArch))) {
            printf("Unsupported GPU architecture '%s' for the P2P+collective "
                   "user-buffer registration test (requires gfx942 or gfx950).\n",
                   unsupportedArch);
            shared->state.store(P2pCollState::Skip, std::memory_order_release);
            return CHILD_SKIP;
        }

        // The IPC registration path is only reached with direct GPU-to-GPU P2P;
        // without it RCCL falls back to SHM and ipcRegisterBuffer is never
        // exercised, so skip rather than report a misleading pass.
        int canAccess01 = 0, canAccess10 = 0;
        CHILD_HC(hipDeviceCanAccessPeer(&canAccess01, 0, 1));
        CHILD_HC(hipDeviceCanAccessPeer(&canAccess10, 1, 0));
        if (!canAccess01 || !canAccess10) {
            printf("No direct P2P (canAccessPeer 0->1=%d 1->0=%d): SHM transport "
                   "would be used, ipcRegisterBuffer NOT exercised.\n",
                   canAccess01, canAccess10);
            shared->state.store(P2pCollState::Skip, std::memory_order_release);
            return CHILD_SKIP;
        }

        // Generate the unique ID and publish it to the other children.  The
        // release store ensures the id write above is visible before the flag.
        CHILD_NC(ncclGetUniqueId(&shared->id));
        shared->state.store(P2pCollState::Ready, std::memory_order_release);
    } else {
        // Wait for rank 0 to publish the ID (or signal skip/error).  The
        // acquire load pairs with rank 0's release store so that, once we
        // observe a non-NotReady state, the id write is visible to us.
        P2pCollState st;
        while ((st = shared->state.load(std::memory_order_acquire)) ==
               P2pCollState::NotReady) { /* spin */ }
        if (st == P2pCollState::Skip) return CHILD_SKIP;  // rank 0 reported skip/error
    }

    ncclUniqueId id = shared->id;

    CHILD_HC(hipSetDevice(rank));

    ncclComm_t comm;
    CHILD_NC(ncclCommInitRank(&comm, nranks, id, rank));

    hipStream_t stream;
    CHILD_HC(hipStreamCreate(&stream));

    // Input buffer filled with rank+1; separate out-of-place output buffer.
    void *devBuf = nullptr, *devOut = nullptr;
    CHILD_HC(hipMalloc(&devBuf, bufBytes));
    CHILD_HC(hipMalloc(&devOut, bufBytes));
    std::vector<float> hostIn(numElements, static_cast<float>(rank + 1));
    CHILD_HC(hipMemcpy(devBuf, hostIn.data(), bufBytes, hipMemcpyHostToDevice));
    CHILD_HC(hipMemset(devOut, 0, bufBytes));

    // Register the input buffer — used for both the P2P and the AllReduce.
    void* regHandle = nullptr;
    CHILD_NC(ncclCommRegister(comm, devBuf, bufBytes, &regHandle));
    if (regHandle == nullptr) {
        printf("[rank %d] ncclCommRegister returned NULL handle "
               "(is NCCL_LOCAL_REGISTER=1 set?).\n", rank);
        return CHILD_FAIL;
    }

    // Also register the AllReduce *output* buffer.  ncclRegisterCollBuffers
    // calls ncclRegFind on the recvbuff for RING/TREE and bails out via
    // `goto exit` (skipping the NCCL_IPC_COLLECTIVE ipcRegisterBuffer path)
    // if it is not registered — which would mean the bug is never exercised.
    void* regHandleOut = nullptr;
    CHILD_NC(ncclCommRegister(comm, devOut, bufBytes, &regHandleOut));
    if (regHandleOut == nullptr) {
        printf("[rank %d] ncclCommRegister (output) returned NULL handle "
               "(is NCCL_LOCAL_REGISTER=1 set?).\n", rank);
        return CHILD_FAIL;
    }

    // --- Step 1: P2P send/recv on the registered buffer ---
    // Rank 0 sends devBuf -> rank 1; rank 1 receives into its devBuf.  This
    // populates regRecord->regIpcAddrs.hostPeerRmtAddrs for the P2P path but
    // leaves the NCCL_IPC_COLLECTIVE devPeerRmtAddrs uninitialised.
    CHILD_NC(ncclGroupStart());
    if (rank == 0) CHILD_NC(ncclSend(devBuf, numElements, ncclFloat, 1, comm, stream));
    if (rank == 1) CHILD_NC(ncclRecv(devBuf, numElements, ncclFloat, 0, comm, stream));
    CHILD_NC(ncclGroupEnd());
    CHILD_HC(hipStreamSynchronize(stream));

    // --- Step 2: AllReduce on the *same* registered buffer (bug trigger) ---
    // The collective registers its *recvbuff* for NCCL_IPC_COLLECTIVE.  By using
    // devBuf (already IPC-registered during the P2P) as the recvbuff, the
    // collective hits the *reuse* branch in ipcRegisterBuffer, which leaves
    // needUpdate=false and therefore skips the devPeerRmtAddrs memcpy.  Without
    // the fix this crashes with an illegal memory access because devPeerRmtAddrs
    // is allocated but never populated.  RING in-place AllReduce
    // (sendbuff==recvbuff) is skipped by the registration path, so this must be
    // out-of-place: seed devOut from devBuf and reduce devOut -> devBuf.
    CHILD_HC(hipMemcpy(devOut, devBuf, bufBytes, hipMemcpyDeviceToDevice));
    CHILD_NC(ncclGroupStart());
    CHILD_NC(ncclAllReduce(devOut, devBuf, numElements, ncclFloat, ncclSum, comm, stream));
    CHILD_NC(ncclGroupEnd());
    CHILD_HC(hipStreamSynchronize(stream));

    // --- Verify AllReduce result ---
    // After the P2P both ranks hold 1.0f in devBuf (rank 0 kept its own value,
    // rank 1 received 1.0f from rank 0), so the sum is numRanks per element.
    const float expectedSum = static_cast<float>(nranks);
    std::vector<float> hostOut(numElements, -1.0f);
    CHILD_HC(hipMemcpy(hostOut.data(), devBuf, bufBytes, hipMemcpyDeviceToHost));
    int rc = CHILD_OK;
    for (size_t i = 0; i < numElements; i++) {
        if (hostOut[i] != expectedSum) {
            printf("[rank %d] MISMATCH elem %zu: expected %f got %f\n",
                   rank, i, expectedSum, hostOut[i]);
            rc = CHILD_FAIL;
            break;
        }
    }

    CHILD_NC(ncclCommDeregister(comm, regHandle));
    CHILD_NC(ncclCommDeregister(comm, regHandleOut));
    CHILD_HC(hipFree(devBuf));
    CHILD_HC(hipFree(devOut));
    CHILD_HC(hipStreamDestroy(stream));
    CHILD_NC(ncclCommDestroy(comm));

    return rc;
}
} // namespace

static void testP2pThenCollectiveSameBuffer()
{
    const int numRanks = 2;

    // Set up shared memory BEFORE fork and BEFORE any HIP/NCCL call.  The
    // HIP/HSA runtime is not fork-safe, so the parent does no HIP/NCCL work at
    // all — rank-0's child generates the ncclUniqueId here and signals the
    // others through this MAP_SHARED region.
    P2pCollShared* shared = static_cast<P2pCollShared*>(mmap(
        nullptr, sizeof(P2pCollShared),
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(shared, MAP_FAILED) << "mmap for shared bootstrap failed";
    // Construct the atomic in place in the shared region before forking so all
    // children operate on the same object.  MAP_ANONYMOUS zeroes the page, but
    // the atomic is initialised explicitly for clarity and correctness.
    new (&shared->state) std::atomic<P2pCollState>(P2pCollState::NotReady);

    std::vector<pid_t> pids(numRanks, -1);
    for (int r = 0; r < numRanks; r++) {
        pids[r] = fork();
        ASSERT_GE(pids[r], 0) << "fork() failed for rank " << r;
        if (pids[r] == 0) {
            // Child: first HIP/NCCL calls happen here, on a clean runtime.
            int rc = p2pCollRunRank(r, numRanks, shared);
            _exit(rc);
        }
    }

    // Parent: reap children and translate their exit codes into the GTest
    // result.  No HIP/NCCL calls here.
    bool anySkip = false;
    bool anyFail = false;
    for (int r = 0; r < numRanks; r++) {
        int status = 0;
        waitpid(pids[r], &status, 0);
        if (WIFEXITED(status)) {
            int code = WEXITSTATUS(status);
            if (code == CHILD_SKIP) {
                anySkip = true;
            } else if (code != CHILD_OK) {
                anyFail = true;
                ADD_FAILURE() << "Rank " << r << " child exited with failure code "
                              << code << ".";
            }
        } else {
            anyFail = true;
            ADD_FAILURE() << "Rank " << r << " child terminated abnormally "
                          << "(status " << status << ") — likely the GH#1859 "
                          << "crash (illegal memory access in the collective).";
        }
    }

    munmap(shared, sizeof(P2pCollShared));

    if (anySkip && !anyFail) {
        GTEST_SKIP() << "Test requires " << numRanks << " supported GPUs "
                        "(gfx942 / gfx950) with direct GPU-to-GPU P2P access; "
                        "preconditions not met.";
    }
}

/**
 * @brief Test deregistering NULL handle (should succeed as no-op)
 */
static void testDeregisterNullHandle() {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    // Deregister NULL handle - should be a no-op
    NCCLCHECK(ncclCommDeregister(comm, nullptr));

    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Exercises the non-symmetric window register/deregister/destroy path
 *        on a single-rank communicator.
 *
 * With cuMem disabled, symmetricSupport is false, so ncclCommWindowRegister
 * takes the non-symmetric fallback path. This initializes devrState (via
 * ncclDevrInitOnce) and then tears it down through ncclDevrFinalize when the
 * comm is destroyed. The test verifies that register, deregister and destroy
 * all complete cleanly on that path (no crash / error).
 *
 * Window register/deregister errors are tolerated (they depend on cuMem
 * availability); the comm must still register and destroy without error.
 * (Byte-level leak verification is done separately via the standalone ASan
 * reproducer, since the isolated test runner _exit()s and bypasses LSan.)
 */
static void testWindowRegisterSingleRankNonSymTeardown() {
    SKIP_IF_NO_GPU();

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    NCCLCHECK(initSingleRankComm(&comm));

    const size_t bufferSize = NCCL_WIN_REQUIRED_ALIGNMENT; // required alignment for window registration
    void* buf = nullptr;
    NCCLCHECK(ncclMemAlloc(&buf, bufferSize));

    ncclWindow_t win = nullptr;
    ncclResult_t reg = ncclCommWindowRegister(comm, buf, bufferSize, &win,
                                              NCCL_WIN_COLL_SYMMETRIC);
    if (reg == ncclSuccess && win != nullptr) {
        NCCLCHECK(ncclCommWindowDeregister(comm, win));
    }

    NCCLCHECK(ncclMemFree(buf));
    NCCLCHECK(ncclCommDestroy(comm));
}

//==============================================================================
// Test configuration helpers
//==============================================================================

// Environment configuration for explicitly disabled local registration
static ProcessIsolatedTestRunner::TestConfig
makeDisabledConfig(const std::string& name, std::function<void()> testFn) {
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_LOCAL_REGISTER", "0"}});
}

// Environment configuration for enabled registration
static ProcessIsolatedTestRunner::TestConfig
makeEnabledConfig(const std::string& name, std::function<void()> testFn) {
    return ProcessIsolatedTestRunner::TestConfig(name, testFn)
        .withEnvironment({{"NCCL_LOCAL_REGISTER", "1"}});
}

/**
 * @brief Test ncclCommRegister and ncclCommDeregister APIs with process isolation
 *
 * This test suite verifies that:
 * 1. A device buffer can be registered with ncclCommRegister (API returns success)
 * 2. When NCCL_LOCAL_REGISTER=1, the registration returns a valid (non-NULL) handle
 * 3. When NCCL_LOCAL_REGISTER=0, NULL handle is expected (local registration off)
 * 4. The buffer can be deregistered with ncclCommDeregister
 *
 * Note: On newer HIP, NCCL_LOCAL_REGISTER defaults to 1 when unset; the "disabled"
 * cases therefore set NCCL_LOCAL_REGISTER=0 explicitly rather than clearing the var.
 */
TEST(Register, ProcessIsolatedRegisterTests)
{
    RUN_ISOLATED_TESTS(
        // CommRegisterDeregister tests
        makeDisabledConfig("CommRegisterDeregister_Disabled",
            []() { testCommRegisterDeregister(false); }),
        makeEnabledConfig("CommRegisterDeregister_Enabled",
            []() { testCommRegisterDeregister(true); }),

        // MultipleBufferRegistration tests
        makeDisabledConfig("MultipleBufferRegistration_Disabled",
            []() { testMultipleBufferRegistration(false); }),
        makeEnabledConfig("MultipleBufferRegistration_Enabled",
            []() { testMultipleBufferRegistration(true); }),

        // VariableSizeBuffers tests
        makeDisabledConfig("VariableSizeBuffers_Disabled",
            []() { testVariableSizeBuffers(false); }),
        makeEnabledConfig("VariableSizeBuffers_Enabled",
            []() { testVariableSizeBuffers(true); }),

        // DeregisterNullHandle test (no enable/disable variants needed)
        ProcessIsolatedTestRunner::TestConfig("DeregisterNullHandle", testDeregisterNullHandle),

        // Regression test: NCCL GH#1859 — p2p followed by collective on the same
        // registered user buffer must not crash (requires ≥ 2 GPUs).
        ProcessIsolatedTestRunner::TestConfig(
            "P2pThenCollective_SameBuffer", testP2pThenCollectiveSameBuffer)
            .withEnvironment({{"NCCL_LOCAL_REGISTER", "1"},
                              {"NCCL_P2P_LL_THRESHOLD", "0"},
                              {"NCCL_LEGACY_CUDA_REGISTER", "1"},
                              {"NCCL_DEBUG", "INFO"},
                              {"NCCL_DEBUG_SUBSYS", "REG,P2P"}})
            .withNumGpus(2)
            .withTimeout(std::chrono::seconds(120)),

        // Force the non-sym path (NCCL_CUMEM_ENABLE=0 => symmetricSupport=false) so the
        // non-symmetric register/teardown is exercised; NCCL_LOCAL_REGISTER=1 ensures the register path runs.
        ProcessIsolatedTestRunner::TestConfig("WindowRegisterSingleRankNonSymTeardown", testWindowRegisterSingleRankNonSymTeardown)
            .withEnvironment({{"NCCL_CUMEM_ENABLE", "0"}, {"NCCL_LOCAL_REGISTER", "1"}})
    );
}

} // namespace RcclUnitTesting
