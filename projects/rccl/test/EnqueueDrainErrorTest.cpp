/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <dlfcn.h>
#include <atomic>

/*
 * Validates the RCCL port of NVIDIA NCCL PR #1864
 *   "Drain error code when kernel is not found"
 *   https://github.com/NVIDIA/nccl/pull/1864
 *
 * Bug: src/enqueue.cc::ncclInitKernelsForDevice() loops over kernel function
 * pointers calling hipFuncGetAttributes() (hipified from cudaFuncGetAttributes).
 * If a lookup fails, the unfixed code does `if (...) continue;` without draining
 * the per-thread sticky last-error slot. The next user kernel launch then sees
 * a spurious stale error from hipGetLastError().
 *
 * This test deterministically forces ONE hipFuncGetAttributes call to fail
 * during ncclCommInitAll, then verifies that hipGetLastError() after a
 * subsequent benign user kernel returns hipSuccess.
 *
 * Mechanism: we define hipFuncGetAttributes directly in the test binary
 * (with a .symver alias to match the versioned symbol that librccl.so
 * imports).  The test binary is linked with -rdynamic so this symbol is
 * exported into its dynamic symbol table.  At runtime, the dynamic linker
 * resolves librccl's import of hipFuncGetAttributes@hip_4.2 against the
 * main executable BEFORE libamdhip64.so (default ELF resolution order).
 * When injection is disarmed (g_inject_remaining == 0) the override is a
 * transparent pass-through to the real symbol, so other tests are unaffected.
 *
 * Single-thread invariant: HIP last-error is thread-local. Init, kernel launch,
 * and the gate check must all be on the same host thread.
 */

namespace {
    std::atomic<int> g_inject_remaining{0};
    std::atomic<int> g_inject_count{0};

    using fn_t = hipError_t (*)(hipFuncAttributes*, const void*);
    fn_t real_hipFuncGetAttributes() {
        static fn_t cached =
            reinterpret_cast<fn_t>(dlsym(RTLD_NEXT, "hipFuncGetAttributes"));
        return cached;
    }
}

extern "C" hipError_t intercept_hipFuncGetAttributes(hipFuncAttributes* attr,
                                                     const void* func) {
    fn_t real = real_hipFuncGetAttributes();
    if (g_inject_remaining.load() > 0) {
        if (g_inject_remaining.fetch_sub(1) > 0) {
            g_inject_count.fetch_add(1);
            /* Poison HIP's per-thread sticky last_error_ with a real failing
             * call.  Passing nullptr makes the real runtime return
             * hipErrorInvalidValue and write that into thread-local state. */
            if (real) (void)real(nullptr, nullptr);
            return hipErrorInvalidDeviceFunction;
        }
    }
    return real ? real(attr, func) : hipErrorNotInitialized;
}

/* Bind our function to the exact versioned symbol librccl.so imports
 * (hipFuncGetAttributes@hip_4.2).  The "@@" form makes it the default
 * version, exported to other shared objects via -rdynamic. */
__asm__(".symver intercept_hipFuncGetAttributes,hipFuncGetAttributes@@hip_4.2");

namespace {
    __global__ void drain_test_noop_kernel() {}
}

namespace RcclUnitTesting
{
    TEST(EnqueueDrainError, DoesNotLeakStickyErrorToUserKernel) {
        int devCount = 0;
        ASSERT_EQ(hipGetDeviceCount(&devCount), hipSuccess);
        if (devCount < 1) GTEST_SKIP() << "needs at least 1 GPU";

        ASSERT_EQ(hipSetDevice(0), hipSuccess);

        /* Step 1: clean baseline BEFORE init.  Draining AFTER init would
         * mask the very bug under test. */
        (void)hipGetLastError();

        /* Step 2: arm exactly one fault injection. */
        g_inject_count.store(0);
        g_inject_remaining.store(1);

        /* Step 3: real RCCL init on a single rank.  The interceptor fires
         * once during ncclInitKernelsForDevice() inside this call. */
        ncclComm_t comm = nullptr;
        int devs[1] = {0};
        ASSERT_EQ(ncclCommInitAll(&comm, 1, devs), ncclSuccess)
            << "ncclCommInitAll failed; bug under test does not block init";

        /* Step 4: confirm the bug condition was actually exercised.  Without
         * this guard a missed injection would falsely "pass". */
        ASSERT_EQ(g_inject_count.load(), 1)
            << "interceptor never fired; check that test binary was linked "
               "with -rdynamic and that librccl.so still calls "
               "hipFuncGetAttributes during ncclInitKernelsForDevice";

        /* Step 5: launch a benign user kernel — what PyTorch does next. */
        drain_test_noop_kernel<<<1, 1, 0, 0>>>();

        /* Step 6: THE GATE.  Read sticky error immediately, no intervening
         * HIP calls.  With PR #1864's drain in place this is hipSuccess;
         * without it, the hipErrorInvalidValue from step 3 leaks here. */
        hipError_t after = hipGetLastError();

        /* Confirm the user kernel itself was actually fine, AFTER the gate. */
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);

        EXPECT_EQ(after, hipSuccess)
            << "Sticky HIP error (" << static_cast<int>(after) << " = "
            << hipGetErrorString(after)
            << ") leaked from ncclInitKernelsForDevice into the user kernel "
               "launch path. PR #1864 fix is NOT in effect.";
    }
}
