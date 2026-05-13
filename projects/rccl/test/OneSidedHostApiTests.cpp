/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Functional tests for the one-sided host APIs (put / wait) added in
// NCCL 2.29.2.
//
// The release notes describe these as zero-SM host-callable APIs for both
// network and NVL transport. At the time this test was written, the
// corresponding symbols are not yet exported by librccl. To keep the test
// file useful and linkable today while staying ready for when the
// implementation lands, we resolve the entry points lazily via dlsym() and
// skip cleanly if they're missing.
//
// The exact names of the put / wait entry points are not yet committed in
// the synced RCCL headers; we therefore probe a small set of plausible
// names so that the test picks up the implementation regardless of which
// naming variant lands first.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

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

// Look up the first symbol from a candidate list.
static void* resolveAny(const std::vector<const char*>& names) {
    for (const char* n : names) {
        void* p = dlsym(RTLD_DEFAULT, n);
        if (p) return p;
    }
    return nullptr;
}

static bool hostPutSymbolAvailable() {
    return resolveAny({"ncclWindowPut",     "ncclWinPut",
                       "ncclHostPut",       "ncclNetPut"}) != nullptr;
}
static bool hostWaitSymbolAvailable() {
    return resolveAny({"ncclWindowWait",    "ncclWinWait",
                       "ncclHostWait",      "ncclNetWait"}) != nullptr;
}

#define SKIP_IF_HOSTPUT_MISSING()                                              \
    do {                                                                       \
        if (!hostPutSymbolAvailable() || !hostWaitSymbolAvailable()) {         \
            GTEST_SKIP() << "Skipping: one-sided host put/wait symbols are "   \
                            "not exported by librccl. This release of RCCL "   \
                            "does not yet implement the API; the test is "    \
                            "preserved for once it lands.";                    \
            return;                                                            \
        }                                                                      \
    } while (0)

// At least exercise the symbol-discovery path and the documented "requires
// CUDA 12.5 or greater" gating. The HIP equivalent gating is left for the
// implementer to express; for now we just verify a clean skip.
static void testHostPutWait_SymbolsDiscovered() {
    SKIP_IF_NO_GPU();
    SKIP_IF_HOSTPUT_MISSING();

    // Reach here only when the symbols exist. The actual functional test
    // (rank 0 puts to rank 1's window, rank 1 waits) would go here once the
    // public type/signature is settled. For now record an informational
    // pass that the symbols are resolvable.
    SUCCEED() << "Host put/wait symbols are present; functional drive will "
                 "be filled in once the API signatures are finalized.";
}

TEST(OneSidedHostApi, ProcessIsolatedSuite)
{
    RUN_ISOLATED_TESTS(
        ProcessIsolatedTestRunner::TestConfig("HostPutWait_SymbolsDiscovered",
            testHostPutWait_SymbolsDiscovered)
    );
}

} // namespace RcclUnitTesting
