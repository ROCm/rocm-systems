/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "test-options.h"

#include <chrono> // IWYU pragma: keep
#include <cstdlib>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <thread>

extern SystemTestOptions test_env;
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF
HIPFILE_WARN_NO_EXIT_DTOR_OFF
SystemTestOptions test_env;
HIPFILE_WARN_NO_EXIT_DTOR_ON
HIPFILE_WARN_NO_GLOBAL_CTOR_ON

static void
sleepOnExit()
{
    if (test_env.sleep_seconds > 0) {
        std::this_thread::sleep_for(std::chrono::seconds(test_env.sleep_seconds));
    }
}

// Reset every HIP device once all tests have finished, so per-device runtime
// allocations are released before LSan reports leaks. The irreducible HSA-init
// allocations that survive hipDeviceReset are handled by lsan.supp.
class HipDeviceResetEnv : public ::testing::Environment {
public:
    void TearDown() override
    {
        int device_count = 0;
        if (::hipGetDeviceCount(&device_count) != hipSuccess) {
            return;
        }
        for (int i = 0; i < device_count; ++i) {
            (void)::hipSetDevice(i);
            (void)::hipDeviceReset();
        }
    }
};

int
main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new HipDeviceResetEnv);

    test_env.parseTestOptions(argc, argv);
    std::atexit(sleepOnExit);

    return RUN_ALL_TESTS();
}
