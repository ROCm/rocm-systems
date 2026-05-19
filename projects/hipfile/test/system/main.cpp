/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "test-options.h"

#include <chrono> // IWYU pragma: keep
#include <cstdlib>
#include <gtest/gtest.h>
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

int
main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    test_env.parseTestOptions(argc, argv);
    std::atexit(sleepOnExit);

    return RUN_ALL_TESTS();
}
