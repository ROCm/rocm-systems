/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Tests for the process-global Runtime root object in src/amd_detail/runtime.
//
// Runtime::backends() is the single home for backend construction (Fallback +
// Fastpath, built once via std::call_once). Because the C-API shims now inject
// backends as an explicit parameter, this is the only place that construction is
// exercised, so it gets its own coverage here.

#include "backend.h"
#include "hipfile-warnings.h"
#include "runtime.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace hipFile;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

// backends() must return the Fallback/Fastpath pair, both non-null.
TEST(Runtime, BackendsAreBuiltAndNonNull)
{
    const std::vector<std::shared_ptr<Backend>> &backends = Runtime::instance().backends();

    ASSERT_EQ(backends.size(), 2u);
    for (const auto &backend : backends) {
        EXPECT_NE(backend, nullptr);
    }
}

// backends() caches via std::call_once, so repeated calls must hand back the
// same collection (same element pointers), not rebuild it.
TEST(Runtime, BackendsAreStableAcrossCalls)
{
    const std::vector<std::shared_ptr<Backend>> &first  = Runtime::instance().backends();
    const std::vector<std::shared_ptr<Backend>> &second = Runtime::instance().backends();

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); i++) {
        EXPECT_EQ(first[i], second[i]);
    }
}

// state() must hand back the one owned DriverState, stable across calls.
TEST(Runtime, StateIsStableAcrossCalls)
{
    DriverState &first  = Runtime::instance().state();
    DriverState &second = Runtime::instance().state();

    EXPECT_EQ(&first, &second);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
