/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile-warnings.h"
#include "hipfile.h"

#include "ais-capability.h"
#include "test-options.h"

#include <array>
#include <gtest/gtest.h>
#include <string>

extern SystemTestOptions test_env;

enum class IoTestBackend {
    Fastpath,
    Fallback,
};

struct IoTestParam {
    IoTestBackend backend;
    std::string   name;
};

HIPFILE_WARN_NO_EXIT_DTOR_OFF
inline const std::array<IoTestParam, 2> io_test_params{
    {{IoTestBackend::Fastpath, "Fastpath"}, {IoTestBackend::Fallback, "Fallback"}}};
HIPFILE_WARN_NO_EXIT_DTOR_ON

// Gate fastpath-only tests on AIS capability.
inline void
enforceFastpathGate(hipFileHandle_t handle, void *device_buffer)
{
    hipFile::test::AisCapability ais_capability{test_env.allow_skip_fastpath};

    const auto decision = ais_capability.populate(handle, device_buffer);

    if (decision == hipFile::test::AisCapability::GateDecision::Run) {
        return;
    }

    if (decision == hipFile::test::AisCapability::GateDecision::Skip) {
        // Keep this marker synchronized with test/CMakeLists.txt SKIP_REGULAR_EXPRESSION.
        GTEST_SKIP() << "fastpath not available in this environment\n" << ais_capability.report();
    }

    FAIL() << "Fastpath Validation Failed!\n" << ais_capability.report() << "\n" << ais_capability.skipHint();
}
