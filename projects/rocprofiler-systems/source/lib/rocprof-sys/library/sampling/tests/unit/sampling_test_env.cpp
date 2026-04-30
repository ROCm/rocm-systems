// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Global gtest environment for the rocprof-sys-unit-tests binary.
//
// Calls configure_settings(false) to construct the settings registry so that
// any code transitively reaching a get_*() accessor (e.g. through the
// interface-library link chain) does not hit recursive_init_error.
//
// Sampling-specific config (signals, frequencies, etc.) is no longer set here
// — the sampling_service constructor receives all config via sampling_config,
// and unit tests provide values through make_test_config().

#include "core/config.hpp"

#include <gtest/gtest.h>

namespace
{
class SamplingUnitTestConfigEnv : public ::testing::Environment
{
public:
    void SetUp() override { rocprofsys::configure_settings(false); }
};

[[maybe_unused]] auto* const _registered_env =
    ::testing::AddGlobalTestEnvironment(new SamplingUnitTestConfigEnv);
}  // namespace
