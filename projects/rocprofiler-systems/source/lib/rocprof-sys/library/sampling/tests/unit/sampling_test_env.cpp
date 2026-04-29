// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Global gtest environment for the rocprof-sys-unit-tests binary.
//
// The sampling-service unit suites instantiate sampling_service<Policies>
// and call setup() / shutdown(). Those entry points now drive the real
// rocprofsys::config accessors (the config_stubs.cpp shadow file is gone)
// because the production_hooks fold pulled the lifecycle methods directly
// into sampling_service. The first call to get_sampling_realtime_signal()
// chains into get_config(), whose Meyers-singleton magic-static would hit
// __gnu_cxx::recursive_init_error if it has not been primed.
//
// Calling rocprofsys::configure_settings(false) once before any test runs
// constructs the settings registry without launching the rest of the
// rocprof-sys runtime, mirroring the integration suites
// (sampling_service_production_hooks_test, etc.). false skips the Perfetto /
// gotcha bring-up that the unit tests do not need.

#include "core/config.hpp"

#include <gtest/gtest.h>

namespace
{
class SamplingUnitTestConfigEnv : public ::testing::Environment
{
public:
    void SetUp() override
    {
        rocprofsys::configure_settings(false);
        // Enable both timer-based signals so setup() returns a non-empty
        // signal set under the test config (the previous unit-test config
        // stub returned {RT, CPU} unconditionally; matching that behaviour
        // here keeps the existing assertion contracts intact).
        rocprofsys::set_setting_value("ROCPROFSYS_USE_SAMPLING", true);
        rocprofsys::set_setting_value("ROCPROFSYS_SAMPLING_REALTIME", true);
        rocprofsys::set_setting_value("ROCPROFSYS_SAMPLING_CPUTIME", true);
    }
};

// Register the environment at static-init time. AddGlobalTestEnvironment
// returns the pointer it stored, so discarding it here is intentional.
[[maybe_unused]] auto* const _registered_env =
    ::testing::AddGlobalTestEnvironment(new SamplingUnitTestConfigEnv);
}  // namespace
