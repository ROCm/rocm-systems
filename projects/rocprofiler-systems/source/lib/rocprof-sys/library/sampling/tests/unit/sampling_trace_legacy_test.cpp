// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Verifies that sampling_config::trace_legacy defaults to false.

#include <gtest/gtest.h>

#include "sampling/sampling_config.hpp"

TEST(sampling_trace_legacy, default_is_false)
{
    rocprofsys::sampling::sampling_config cfg;
    EXPECT_FALSE(cfg.trace_legacy)
        << "sampling_config::trace_legacy must default to false "
           "(trace_cache-only path)";
}
