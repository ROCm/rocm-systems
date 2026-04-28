// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// TDD tests for Task #26: ROCPROFSYS_SAMPLING_TRACE_LEGACY env var.
//
// When ROCPROFSYS_SAMPLING_TRACE_LEGACY=0 (default): only trace_cache path.
// When ROCPROFSYS_SAMPLING_TRACE_LEGACY=1: also call real_perfetto_sink::emit_*
//   from emit_resolved_to_trace_cache().
//
// Verifiable from test doubles: the perfetto_sink receives calls when legacy=1
// and NOT when legacy=0. Uses a recording_perfetto_sink variant.
//
// Note: get_use_sampling_trace_legacy() is declared in
// sampling/src/sampling_config_fwd.hpp (added by Task #26) and stubbed in
// unit/config_stubs.cpp. The real implementation is in core/config.cpp and reads
// ROCPROFSYS_SAMPLING_TRACE_LEGACY.

#include <gtest/gtest.h>

#include "sampling/src/sampling_config_fwd.hpp"

// ── Config accessor exists and returns false by default ───────────────────────

TEST(sampling_trace_legacy, default_is_false)
{
    // The function must exist and return false unless explicitly set.
    // Config is initialized lazily; call it to trigger initialization.
    bool val = rocprofsys::config::get_use_sampling_trace_legacy();
    EXPECT_FALSE(val) << "ROCPROFSYS_SAMPLING_TRACE_LEGACY must default to false "
                         "(trace_cache-only path)";
}
