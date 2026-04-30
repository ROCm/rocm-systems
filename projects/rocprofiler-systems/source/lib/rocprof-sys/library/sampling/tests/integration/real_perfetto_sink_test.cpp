// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Smoke tests for perfetto_sink_impl — the PerfettoSinkPolicy production template.
//
// The sink receives use_perfetto as a constructor argument. When false (the
// default in tests), both emit_timer() and emit_overflow() return immediately
// without touching thread_info or the perfetto runtime.

#include <gtest/gtest.h>

#include "core/config.hpp"
#include "sampling/default_policies.hpp"

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <vector>

namespace
{
class PerfettoSinkConfigEnv : public ::testing::Environment
{
public:
    void SetUp() override { rocprofsys::configure_settings(false); }
};

using prod_perfetto_sink = rocprofsys::sampling::default_perfetto_sink;
using resolver_t         = rocprofsys::sampling::real_thread_info_resolver;
}  // namespace

TEST(real_perfetto_sink_smoke, emit_timer_empty_samples_does_not_crash)
{
    resolver_t         resolver;
    prod_perfetto_sink sink{ resolver, false, false };

    std::vector<rocprofsys::sampling::timer_sample> empty;
    EXPECT_NO_FATAL_FAILURE(sink.emit_timer(0, nullptr, empty))
        << "emit_timer() with empty samples must not crash (early-return guard)";
}

TEST(real_perfetto_sink_smoke, emit_overflow_empty_samples_does_not_crash)
{
    resolver_t         resolver;
    prod_perfetto_sink sink{ resolver, false, false };

    std::vector<rocprofsys::sampling::overflow_sample> empty;
    EXPECT_NO_FATAL_FAILURE(sink.emit_overflow(0, nullptr, empty))
        << "emit_overflow() with empty samples must not crash (early-return guard)";
}

TEST(real_perfetto_sink_smoke, emit_timer_perfetto_disabled_does_not_access_thread_info)
{
    resolver_t         resolver;
    prod_perfetto_sink sink{ resolver, false, false };

    rocprofsys::sampling::timer_sample s;
    s.tid    = 0;
    s.beg_ns = 0;
    s.end_ns = 1'000'000ULL;

    std::vector<rocprofsys::sampling::timer_sample> samples = { s };
    EXPECT_NO_FATAL_FAILURE(sink.emit_timer(0, nullptr, samples))
        << "emit_timer() with perfetto disabled must return without accessing "
           "thread_info";
}

TEST(real_perfetto_sink_smoke,
     emit_overflow_perfetto_disabled_does_not_access_thread_info)
{
    resolver_t         resolver;
    prod_perfetto_sink sink{ resolver, false, false };

    rocprofsys::sampling::overflow_sample s;
    s.tid    = 0;
    s.beg_ns = 0;
    s.end_ns = 1'000'000ULL;

    std::vector<rocprofsys::sampling::overflow_sample> samples = { s };
    EXPECT_NO_FATAL_FAILURE(sink.emit_overflow(0, nullptr, samples))
        << "emit_overflow() with perfetto disabled must return without accessing "
           "thread_info";
}

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PerfettoSinkConfigEnv);
    return RUN_ALL_TESTS();
}
