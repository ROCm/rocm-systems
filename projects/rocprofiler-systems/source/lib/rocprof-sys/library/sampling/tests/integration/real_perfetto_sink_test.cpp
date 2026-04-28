// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Smoke tests for real_perfetto_sink — the PerfettoSinkPolicy production type.
//
// The real sink calls config::get_use_perfetto() as its very first guard;
// when perfetto is disabled (the default), both emit_timer() and emit_overflow()
// return immediately without touching thread_info or the perfetto runtime.
// This makes it safe to call them in a unit-test binary that is not a perfetto
// session host.
//
// Verified behaviour:
//   - emit_timer(tid, nullptr, {})    — empty samples, no crash
//   - emit_overflow(tid, nullptr, {}) — empty samples, no crash
//   - emit_timer with samples when get_use_perfetto()==false — returns without
//     touching thread_info (second guard is samples.empty() then get_use_perfetto)
//
// Config initialization: configure_settings(false) pre-initializes the settings
// map so get_config() doesn't hit the recursive_init_error path.
// ROCPROFSYS_TRACE is false by default → get_use_perfetto() returns false.

#include <gtest/gtest.h>

#include "core/config.hpp"
#include "sampling_production_policies.hpp"

#include "sampling/data/overflow_sample.hpp"
#include "sampling/data/timer_sample.hpp"

#include <vector>

// ── Config environment ────────────────────────────────────────────────────────
namespace
{
class PerfettoSinkConfigEnv : public ::testing::Environment
{
public:
    void SetUp() override { rocprofsys::configure_settings(false); }
};
}  // namespace

using rocprofsys::sampling::real_perfetto_sink;

// ── emit_timer: empty vector — no crash ──────────────────────────────────────

TEST(real_perfetto_sink_smoke, emit_timer_empty_samples_does_not_crash)
{
    real_perfetto_sink                              sink;
    std::vector<rocprofsys::sampling::timer_sample> empty;
    EXPECT_NO_FATAL_FAILURE(sink.emit_timer(0, nullptr, empty))
        << "emit_timer() with empty samples must not crash (early-return guard)";
}

// ── emit_overflow: empty vector — no crash ───────────────────────────────────

TEST(real_perfetto_sink_smoke, emit_overflow_empty_samples_does_not_crash)
{
    real_perfetto_sink                                 sink;
    std::vector<rocprofsys::sampling::overflow_sample> empty;
    EXPECT_NO_FATAL_FAILURE(sink.emit_overflow(0, nullptr, empty))
        << "emit_overflow() with empty samples must not crash (early-return guard)";
}

// ── emit_timer: perfetto disabled (default) — returns before thread_info ─────

TEST(real_perfetto_sink_smoke, emit_timer_perfetto_disabled_does_not_access_thread_info)
{
    // Default config: ROCPROFSYS_TRACE=false → get_use_perfetto()==false.
    // emit_timer() returns at the very first check — never calls thread_info::get().
    real_perfetto_sink sink;

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
    real_perfetto_sink sink;

    rocprofsys::sampling::overflow_sample s;
    s.tid    = 0;
    s.beg_ns = 0;
    s.end_ns = 1'000'000ULL;

    std::vector<rocprofsys::sampling::overflow_sample> samples = { s };
    EXPECT_NO_FATAL_FAILURE(sink.emit_overflow(0, nullptr, samples))
        << "emit_overflow() with perfetto disabled must return without accessing "
           "thread_info";
}

// ── Type properties ───────────────────────────────────────────────────────────

static_assert(std::is_default_constructible_v<real_perfetto_sink>,
              "real_perfetto_sink must be default-constructible");

// ── Custom main ───────────────────────────────────────────────────────────────
int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PerfettoSinkConfigEnv);
    return RUN_ALL_TESTS();
}
