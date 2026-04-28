// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for native_report_writer — NFR-T-7, NFR-P-4.

#include <gtest/gtest.h>

#include "sampling/data/stack_frame.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/src/native_report_writer.hpp"

#include <sstream>
#include <vector>

using namespace rocprofsys::sampling;

// Helper: build a minimal timer_sample.
static timer_sample
make_timer_sample(int64_t tid, uint64_t beg, uint64_t end, std::string const& func_name)
{
    timer_sample sample;
    sample.tid    = tid;
    sample.beg_ns = beg;
    sample.end_ns = end;
    stack_frame frame;
    frame.name = func_name;
    sample.stack.push_back(frame);
    return sample;
}

// ─── NFR-T-7: Output structure — required columns present ────────────────────

TEST(native_report_writer, output_contains_required_columns)
{
    std::ostringstream wall_out;
    std::ostringstream cpu_out;
    std::ostringstream pct_out;
    std::ostringstream trip_out;

    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    std::vector<timer_sample> samples = {
        make_timer_sample(0, 0, 2'000'000'000U, "alpha"),
        make_timer_sample(0, 500'000'000U, 1'500'000'000U, "beta"),
    };

    writer.write_timer_samples(0, samples);
    writer.flush();

    const auto out = wall_out.str();
    EXPECT_NE(out.find("alpha"), std::string::npos) << "must contain first function name";
    EXPECT_NE(out.find("beta"), std::string::npos) << "must contain second function name";
    // SUM column: alpha has 2.0s → row contains "2.000000".
    EXPECT_NE(out.find("2."), std::string::npos) << "must contain a 2.x second value";
}

// ─── Smoke test: native_report_writer produces non-empty output ─────────────────

TEST(native_report_writer, produces_non_empty_output_for_single_sample)
{
    std::ostringstream wall_clock_out;
    std::ostringstream cpu_clock_out;
    std::ostringstream percent_out;
    std::ostringstream trip_out;

    native_report_writer writer(wall_clock_out, cpu_clock_out, percent_out, trip_out);

    const int64_t             tid     = 0;
    std::vector<timer_sample> samples = {
        make_timer_sample(tid, 0, 1'000'000'000U, "main"),
    };

    writer.write_timer_samples(tid, samples);
    writer.flush();

    EXPECT_FALSE(wall_clock_out.str().empty())
        << "native_report_writer must write to wall_clock stream";
    EXPECT_NE(wall_clock_out.str().find("main"), std::string::npos)
        << "Output must include the function name";
}

// ─── Structural parity: wall_clock uses seconds with 6 decimal places ─────────

TEST(native_report_writer, wall_clock_uses_six_decimal_places)
{
    std::ostringstream wall_out;
    std::ostringstream cpu_out;
    std::ostringstream pct_out;
    std::ostringstream trip_out;

    native_report_writer writer(wall_out, cpu_out, pct_out, trip_out);

    // 1 second exactly => should appear as "1.000000" in the output
    std::vector<timer_sample> samples = {
        make_timer_sample(0, 0, 1'000'000'000U, "test_func"),
    };

    writer.write_timer_samples(0, samples);
    writer.flush();

    // Check that the SUM column contains "1.000000" (6 decimal places)
    auto out_str = wall_out.str();
    EXPECT_NE(out_str.find("1.000000"), std::string::npos)
        << "wall_clock output must use 6 decimal places for seconds "
           "(required by NFR-P-4 / architecture § 9)";
}
