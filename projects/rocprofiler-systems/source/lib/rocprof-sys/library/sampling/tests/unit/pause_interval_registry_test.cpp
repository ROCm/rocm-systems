// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Direct tests for pause_interval_registry<fake_clock> — NFR-T-8.
// Covers: double-pause (L37), double-resume (L39), spans_pause_interval,
// max_resume_ns_overlapping, intervals stored in chronological order.

#include <gtest/gtest.h>

#include "doubles/fake_clock.hpp"
#include "sampling/src/pause_interval_registry.hpp"

using namespace rocprofsys::sampling;
using namespace rocprofsys::sampling::test;
using test_registry = pause_interval_registry<fake_clock>;

// Helper: construct a fresh clock+registry pair with clock reset to 0.
static std::pair<fake_clock, test_registry*>
make_registry()
{
    fake_clock::reset(0);
    static test_registry* reg = nullptr;  // static so registry outlives call
    return {};
}

// ─── NFR-T-8: double-pause — returns false, no-op ────────────────────────────

TEST(pause_interval_registry, double_pause_returns_false)
{
    fake_clock::reset(0);
    fake_clock    clk;
    test_registry reg(clk);

    EXPECT_TRUE(reg.pause()) << "First pause() must return true";
    EXPECT_FALSE(reg.pause())
        << "Second pause() with no intervening resume() must return false";
}

TEST(pause_interval_registry, double_pause_does_not_add_interval)
{
    fake_clock::reset(0);
    fake_clock    clk;
    test_registry reg(clk);

    reg.pause();
    reg.pause();  // no-op

    // Only one pause was recorded — but no interval is stored until resume.
    EXPECT_TRUE(reg.intervals().empty())
        << "No interval must be stored until resume() is called";
    EXPECT_TRUE(reg.is_paused()) << "Registry must still be paused after double-pause";
}

// ─── NFR-T-8: double-resume — returns false, no-op ───────────────────────────

TEST(pause_interval_registry, double_resume_returns_false)
{
    fake_clock::reset(0);
    fake_clock    clk;
    test_registry reg(clk);

    reg.pause();
    fake_clock::advance_ns(100);
    EXPECT_TRUE(reg.resume()) << "First resume() must return true";
    EXPECT_FALSE(reg.resume())
        << "Second resume() with no intervening pause() must return false";
}

TEST(pause_interval_registry, double_resume_does_not_add_extra_interval)
{
    fake_clock::reset(0);
    fake_clock    clk;
    test_registry reg(clk);

    reg.pause();
    fake_clock::advance_ns(100);
    reg.resume();
    reg.resume();  // no-op

    EXPECT_EQ(reg.intervals().size(), 1U)
        << "Exactly one interval stored after one pause+resume cycle";
}

// ─── NFR-T-8: spans_pause_interval filters correctly ─────────────────────────

TEST(pause_interval_registry, spans_pause_interval_detects_overlap)
{
    fake_clock::reset(0);
    fake_clock    clk;
    test_registry reg(clk);

    // Record interval [100, 200].
    fake_clock::reset(100);
    reg.pause();
    fake_clock::advance_ns(100);
    reg.resume();

    // [50, 150] overlaps [100, 200].
    EXPECT_TRUE(reg.spans_pause_interval(50U, 150U))
        << "Interval [50,150] overlaps pause [100,200]";

    // [50, 99] does NOT overlap [100, 200].
    EXPECT_FALSE(reg.spans_pause_interval(50U, 99U))
        << "Interval [50,99] does not overlap pause [100,200]";

    // [201, 300] does NOT overlap [100, 200].
    EXPECT_FALSE(reg.spans_pause_interval(201U, 300U))
        << "Interval [201,300] does not overlap pause [100,200]";
}

TEST(pause_interval_registry, spans_pause_interval_exact_boundary)
{
    fake_clock::reset(100);
    fake_clock    clk;
    test_registry reg(clk);

    // Record interval [100, 200].
    reg.pause();
    fake_clock::advance_ns(100);
    reg.resume();

    // Half-open convention: overlap is defined as (resume_ns > beg_ns && pause_ns <
    // end_ns). A sample touching only at a single point (one endpoint equals exactly
    // pause_ns or resume_ns) is NOT considered overlapping — it completed just before the
    // pause started, or started exactly when the pause ended.

    // [200, 300]: beg_ns==resume_ns — sample starts exactly when pause ended.
    // Not overlapping (half-open: resume_ns(200) > beg_ns(200) is false).
    EXPECT_FALSE(reg.spans_pause_interval(200U, 300U))
        << "Sample starting at resume_ns did not overlap the pause (half-open "
           "convention)";

    // [0, 100]: end_ns==pause_ns — sample ends exactly when pause started.
    // Not overlapping (half-open: pause_ns(100) < end_ns(100) is false).
    EXPECT_FALSE(reg.spans_pause_interval(0U, 100U))
        << "Sample ending at pause_ns did not overlap the pause (half-open convention)";

    // A sample fully inside the pause interval IS overlapping.
    EXPECT_TRUE(reg.spans_pause_interval(110U, 180U))
        << "Sample fully within pause interval must overlap";

    // A sample straddling the pause start IS overlapping.
    EXPECT_TRUE(reg.spans_pause_interval(50U, 150U))
        << "Sample straddling pause start must overlap";

    // A sample straddling the pause end IS overlapping.
    EXPECT_TRUE(reg.spans_pause_interval(150U, 250U))
        << "Sample straddling pause end must overlap";
}

// ─── NFR-T-8: intervals stored in chronological (append) order ───────────────

TEST(pause_interval_registry, intervals_stored_in_chronological_order)
{
    fake_clock::reset(0);
    fake_clock    clk;
    test_registry reg(clk);

    // Interval 1: [10, 20].
    fake_clock::reset(10);
    reg.pause();
    fake_clock::advance_ns(10);
    reg.resume();  // resume at 20

    // Interval 2: [50, 100].
    fake_clock::reset(50);
    reg.pause();
    fake_clock::advance_ns(50);
    reg.resume();  // resume at 100

    auto const& ivs = reg.intervals();
    ASSERT_EQ(ivs.size(), 2U);
    EXPECT_LT(ivs.at(0).resume_ns, ivs.at(1).pause_ns)
        << "First interval must end before second begins (chronological order)";
    EXPECT_EQ(ivs.at(0).pause_ns, 10U);
    EXPECT_EQ(ivs.at(0).resume_ns, 20U);
    EXPECT_EQ(ivs.at(1).pause_ns, 50U);
    EXPECT_EQ(ivs.at(1).resume_ns, 100U);
}

// ─── max_resume_ns_overlapping — returns correct value ───────────────────────

TEST(pause_interval_registry, max_resume_ns_overlapping_returns_highest)
{
    fake_clock::reset(100);
    fake_clock    clk;
    test_registry reg(clk);

    // Interval [100, 200].
    reg.pause();
    fake_clock::advance_ns(100);
    reg.resume();  // resume at 200

    // Interval [300, 500].
    fake_clock::reset(300);
    reg.pause();
    fake_clock::advance_ns(200);
    reg.resume();  // resume at 500

    // [150, 350] overlaps both intervals.
    uint64_t max_res = reg.max_resume_ns_overlapping(150U, 350U);
    EXPECT_EQ(max_res, 500U) << "max_resume_ns_overlapping must return the highest "
                                "resume_ns across all overlapping intervals";
}

TEST(pause_interval_registry, max_resume_ns_overlapping_returns_zero_when_no_overlap)
{
    fake_clock::reset(100);
    fake_clock    clk;
    test_registry reg(clk);

    reg.pause();
    fake_clock::advance_ns(100);
    reg.resume();  // interval [100, 200]

    uint64_t max_res = reg.max_resume_ns_overlapping(300U, 400U);
    EXPECT_EQ(max_res, 0U) << "No overlap: max_resume_ns_overlapping must return 0";
}

// ─── is_paused tracks state ───────────────────────────────────────────────────

TEST(pause_interval_registry, is_paused_tracks_state)
{
    fake_clock::reset(0);
    fake_clock    clk;
    test_registry reg(clk);

    EXPECT_FALSE(reg.is_paused()) << "Initially not paused";

    reg.pause();
    EXPECT_TRUE(reg.is_paused()) << "After pause()";

    fake_clock::advance_ns(10);
    reg.resume();
    EXPECT_FALSE(reg.is_paused()) << "After resume()";
}

// ─── empty registry: spans_pause_interval always false ────────────────────────

TEST(pause_interval_registry, empty_registry_never_spans_pause)
{
    fake_clock::reset(0);
    fake_clock    clk;
    test_registry reg(clk);

    EXPECT_FALSE(reg.spans_pause_interval(0U, 1'000'000'000U))
        << "Empty registry must never report a pause interval";
}
