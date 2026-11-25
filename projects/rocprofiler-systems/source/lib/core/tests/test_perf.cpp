// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "common/clocks.hpp"
#include "core/perf.hpp"

#include <linux/perf_event.h>

#include <cstdint>
#include <ctime>
#include <ostream>
#include <stdexcept>
#include <string_view>

namespace
{
using rocprofsys::perf::event_type;
using rocprofsys::perf::get_event_type;
using rocprofsys::perf::get_hw_cache_config;

constexpr unsigned int k_op_shift     = 8;
constexpr unsigned int k_result_shift = 16;

// Layout defined by perf_event_open(2) for PERF_TYPE_HW_CACHE:
// cache id | (op id << 8) | (result id << 16).
constexpr int
expected_cache_config(int cache, int cache_op, int cache_result)
{
    return cache | (cache_op << k_op_shift) | (cache_result << k_result_shift);
}

struct valid_cache_name
{
    std::string_view name;
    int              expected_config;
};

// Without this, gtest prints a failing parameter as raw bytes.
std::ostream&
operator<<(std::ostream& out, const valid_cache_name& value)
{
    return out << value.name;
}

constexpr int k_read   = PERF_COUNT_HW_CACHE_OP_READ;
constexpr int k_write  = PERF_COUNT_HW_CACHE_OP_WRITE;
constexpr int k_access = PERF_COUNT_HW_CACHE_RESULT_ACCESS;
constexpr int k_miss   = PERF_COUNT_HW_CACHE_RESULT_MISS;
}  // namespace

// --- get_event_type ---

// regression: the hw_cache branch tested PERF_COUNT_SW_ twice and was
// unreachable.
TEST(perf_event_type, hw_cache_name_is_classified_as_hw_cache)
{
    EXPECT_EQ(get_event_type("perf::PERF_COUNT_HW_CACHE_L1D"), event_type::hw_cache);
}

TEST(perf_event_type, hw_cache_name_with_modifiers_is_classified_as_hw_cache)
{
    EXPECT_EQ(get_event_type("perf::PERF_COUNT_HW_CACHE_L1D:READ:MISS"),
              event_type::hw_cache);
}

// The hw_cache check must not swallow these two plain hardware events that share its
// prefix.
TEST(perf_event_type, cache_references_is_classified_as_hardware)
{
    EXPECT_EQ(get_event_type("perf::PERF_COUNT_HW_CACHE_REFERENCES"),
              event_type::hardware);
}

TEST(perf_event_type, cache_misses_is_classified_as_hardware)
{
    EXPECT_EQ(get_event_type("perf::PERF_COUNT_HW_CACHE_MISSES"), event_type::hardware);
}

TEST(perf_event_type, cpu_cycles_is_classified_as_hardware)
{
    EXPECT_EQ(get_event_type("perf::PERF_COUNT_HW_CPU_CYCLES"), event_type::hardware);
}

TEST(perf_event_type, software_name_is_classified_as_software)
{
    EXPECT_EQ(get_event_type("perf::PERF_COUNT_SW_CPU_CLOCK"), event_type::software);
}

TEST(perf_event_type, unknown_name_is_classified_as_max)
{
    EXPECT_EQ(get_event_type("perf::INSTRUCTIONS"), event_type::max);
}

// --- get_hw_cache_config: accepted names ---
// The grammar and the defaults (READ, MISS) are those of the libpfm4 perf event table.

using perf_hw_cache_valid_name = ::testing::TestWithParam<valid_cache_name>;

TEST_P(perf_hw_cache_valid_name, selects_cache_operation_and_result)
{
    const auto param  = GetParam();
    const auto config = get_hw_cache_config(param.name);

    EXPECT_EQ(config, param.expected_config) << param.name;
}

INSTANTIATE_TEST_SUITE_P(
    every_cache_id_with_default_modifiers, perf_hw_cache_valid_name,
    ::testing::Values(valid_cache_name{ .name            = "PERF_COUNT_HW_CACHE_L1D",
                                        .expected_config = expected_cache_config(
                                            PERF_COUNT_HW_CACHE_L1D, k_read, k_miss) },
                      valid_cache_name{ .name            = "PERF_COUNT_HW_CACHE_L1I",
                                        .expected_config = expected_cache_config(
                                            PERF_COUNT_HW_CACHE_L1I, k_read, k_miss) },
                      valid_cache_name{ .name            = "PERF_COUNT_HW_CACHE_LL",
                                        .expected_config = expected_cache_config(
                                            PERF_COUNT_HW_CACHE_LL, k_read, k_miss) },
                      valid_cache_name{ .name            = "PERF_COUNT_HW_CACHE_DTLB",
                                        .expected_config = expected_cache_config(
                                            PERF_COUNT_HW_CACHE_DTLB, k_read, k_miss) },
                      valid_cache_name{ .name            = "PERF_COUNT_HW_CACHE_ITLB",
                                        .expected_config = expected_cache_config(
                                            PERF_COUNT_HW_CACHE_ITLB, k_read, k_miss) },
                      valid_cache_name{ .name            = "PERF_COUNT_HW_CACHE_BPU",
                                        .expected_config = expected_cache_config(
                                            PERF_COUNT_HW_CACHE_BPU, k_read, k_miss) },
                      valid_cache_name{ .name            = "PERF_COUNT_HW_CACHE_NODE",
                                        .expected_config = expected_cache_config(
                                            PERF_COUNT_HW_CACHE_NODE, k_read, k_miss) }));

// regression: the result was derived from the operation (WRITE meant MISS, a
// READ could never be a MISS), and a result modifier was never read.
INSTANTIATE_TEST_SUITE_P(
    modifiers, perf_hw_cache_valid_name,
    ::testing::Values(
        // one modifier, as rocprof-sys-avail lists them
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_LL:ACCESS",
                          .expected_config = expected_cache_config(PERF_COUNT_HW_CACHE_LL,
                                                                   k_read, k_access) },
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_LL:MISS",
                          .expected_config = expected_cache_config(PERF_COUNT_HW_CACHE_LL,
                                                                   k_read, k_miss) },
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_LL:READ",
                          .expected_config = expected_cache_config(PERF_COUNT_HW_CACHE_LL,
                                                                   k_read, k_miss) },
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_LL:WRITE",
                          .expected_config = expected_cache_config(PERF_COUNT_HW_CACHE_LL,
                                                                   k_write, k_miss) },
        valid_cache_name{
            .name            = "perf::PERF_COUNT_HW_CACHE_LL:PREFETCH",
            .expected_config = expected_cache_config(
                PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_PREFETCH, k_miss) },
        // operation and result together, in either order
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_L1D:READ:MISS",
                          .expected_config = expected_cache_config(
                              PERF_COUNT_HW_CACHE_L1D, k_read, k_miss) },
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_L1D:READ:ACCESS",
                          .expected_config = expected_cache_config(
                              PERF_COUNT_HW_CACHE_L1D, k_read, k_access) },
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_L1D:WRITE:ACCESS",
                          .expected_config = expected_cache_config(
                              PERF_COUNT_HW_CACHE_L1D, k_write, k_access) },
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_L1D:WRITE:MISS",
                          .expected_config = expected_cache_config(
                              PERF_COUNT_HW_CACHE_L1D, k_write, k_miss) },
        valid_cache_name{ .name            = "perf::PERF_COUNT_HW_CACHE_L1D:ACCESS:WRITE",
                          .expected_config = expected_cache_config(
                              PERF_COUNT_HW_CACHE_L1D, k_write, k_access) }));

// --- get_hw_cache_config: rejected names ---

using perf_hw_cache_invalid_name = ::testing::TestWithParam<std::string_view>;

TEST_P(perf_hw_cache_invalid_name, throws_runtime_error)
{
    EXPECT_THROW(get_hw_cache_config(GetParam()), std::runtime_error) << GetParam();
}

INSTANTIATE_TEST_SUITE_P(
    perf_hw_cache, perf_hw_cache_invalid_name,
    ::testing::Values(
        "perf::PERF_COUNT_HW_CPU_CYCLES",            // not a cache event
        "perf::PERF_COUNT_HW_CACHE_",                // no cache id
        "perf::PERF_COUNT_HW_CACHE_BOGUS",           // unknown cache id
        "perf::PERF_COUNT_HW_CACHE_L1DX",            // known id followed by extra text
        "perf::PERF_COUNT_HW_CACHE_L1D:BOGUS",       // unknown modifier
        "perf::PERF_COUNT_HW_CACHE_L1D:READ:BOGUS",  // unknown modifier after a valid one
        "perf::PERF_COUNT_HW_CACHE_L1D:READ:WRITE",  // two operations
        "perf::PERF_COUNT_HW_CACHE_L1D:MISS:ACCESS",  // two results
        "perf::PERF_COUNT_HW_CACHE_L1D:",             // empty modifier
        "perf::PERF_COUNT_HW_CACHE_L1D::MISS"));      // empty modifier in the middle

// --- config_overflow_sampling ---

TEST(perf_config_overflow_sampling, hw_cache_event_fills_type_config_and_period)
{
    constexpr double k_events_per_sample = 1000.0;

    perf_event_attr attr{};
    rocprofsys::perf::config_overflow_sampling(
        attr, "perf::PERF_COUNT_HW_CACHE_L1D:WRITE:ACCESS", k_events_per_sample);

    EXPECT_EQ(attr.type, static_cast<std::uint32_t>(PERF_TYPE_HW_CACHE));
    EXPECT_EQ(attr.config, static_cast<std::uint64_t>(expected_cache_config(
                               PERF_COUNT_HW_CACHE_L1D, k_write, k_access)));
    EXPECT_EQ(attr.sample_period, static_cast<std::uint64_t>(k_events_per_sample));
}

// Guards the other side of the fix: the default overflow event stays a plain hardware
// event.
TEST(perf_config_overflow_sampling, cache_references_stays_hardware_event)
{
    constexpr double k_events_per_sample = 1000.0;

    perf_event_attr attr{};
    rocprofsys::perf::config_overflow_sampling(
        attr, "perf::PERF_COUNT_HW_CACHE_REFERENCES", k_events_per_sample);

    EXPECT_EQ(attr.type, static_cast<std::uint32_t>(PERF_TYPE_HARDWARE));
    EXPECT_EQ(attr.config, static_cast<std::uint64_t>(PERF_COUNT_HW_CACHE_REFERENCES));
}

// --- config_overflow_sampling: sample clock ---

namespace
{
struct sample_clock_case
{
    std::string_view event;
    clockid_t        expected_clock;
};

std::ostream&
operator<<(std::ostream& out, const sample_clock_case& value)
{
    return out << value.event;
}
}  // namespace

using perf_overflow_sample_clock = ::testing::TestWithParam<sample_clock_case>;

TEST_P(perf_overflow_sample_clock, selects_clock_the_kernel_accepts)
{
    constexpr double k_events_per_sample = 1000.0;
    const auto       param               = GetParam();

    perf_event_attr attr{};
    rocprofsys::perf::config_overflow_sampling(attr, param.event, k_events_per_sample);
    const std::uint64_t uses_clockid = attr.use_clockid;

    EXPECT_EQ(uses_clockid, 1U);
    EXPECT_EQ(attr.clockid, param.expected_clock);
}

// regression: every overflow event asked for the timeline clock
// (CLOCK_BOOTTIME), and perf_event_open() fails with EINVAL for that clock on a hardware
// event.
INSTANTIATE_TEST_SUITE_P(
    hardware_events_use_monotonic, perf_overflow_sample_clock,
    ::testing::Values(sample_clock_case{ .event = "perf::PERF_COUNT_HW_CACHE_REFERENCES",
                                         .expected_clock = CLOCK_MONOTONIC },
                      sample_clock_case{ .event = "perf::PERF_COUNT_HW_CPU_CYCLES",
                                         .expected_clock = CLOCK_MONOTONIC },
                      sample_clock_case{
                          .event          = "perf::PERF_COUNT_HW_CACHE_L1D:READ:MISS",
                          .expected_clock = CLOCK_MONOTONIC }));

INSTANTIATE_TEST_SUITE_P(
    software_events_use_timeline_clock, perf_overflow_sample_clock,
    ::testing::Values(
        sample_clock_case{
            .event          = "perf::PERF_COUNT_SW_CPU_CLOCK",
            .expected_clock = rocprofsys::clocks::timeline::k_timeline_clock_id },
        sample_clock_case{
            .event          = "perf::PERF_COUNT_SW_PAGE_FAULTS",
            .expected_clock = rocprofsys::clocks::timeline::k_timeline_clock_id }));
