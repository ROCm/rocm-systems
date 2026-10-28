// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/trace_cache/wall_clock_processor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

using rocprofsys::trace_cache::kernel_dispatch_sample;
using rocprofsys::trace_cache::metadata_registry;
using rocprofsys::trace_cache::region_sample;
using rocprofsys::trace_cache::wall_clock_processor_t;

TEST(wall_clock_processor, kernel_with_unregistered_symbol_throws)
{
    constexpr std::uint64_t k_unknown_kernel = 999;
    constexpr std::uint64_t k_begin_ns       = 0;
    constexpr std::uint64_t k_end_ns         = 1'000'000;
    constexpr std::uint64_t k_thread         = 1;
    wall_clock_processor_t  processor{
        std::make_shared<metadata_registry>(), nullptr, 1, {}
    };
    EXPECT_THROW(processor.handle(kernel_dispatch_sample{ k_begin_ns, k_end_ns, k_thread,
                                                          0, k_unknown_kernel, 0, 0, 0, 0,
                                                          0, 0, 1, 1, 1, 1, 1, 1, 0 }),
                 std::runtime_error);
}

TEST(wall_clock_processor, region_ending_before_it_begins_is_skipped)
{
    constexpr std::uint64_t k_begin_ns = 2'000'000;
    constexpr std::uint64_t k_end_ns   = 1'000'000;
    constexpr std::uint64_t k_thread   = 1;
    wall_clock_processor_t  processor{
        std::make_shared<metadata_registry>(), nullptr, 1, {}
    };
    EXPECT_NO_THROW(processor.handle(
        region_sample{ k_thread, "reversed", 0, 0, k_begin_ns, k_end_ns, "{}", "", "" }));
}
