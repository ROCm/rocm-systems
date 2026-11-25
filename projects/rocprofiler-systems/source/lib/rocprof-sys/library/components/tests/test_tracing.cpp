// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/categories.hpp"
#include "library/thread_deleter.hpp"
#include "library/tracing.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <thread>

// The real definition finalizes the process and is not linked into the unit tests
void
rocprofsys::thread_deleter<void>::operator()() const
{}

namespace
{
using category_t                    = rocprofsys::category::host;
using runtime_enabled_t             = tim::trait::runtime_enabled<category_t>;
constexpr std::string_view k_region = "region";

class scoped_runtime_enabled
{
public:
    explicit scoped_runtime_enabled(bool enabled)
    : m_previous{ runtime_enabled_t::get() }
    {
        runtime_enabled_t::set(enabled);
    }

    ~scoped_runtime_enabled() { runtime_enabled_t::set(m_previous); }

    scoped_runtime_enabled(const scoped_runtime_enabled&)            = delete;
    scoped_runtime_enabled& operator=(const scoped_runtime_enabled&) = delete;

private:
    bool m_previous;
};
}  // namespace

TEST(tracing_push_timemory, region_pushed_in_a_trace_window_can_be_popped_after_it)
{
    std::thread{ [] {
        ASSERT_FALSE(rocprofsys::tracing::has_timemory_components());

        const scoped_runtime_enabled window_open{ true };
        rocprofsys::tracing::push_timemory(category_t{}, k_region);

        runtime_enabled_t::set(false);
        EXPECT_FALSE(rocprofsys::tracing::category_pop_disabled<category_t>());
    } }.join();
}

TEST(tracing_push_timemory, nothing_was_pushed_so_a_closed_window_disables_pop)
{
    std::thread{ [] {
        const scoped_runtime_enabled window_closed{ false };
        rocprofsys::tracing::push_timemory(category_t{}, k_region);
        EXPECT_TRUE(rocprofsys::tracing::category_pop_disabled<category_t>());
    } }.join();
}
