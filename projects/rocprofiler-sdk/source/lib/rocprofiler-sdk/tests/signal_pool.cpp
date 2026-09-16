// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/signal_pool.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <gtest/gtest.h>
#include <map>

namespace
{
uint64_t                     signal_allocations = 0;
uint64_t                     signal_resets      = 0;
hsa_signal_value_t           signal_value       = 0;
std::map<uint64_t, uint64_t> destroyed_signals{};
}  // namespace

TEST(hsa, pooled_signal_reuses_handle_and_resets_value)
{
    namespace hsa = ::rocprofiler::hsa;

    signal_allocations = 0;
    signal_resets      = 0;
    signal_value       = 0;
    destroyed_signals.clear();

    auto* core = hsa::get_core_table();
    auto* ext  = hsa::get_amd_ext_table();
    ASSERT_NE(core, nullptr);
    ASSERT_NE(ext, nullptr);
    hsa::signal_pool_fini();

    auto old_create  = ext->hsa_amd_signal_create_fn;
    auto old_store   = core->hsa_signal_store_screlease_fn;
    auto old_destroy = core->hsa_signal_destroy_fn;
    auto restore     = ::rocprofiler::common::scope_destructor{[&]() {
        ext->hsa_amd_signal_create_fn       = old_create;
        core->hsa_signal_store_screlease_fn = old_store;
        core->hsa_signal_destroy_fn         = old_destroy;
    }};
    auto cleanup     = ::rocprofiler::common::scope_destructor{[]() { hsa::signal_pool_fini(); }};

    ext->hsa_amd_signal_create_fn = +[](hsa_signal_value_t initial,
                                        uint32_t,
                                        const hsa_agent_t*,
                                        uint64_t,
                                        hsa_signal_t* signal) {
        signal->handle = ++signal_allocations;
        signal_value   = initial;
        return HSA_STATUS_SUCCESS;
    };
    core->hsa_signal_store_screlease_fn = +[](hsa_signal_t, hsa_signal_value_t initial) {
        ++signal_resets;
        signal_value = initial;
    };

    core->hsa_signal_destroy_fn = +[](hsa_signal_t signal) {
        ++destroyed_signals[signal.handle];
        return HSA_STATUS_SUCCESS;
    };

    ::rocprofiler::common::container::pool<hsa::signal_t> pool{
        std::piecewise_construct, 1, [](auto& signal) { hsa::construct_hsa_signal(signal); }};

    constexpr auto iterations = size_t{1000};
    auto           last       = hsa_signal_t{};
    for(size_t i = 0; i < iterations; ++i)
    {
        auto& slot = pool.acquire(hsa::construct_hsa_signal, 7, 0, nullptr, 0);
        last       = slot.get().value;
        EXPECT_EQ(signal_value, 7);
        signal_value = -1;
        EXPECT_TRUE(slot.release());
    }

    EXPECT_EQ(signal_allocations, 1);
    EXPECT_EQ(last, (hsa_signal_t{.handle = 1}));
    EXPECT_EQ(signal_resets, iterations);

    EXPECT_FALSE(hsa::get_signal_pool());
    hsa::signal_pool_init();
    auto first_handle = hsa_signal_t{};
    auto first        = hsa::Queue::create_signal(0, &first_handle, true);
    ASSERT_TRUE(first);
    auto first_alias = first;
    EXPECT_TRUE(first->in_use());
    hsa::signal_pool_fini();
    ASSERT_EQ(destroyed_signals.count(first_handle.handle), 0u);
    EXPECT_TRUE(first->in_use());

    hsa::signal_pool_init();
    auto second_handle = hsa_signal_t{};
    auto second        = hsa::Queue::create_signal(0, &second_handle, true);
    ASSERT_TRUE(second);
    EXPECT_NE(first_handle, second_handle);
    hsa::Queue::release_signal(first);
    EXPECT_EQ(destroyed_signals.count(first_handle.handle), 0u);
    first = nullptr;
    EXPECT_EQ(destroyed_signals.count(first_handle.handle), 0u);
    first_alias = nullptr;
    EXPECT_EQ(destroyed_signals[first_handle.handle], 1u);

    hsa::signal_pool_fini();
    ASSERT_EQ(destroyed_signals.count(second_handle.handle), 0u);
    hsa::Queue::release_signal(second);
    second = nullptr;
    EXPECT_EQ(destroyed_signals[second_handle.handle], 1u);
}
