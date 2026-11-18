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

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace
{
uint64_t           signal_allocations = 0;
uint64_t           signal_resets      = 0;
hsa_signal_value_t signal_value       = 0;
}  // namespace

TEST(hsa, pooled_signal_reuses_handle_and_resets_value)
{
    namespace hsa = ::rocprofiler::hsa;

    signal_allocations = 0;
    signal_resets      = 0;
    signal_value       = 0;

    auto* core = hsa::get_core_table();
    auto* ext  = hsa::get_amd_ext_table();
    ASSERT_NE(core, nullptr);
    ASSERT_NE(ext, nullptr);

    auto old_create = ext->hsa_amd_signal_create_fn;
    auto old_store  = core->hsa_signal_store_screlease_fn;
    auto restore    = ::rocprofiler::common::scope_destructor{[&]() {
        ext->hsa_amd_signal_create_fn       = old_create;
        core->hsa_signal_store_screlease_fn = old_store;
    }};

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
}

TEST(hsa, device_signal_accounting)
{
    namespace hsa = ::rocprofiler::hsa;

    auto signal       = hsa::signal_t{};
    signal.idle_value = 100;

    // one decrement per completed dispatch
    EXPECT_TRUE(hsa::retire_device_signal(signal, 99));
    EXPECT_EQ(signal.idle_value, 99);

    // not yet decremented (dispatch still running, or a stale read): unchanged
    EXPECT_FALSE(hsa::retire_device_signal(signal, 99));
    EXPECT_EQ(signal.idle_value, 99);

    // decremented twice
    EXPECT_FALSE(hsa::retire_device_signal(signal, 97));
    EXPECT_EQ(signal.idle_value, 99);
}

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x17
namespace
{
struct batch_create_call
{
    uint32_t    count         = 0;
    uint32_t    num_consumers = 0;
    hsa_agent_t consumer      = {};
    uint64_t    attributes    = 0;
    uint64_t    pool          = 0;
};

std::vector<batch_create_call> batch_create_calls  = {};
hsa_status_t                   batch_create_status = HSA_STATUS_SUCCESS;
uint64_t                       batch_destroys      = 0;
uint64_t                       signal_destroys     = 0;
uint64_t                       next_device_signal  = 0;
}  // namespace

TEST(hsa, device_signal_source_creates_gpu_only_signals_for_its_agent)
{
    namespace hsa = ::rocprofiler::hsa;

    batch_create_calls.clear();
    batch_create_status = HSA_STATUS_SUCCESS;
    batch_destroys      = 0;
    signal_destroys     = 0;
    next_device_signal  = 0;
    signal_allocations  = 0;

    auto* core = hsa::get_core_table();
    auto* ext  = hsa::get_amd_ext_table();
    ASSERT_NE(core, nullptr);
    ASSERT_NE(ext, nullptr);

    auto old_batch_create  = ext->hsa_amd_signal_batch_create_fn;
    auto old_batch_destroy = ext->hsa_amd_signal_batch_destroy_fn;
    auto old_create        = ext->hsa_amd_signal_create_fn;
    auto old_destroy       = core->hsa_signal_destroy_fn;
    auto restore           = ::rocprofiler::common::scope_destructor{[&]() {
        ext->hsa_amd_signal_batch_create_fn  = old_batch_create;
        ext->hsa_amd_signal_batch_destroy_fn = old_batch_destroy;
        ext->hsa_amd_signal_create_fn        = old_create;
        core->hsa_signal_destroy_fn          = old_destroy;
    }};

    ext->hsa_amd_signal_batch_create_fn = +[](uint32_t                  count,
                                              const hsa_signal_value_t* values,
                                              uint32_t                  num_consumers,
                                              const hsa_agent_t*        consumers,
                                              uint64_t                  attributes,
                                              hsa_amd_memory_pool_t     pool,
                                              hsa_amd_signal_batch_t*   batch,
                                              hsa_signal_t*             signals) {
        batch_create_calls.emplace_back(
            batch_create_call{count,
                              num_consumers,
                              num_consumers ? consumers[0] : hsa_agent_t{},
                              attributes,
                              pool.handle});
        if(batch_create_status != HSA_STATUS_SUCCESS) return batch_create_status;
        EXPECT_EQ(values[0], std::numeric_limits<hsa_signal_value_t>::max());
        // aligned like amd_signal_t
        for(uint32_t i = 0; i < count; ++i)
            signals[i].handle = 64 * ++next_device_signal;
        batch->handle = 1;
        return HSA_STATUS_SUCCESS;
    };
    ext->hsa_amd_signal_batch_destroy_fn = +[](hsa_amd_signal_batch_t) {
        ++batch_destroys;
        return HSA_STATUS_SUCCESS;
    };
    ext->hsa_amd_signal_create_fn =
        +[](hsa_signal_value_t, uint32_t, const hsa_agent_t*, uint64_t, hsa_signal_t* signal) {
            signal->handle = 1000000 + ++signal_allocations;
            return HSA_STATUS_SUCCESS;
        };
    core->hsa_signal_destroy_fn = +[](hsa_signal_t) {
        ++signal_destroys;
        return HSA_STATUS_SUCCESS;
    };

    auto source        = hsa::device_signal_source{};
    source.agent       = hsa_agent_t{.handle = 42};
    source.memory_pool = hsa_amd_memory_pool_t{.handle = 7};

    auto first = hsa::signal_t{};
    source.construct(first);
    ASSERT_EQ(batch_create_calls.size(), 1);
    EXPECT_EQ(batch_create_calls[0].num_consumers, 1);
    EXPECT_EQ(batch_create_calls[0].consumer.handle, 42);
    EXPECT_EQ(batch_create_calls[0].attributes, HSA_AMD_SIGNAL_AMD_GPU_ONLY);
    EXPECT_EQ(batch_create_calls[0].pool, 7);
    EXPECT_EQ(batch_destroys, 1);
    EXPECT_TRUE(first.device_memory);
    EXPECT_EQ(first.idle_value, std::numeric_limits<hsa_signal_value_t>::max());
    EXPECT_EQ(first.value.handle % 64, 0);

    // the rest of the batch is handed out without another allocation
    const auto batch = batch_create_calls[0].count;
    for(uint32_t i = 1; i < batch; ++i)
    {
        auto signal = hsa::signal_t{};
        source.construct(signal);
        EXPECT_TRUE(signal.device_memory);
    }
    EXPECT_EQ(batch_create_calls.size(), 1);
    EXPECT_EQ(source.created, batch);
    EXPECT_EQ(source.handed_out, batch);

    // the next batch fails: host signals from then on, without retrying
    batch_create_status = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    for(int i = 0; i < 2; ++i)
    {
        auto signal = hsa::signal_t{};
        source.construct(signal);
        EXPECT_FALSE(signal.device_memory);
        EXPECT_GT(signal.value.handle, 1000000);
    }
    EXPECT_EQ(batch_create_calls.size(), 2);
    EXPECT_EQ(signal_allocations, 2);

    EXPECT_EQ(source.destroy_unused(), 0);
    EXPECT_EQ(signal_destroys, 0);
}

TEST(hsa, device_signal_source_destroys_unused)
{
    namespace hsa = ::rocprofiler::hsa;

    batch_create_calls.clear();
    batch_create_status = HSA_STATUS_SUCCESS;
    signal_destroys     = 0;
    next_device_signal  = 0;

    auto* core = hsa::get_core_table();
    auto* ext  = hsa::get_amd_ext_table();
    ASSERT_NE(core, nullptr);
    ASSERT_NE(ext, nullptr);

    auto old_batch_create  = ext->hsa_amd_signal_batch_create_fn;
    auto old_batch_destroy = ext->hsa_amd_signal_batch_destroy_fn;
    auto old_destroy       = core->hsa_signal_destroy_fn;
    auto restore           = ::rocprofiler::common::scope_destructor{[&]() {
        ext->hsa_amd_signal_batch_create_fn  = old_batch_create;
        ext->hsa_amd_signal_batch_destroy_fn = old_batch_destroy;
        core->hsa_signal_destroy_fn          = old_destroy;
    }};

    ext->hsa_amd_signal_batch_create_fn = +[](uint32_t count,
                                              const hsa_signal_value_t*,
                                              uint32_t,
                                              const hsa_agent_t*,
                                              uint64_t,
                                              hsa_amd_memory_pool_t,
                                              hsa_amd_signal_batch_t* batch,
                                              hsa_signal_t*           signals) {
        batch_create_calls.emplace_back();
        for(uint32_t i = 0; i < count; ++i)
            signals[i].handle = 64 * ++next_device_signal;
        batch->handle = 1;
        return HSA_STATUS_SUCCESS;
    };
    ext->hsa_amd_signal_batch_destroy_fn =
        +[](hsa_amd_signal_batch_t) { return HSA_STATUS_SUCCESS; };
    core->hsa_signal_destroy_fn = +[](hsa_signal_t) {
        ++signal_destroys;
        return HSA_STATUS_SUCCESS;
    };

    auto source        = hsa::device_signal_source{};
    source.agent       = hsa_agent_t{.handle = 1};
    source.memory_pool = hsa_amd_memory_pool_t{.handle = 1};

    auto signal = hsa::signal_t{};
    source.construct(signal);

    const auto unused = source.created - source.handed_out;
    EXPECT_EQ(source.destroy_unused(), unused);
    EXPECT_EQ(signal_destroys, unused);
    EXPECT_TRUE(source.unused.empty());
}
#endif
