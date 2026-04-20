// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include <memory>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
namespace
{
TEST(QueueIntercept, QueueStateDefaultInit)
{
    QueueState state;

    EXPECT_EQ(state.ring_buf, nullptr);
    EXPECT_EQ(state.ring_size, 0U);
    EXPECT_EQ(state.ring_mask, 0U);
    EXPECT_EQ(state.virtual_wptr.load(), 0UL);
    EXPECT_EQ(state.real_wdid, nullptr);
    EXPECT_EQ(state.real_rdid, nullptr);
    EXPECT_EQ(state.next_scan_pos, 0UL);
    EXPECT_EQ(state.next_submit_pos, 0UL);
    EXPECT_EQ(state.doorbell_signal.handle, 0UL);
    EXPECT_EQ(state.k_factor, 0UL);
    EXPECT_EQ(state.metadata_state, nullptr);
}

TEST(QueueIntercept, RegistryInsertAndLookup)
{
    // Create a dummy queue pointer for testing
    // We're just using a dummy address; we never dereference it
    hsa_queue_t        dummy_queue;
    const hsa_queue_t* queue_ptr = &dummy_queue;

    // Create a QueueState and insert it into the registry
    auto state       = std::make_unique<QueueState>();
    state->ring_size = 1024;
    state->ring_mask = 1023;

    QueueState* state_ptr = state.get();

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = std::move(state); });

    // Look up the state
    QueueState* found_state = lookup_queue_state(queue_ptr);
    ASSERT_NE(found_state, nullptr);
    EXPECT_EQ(found_state, state_ptr);
    EXPECT_EQ(found_state->ring_size, 1024U);
    EXPECT_EQ(found_state->ring_mask, 1023U);

    // Clean up
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });

    // Verify removal
    QueueState* after_removal = lookup_queue_state(queue_ptr);
    EXPECT_EQ(after_removal, nullptr);
}

TEST(QueueIntercept, DoorbellMapInsertAndLookup)
{
    // Create a dummy queue for testing
    hsa_queue_t        dummy_queue;
    const hsa_queue_t* queue_ptr = &dummy_queue;

    // Create and register a QueueState
    auto state            = std::make_unique<QueueState>();
    state->ring_size      = 2048;
    QueueState* state_ptr = state.get();

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = std::move(state); });

    // Create a doorbell signal and register it
    hsa_signal_t doorbell;
    doorbell.handle = 0x12345678;

    register_doorbell(queue_ptr, doorbell);

    // Look up by doorbell
    QueueState* found_state = lookup_queue_state_by_doorbell(doorbell);
    ASSERT_NE(found_state, nullptr);
    EXPECT_EQ(found_state, state_ptr);
    EXPECT_EQ(found_state->ring_size, 2048U);

    // Unregister doorbell
    unregister_doorbell(doorbell);

    // Verify removal
    QueueState* after_removal = lookup_queue_state_by_doorbell(doorbell);
    EXPECT_EQ(after_removal, nullptr);

    // Clean up queue registry
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });
}

TEST(QueueIntercept, AddWriteIndexAdvancesVirtualWptr)
{
    QueueState state{};
    uint64_t   idx0 = add_write_index_impl(&state, 1);
    EXPECT_EQ(idx0, 0u);
    EXPECT_EQ(state.virtual_wptr.load(), 1u);

    uint64_t idx1 = add_write_index_impl(&state, 3);
    EXPECT_EQ(idx1, 1u);
    EXPECT_EQ(state.virtual_wptr.load(), 4u);
}

TEST(QueueIntercept, StoreWriteIndexSetsVirtualWptr)
{
    QueueState state{};
    store_write_index_impl(&state, 42);
    EXPECT_EQ(state.virtual_wptr.load(), 42u);
    store_write_index_impl(&state, 0);
    EXPECT_EQ(state.virtual_wptr.load(), 0u);
}

TEST(QueueIntercept, CasWriteIndexSuccess)
{
    QueueState state{};
    state.virtual_wptr.store(10);
    uint64_t prev = cas_write_index_impl(&state, 10, 20);
    EXPECT_EQ(prev, 10u);
    EXPECT_EQ(state.virtual_wptr.load(), 20u);
}

TEST(QueueIntercept, CasWriteIndexFailure)
{
    QueueState state{};
    state.virtual_wptr.store(10);
    uint64_t prev = cas_write_index_impl(&state, 5, 20);
    EXPECT_EQ(prev, 10u);
    EXPECT_EQ(state.virtual_wptr.load(), 10u);
}

TEST(QueueIntercept, LoadWriteIndexReturnsVirtualWptr)
{
    QueueState state{};
    state.virtual_wptr.store(99);
    EXPECT_EQ(load_write_index_impl(&state), 99u);
}

}  // namespace
}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
