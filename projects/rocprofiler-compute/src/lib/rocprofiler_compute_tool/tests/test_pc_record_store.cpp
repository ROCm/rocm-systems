// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "gtest/gtest.h"
#include "pc_record_store.h"

#include <thread>
#include <vector>

using namespace rocprofiler_compute_tool;

namespace
{
pc_sample_record_t make_record(uint64_t code_object_id, uint64_t offset, uint64_t dispatch_id)
{
    pc_sample_record_t record{};
    record.code_object_id     = code_object_id;
    record.code_object_offset = offset;
    record.dispatch_id        = dispatch_id;
    return record;
}
}  // namespace

TEST(pc_record_store_t, NewStoreIsEmpty)
{
    pc_record_store_t store;
    EXPECT_TRUE(store.stochastic().empty());
    EXPECT_TRUE(store.host_trap().empty());
}

TEST(pc_record_store_t, AddStochasticAndHostTrapAreSeparated)
{
    pc_record_store_t store;

    pc_sample_record_t stochastic = make_record(1, 0x10, 100);
    stochastic.stall_reason       = "NO_INSTRUCTION_AVAILABLE";
    stochastic.wave_issued        = 1;
    store.add_stochastic(stochastic);

    store.add_host_trap(make_record(2, 0x20, 200));

    ASSERT_EQ(store.stochastic().size(), 1u);
    ASSERT_EQ(store.host_trap().size(), 1u);

    EXPECT_EQ(store.stochastic()[0].code_object_id, 1u);
    EXPECT_EQ(store.stochastic()[0].code_object_offset, 0x10u);
    EXPECT_EQ(store.stochastic()[0].dispatch_id, 100u);
    EXPECT_EQ(store.stochastic()[0].stall_reason, "NO_INSTRUCTION_AVAILABLE");
    EXPECT_EQ(store.stochastic()[0].wave_issued, 1);

    EXPECT_EQ(store.host_trap()[0].code_object_id, 2u);
    EXPECT_EQ(store.host_trap()[0].dispatch_id, 200u);
}

TEST(pc_record_store_t, PreservesInsertionOrder)
{
    pc_record_store_t store;
    for (uint64_t i = 0; i < 5; ++i)
        store.add_stochastic(make_record(i, i * 0x10, i));

    ASSERT_EQ(store.stochastic().size(), 5u);
    for (uint64_t i = 0; i < 5; ++i)
        EXPECT_EQ(store.stochastic()[i].dispatch_id, i);
}

TEST(pc_record_store_t, ConcurrentAddsAreThreadSafe)
{
    pc_record_store_t store;

    constexpr int            kThreads          = 8;
    constexpr int            kRecordsPerThread = 1000;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t)
    {
        threads.emplace_back(
            [&store, t]()
            {
                for (int i = 0; i < kRecordsPerThread; ++i)
                {
                    // Half the threads write stochastic, half host_trap.
                    if (t % 2 == 0)
                        store.add_stochastic(make_record(t, i, i));
                    else
                        store.add_host_trap(make_record(t, i, i));
                }
            });
    }

    for (auto& thread : threads)
        thread.join();

    constexpr int kStochasticThreads = kThreads / 2;
    constexpr int kHostTrapThreads   = kThreads - kStochasticThreads;
    EXPECT_EQ(store.stochastic().size(), static_cast<size_t>(kStochasticThreads * kRecordsPerThread));
    EXPECT_EQ(store.host_trap().size(), static_cast<size_t>(kHostTrapThreads * kRecordsPerThread));
}
