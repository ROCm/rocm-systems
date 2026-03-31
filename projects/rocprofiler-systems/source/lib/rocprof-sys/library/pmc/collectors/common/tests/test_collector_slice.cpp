// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/pmc/collectors/common/collector_slice.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace rocprofsys::pmc::collectors::testing
{

// Mock collector implementations for testing type erasure
class mock_gpu_collector
{
public:
    void setup() { setup_called = true; }
    void config() { config_called = true; }
    void sample(int64_t ts)
    {
        sample_called  = true;
        last_timestamp = ts;
    }
    void post_process() { post_process_called = true; }
    void shutdown() { shutdown_called = true; }

    bool    setup_called        = false;
    bool    config_called       = false;
    bool    sample_called       = false;
    int64_t last_timestamp      = 0;
    bool    post_process_called = false;
    bool    shutdown_called     = false;
};

class mock_nic_collector
{
public:
    void setup() { setup_called = true; }
    void config() { config_called = true; }
    void sample(int64_t ts)
    {
        sample_count++;
        last_timestamp = ts;
    }
    void post_process() { post_process_called = true; }
    void shutdown() { shutdown_called = true; }

    bool    setup_called        = false;
    bool    config_called       = false;
    int     sample_count        = 0;
    int64_t last_timestamp      = 0;
    bool    post_process_called = false;
    bool    shutdown_called     = false;
};

class collector_slice_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gpu_collector = std::make_unique<mock_gpu_collector>();
        nic_collector = std::make_unique<mock_nic_collector>();
    }

    void TearDown() override
    {
        gpu_collector.reset();
        nic_collector.reset();
    }

    std::unique_ptr<mock_gpu_collector> gpu_collector;
    std::unique_ptr<mock_nic_collector> nic_collector;
};

TEST_F(collector_slice_test, single_collector_calls_all_methods)
{
    collector_slice slice(*gpu_collector);

    EXPECT_FALSE(gpu_collector->setup_called);
    slice.setup();
    EXPECT_TRUE(gpu_collector->setup_called);

    EXPECT_FALSE(gpu_collector->config_called);
    slice.config();
    EXPECT_TRUE(gpu_collector->config_called);

    EXPECT_FALSE(gpu_collector->sample_called);
    slice.sample(12345);
    EXPECT_TRUE(gpu_collector->sample_called);
    EXPECT_EQ(gpu_collector->last_timestamp, 12345);

    EXPECT_FALSE(gpu_collector->post_process_called);
    slice.post_process();
    EXPECT_TRUE(gpu_collector->post_process_called);

    EXPECT_FALSE(gpu_collector->shutdown_called);
    slice.shutdown();
    EXPECT_TRUE(gpu_collector->shutdown_called);
}

TEST_F(collector_slice_test, heterogeneous_collectors_in_vector)
{
    std::vector<collector_slice> slices;
    slices.emplace_back(*gpu_collector);
    slices.emplace_back(*nic_collector);

    EXPECT_EQ(slices.size(), 2u);

    for(auto& slice : slices)
    {
        slice.setup();
    }
    EXPECT_TRUE(gpu_collector->setup_called);
    EXPECT_TRUE(nic_collector->setup_called);

    for(auto& slice : slices)
    {
        slice.config();
    }
    EXPECT_TRUE(gpu_collector->config_called);
    EXPECT_TRUE(nic_collector->config_called);

    for(auto& slice : slices)
    {
        slice.sample(1000);
    }
    EXPECT_TRUE(gpu_collector->sample_called);
    EXPECT_EQ(nic_collector->sample_count, 1);

    for(auto& slice : slices)
    {
        slice.sample(2000);
    }
    EXPECT_EQ(nic_collector->sample_count, 2);
}

TEST_F(collector_slice_test, collector_slice_is_non_owning)
{
    collector_slice slice(*gpu_collector);

    // Modify the original collector
    gpu_collector->setup_called = true;

    // The slice should reflect the change (non-owning view)
    slice.sample(5000);
    EXPECT_TRUE(gpu_collector->sample_called);
    EXPECT_TRUE(gpu_collector->setup_called);  // Still true from manual set
}

TEST_F(collector_slice_test, multiple_slices_to_same_collector)
{
    collector_slice slice1(*gpu_collector);
    collector_slice slice2(*gpu_collector);

    // Both slices reference the same underlying object
    slice1.setup();
    EXPECT_TRUE(gpu_collector->setup_called);

    // Calling on slice2 should see the already-called setup
    slice2.config();
    EXPECT_TRUE(gpu_collector->setup_called);
    EXPECT_TRUE(gpu_collector->config_called);
}

TEST_F(collector_slice_test, collectors_can_be_different_types)
{
    std::vector<collector_slice> slices;
    slices.emplace_back(*gpu_collector);
    slices.emplace_back(*nic_collector);

    for(auto& slice : slices)
    {
        slice.sample(100);
        slice.sample(200);
        slice.sample(300);
    }

    // Each collector maintains its own state
    EXPECT_TRUE(gpu_collector->sample_called);  // bool, so just true
    EXPECT_EQ(nic_collector->sample_count, 3);  // int counter
}

}  // namespace rocprofsys::pmc::collectors::testing
