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

#include "../source/types.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace rocprofiler
{
namespace python
{
namespace test
{
/**
 * @brief Tests for CounterInfo struct
 */
class CounterInfoTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CounterInfoTest, DefaultConstruction)
{
    CounterInfo info;

    EXPECT_EQ(info.id, 0);
    EXPECT_TRUE(info.name.empty());
    EXPECT_TRUE(info.description.empty());
    EXPECT_TRUE(info.block.empty());
    EXPECT_TRUE(info.expression.empty());
    EXPECT_FALSE(info.is_constant);
    EXPECT_FALSE(info.is_derived);
}

TEST_F(CounterInfoTest, InitializationWithValues)
{
    CounterInfo info;
    info.id          = 42;
    info.name        = "SQ_WAVES";
    info.description = "Number of waves";
    info.block       = "SQ";
    info.expression  = "";
    info.is_constant = false;
    info.is_derived  = false;

    EXPECT_EQ(info.id, 42);
    EXPECT_EQ(info.name, "SQ_WAVES");
    EXPECT_EQ(info.description, "Number of waves");
    EXPECT_EQ(info.block, "SQ");
    EXPECT_FALSE(info.is_constant);
    EXPECT_FALSE(info.is_derived);
}

TEST_F(CounterInfoTest, DerivedCounterFlag)
{
    CounterInfo info;
    info.name       = "VALU_UTILIZATION";
    info.expression = "(SQ_INSTS_VALU / (SQ_WAVES * 64)) * 100";
    info.is_derived = true;

    EXPECT_TRUE(info.is_derived);
    EXPECT_FALSE(info.expression.empty());
}

/**
 * @brief Tests for AgentInfo struct
 */
class AgentInfoTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AgentInfoTest, DefaultConstruction)
{
    AgentInfo agent;

    EXPECT_EQ(agent.id, 0);
    EXPECT_TRUE(agent.name.empty());
    EXPECT_TRUE(agent.product_name.empty());
    EXPECT_EQ(agent.device_index, 0);
    EXPECT_EQ(agent.gfx_version, 0);
}

TEST_F(AgentInfoTest, InitializationWithValues)
{
    AgentInfo agent;
    agent.id           = 1234;
    agent.name         = "gfx942";
    agent.product_name = "AMD Instinct MI300X";
    agent.device_index = 0;
    agent.gfx_version  = 942;

    EXPECT_EQ(agent.id, 1234);
    EXPECT_EQ(agent.name, "gfx942");
    EXPECT_EQ(agent.product_name, "AMD Instinct MI300X");
    EXPECT_EQ(agent.device_index, 0);
    EXPECT_EQ(agent.gfx_version, 942);
}

/**
 * @brief Tests for DispatchInfo struct
 */
class DispatchInfoTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DispatchInfoTest, DefaultConstruction)
{
    DispatchInfo dispatch;

    EXPECT_EQ(dispatch.dispatch_id, 0);
    EXPECT_EQ(dispatch.kernel_id, 0);
    EXPECT_EQ(dispatch.correlation_id, 0);
    EXPECT_EQ(dispatch.queue_id, 0);
    EXPECT_EQ(dispatch.agent_id, 0);
    EXPECT_EQ(dispatch.start_timestamp, 0);
    EXPECT_EQ(dispatch.end_timestamp, 0);
    EXPECT_TRUE(dispatch.kernel_name.empty());
}

TEST_F(DispatchInfoTest, InitializationWithValues)
{
    DispatchInfo dispatch;
    dispatch.dispatch_id     = 100;
    dispatch.kernel_id       = 200;
    dispatch.correlation_id  = 300;
    dispatch.queue_id        = 400;
    dispatch.agent_id        = 500;
    dispatch.start_timestamp = 1000000;
    dispatch.end_timestamp   = 2000000;
    dispatch.kernel_name     = "vectorAdd";

    EXPECT_EQ(dispatch.dispatch_id, 100);
    EXPECT_EQ(dispatch.kernel_id, 200);
    EXPECT_EQ(dispatch.correlation_id, 300);
    EXPECT_EQ(dispatch.queue_id, 400);
    EXPECT_EQ(dispatch.agent_id, 500);
    EXPECT_EQ(dispatch.start_timestamp, 1000000);
    EXPECT_EQ(dispatch.end_timestamp, 2000000);
    EXPECT_EQ(dispatch.kernel_name, "vectorAdd");
}

/**
 * @brief Tests for CounterRecord struct
 */
class CounterRecordTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CounterRecordTest, DefaultConstruction)
{
    CounterRecord record;

    EXPECT_EQ(record.dispatch_id, 0);
    EXPECT_EQ(record.counter_id, 0);
    EXPECT_TRUE(record.counter_name.empty());
    EXPECT_TRUE(record.kernel_name.empty());
    EXPECT_DOUBLE_EQ(record.value, 0.0);
    EXPECT_EQ(record.agent_id, 0);
    EXPECT_TRUE(record.dimensions.empty());
}

TEST_F(CounterRecordTest, InitializationWithValues)
{
    CounterRecord record;
    record.dispatch_id  = 1;
    record.counter_id   = 42;
    record.counter_name = "SQ_WAVES";
    record.kernel_name  = "matmul";
    record.value        = 12345.678;
    record.agent_id     = 100;
    record.dimensions.emplace_back("shader_engine", 0);
    record.dimensions.emplace_back("shader_array", 1);

    EXPECT_EQ(record.dispatch_id, 1);
    EXPECT_EQ(record.counter_id, 42);
    EXPECT_EQ(record.counter_name, "SQ_WAVES");
    EXPECT_EQ(record.kernel_name, "matmul");
    EXPECT_DOUBLE_EQ(record.value, 12345.678);
    EXPECT_EQ(record.agent_id, 100);
    EXPECT_EQ(record.dimensions.size(), 2);
    EXPECT_EQ(record.dimensions[0].first, "shader_engine");
    EXPECT_EQ(record.dimensions[0].second, 0);
    EXPECT_EQ(record.dimensions[1].first, "shader_array");
    EXPECT_EQ(record.dimensions[1].second, 1);
}

TEST_F(CounterRecordTest, CopyConstruction)
{
    CounterRecord record;
    record.dispatch_id  = 10;
    record.counter_id   = 20;
    record.counter_name = "TCC_HIT";
    record.kernel_name  = "conv2d";
    record.value        = 999.5;
    record.agent_id     = 200;
    record.dimensions.emplace_back("channel", 3);

    CounterRecord copy = record;

    EXPECT_EQ(copy.dispatch_id, record.dispatch_id);
    EXPECT_EQ(copy.counter_id, record.counter_id);
    EXPECT_EQ(copy.counter_name, record.counter_name);
    EXPECT_EQ(copy.kernel_name, record.kernel_name);
    EXPECT_DOUBLE_EQ(copy.value, record.value);
    EXPECT_EQ(copy.agent_id, record.agent_id);
    EXPECT_EQ(copy.dimensions.size(), record.dimensions.size());
}

}  // namespace test
}  // namespace python
}  // namespace rocprofiler
