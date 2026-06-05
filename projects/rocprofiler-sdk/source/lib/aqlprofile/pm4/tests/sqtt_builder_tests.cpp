// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "lib/aqlprofile/pm4/trace_config.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace pm4_builder
{
// Minimal implementation of required types for testing
struct AgentInfo
{
    uint32_t gfxip;
    uint32_t xcc_num;
    uint32_t se_num;
};

enum hsa_status_t
{
    HSA_STATUS_SUCCESS = 0x0,
};

// Minimal primitives for testing
struct TestPrimitives
{
    static constexpr uint32_t GFXIP_LEVEL                = 9;
    static constexpr uint32_t TT_BUFF_ALIGN_SHIFT        = 12;  // 4KB alignment
    static constexpr uint32_t TT_CONTROL_UTC_ERR_MASK    = 0x1;
    static constexpr uint32_t TT_CONTROL_FULL_MASK       = 0x2;
    static constexpr uint32_t TT_WRITE_PTR_MASK          = 0x4;
    static constexpr uint32_t SQ_THREAD_TRACE_USERDATA_2 = 0x1000;

    static uint32_t grbm_broadcast_value() { return 0xFFFFFFFF; }
    static uint32_t sqtt_mode_off_value() { return 0; }
    static uint32_t sqtt_mode_on_value() { return 1; }
    static uint32_t sqtt_buffer_size_value(uint64_t size, uint32_t)
    {
        return static_cast<uint32_t>(size >> TT_BUFF_ALIGN_SHIFT);
    }
};

// Minimal command buffer for testing
class CmdBuffer
{
public:
    void                  Clear() {}
    size_t                DwSize() const { return 0; }
    const void*           Data() const { return nullptr; }
    void                  Assign(size_t, uint32_t) {}
    std::vector<uint32_t> commands;
};

// Minimal command builder for testing
class TestBuilder
{
public:
    TestBuilder(const AgentInfo*) {}

    void BuildWriteUConfigRegPacket(CmdBuffer* cmd_buffer, uint32_t addr, uint32_t value)
    {
        cmd_buffer->commands.push_back(addr);
        cmd_buffer->commands.push_back(value);
    }

    void BuildPredExecPacket(CmdBuffer*, uint32_t, uint32_t) {}
    void BuildWriteWaitIdlePacket(CmdBuffer*) {}
    void BuildCacheFlushPacket(CmdBuffer*, size_t, size_t) {}
};

// Actual GpuSqttBuilder implementation for testing
template <typename Builder, typename Primitives>
class GpuSqttBuilder
{
public:
    explicit GpuSqttBuilder(const AgentInfo* agent_info)
    : xcc_number_(agent_info->xcc_num)
    , se_number_total(agent_info->se_num)
    , builder_(agent_info)
    {}

    size_t   GetUTCErrorMask() const { return Primitives::TT_CONTROL_UTC_ERR_MASK; }
    size_t   GetBufferFullMask() const { return Primitives::TT_CONTROL_FULL_MASK; }
    size_t   GetWritePtrMask() const { return Primitives::TT_WRITE_PTR_MASK; }
    size_t   GetWritePtrBlk() const { return 32; }
    size_t   BufferAlignment() const { return Primitives::TT_BUFF_ALIGN_SHIFT; }
    uint32_t GetXCCNumber() const { return xcc_number_; }

    uint64_t PopCount(uint64_t se_mask) const
    {
        uint64_t num_enabled = 0;
        while(se_mask != 0u)
        {
            num_enabled += se_mask & 1;
            se_mask >>= 1;
        }
        return std::max<uint64_t>(num_enabled, 1u);
    }

    uint64_t GetBaseStep(uint64_t buffersize, uint64_t se_mask) const
    {
        uint64_t num_enabled   = PopCount(se_mask);
        int64_t  num_disabled  = (64 - num_enabled) << Primitives::TT_BUFF_ALIGN_SHIFT;
        int64_t  buffer_per_se = std::max<int64_t>(0, buffersize - num_disabled) / num_enabled;
        return uint64_t(buffer_per_se) & ~((1ULL << Primitives::TT_BUFF_ALIGN_SHIFT) - 1);
    }

private:
    uint32_t xcc_number_;
    size_t   se_number_total;
    Builder  builder_;
};

class SqttBuilderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        agent_info.gfxip   = 9;
        agent_info.xcc_num = 2;
        agent_info.se_num  = 4;
    }

    AgentInfo            agent_info;
    std::vector<uint8_t> data_buffer;
    std::vector<uint8_t> control_buffer;
};

TEST_F(SqttBuilderTest, BufferStepCalculation)
{
    GpuSqttBuilder<TestBuilder, TestPrimitives> builder(&agent_info);

    constexpr uint64_t kAlign = 1ULL << TestPrimitives::TT_BUFF_ALIGN_SHIFT;  // 4KB

    // GetBaseStep reserves one aligned block for every disabled SE (out of the 64 possible
    // SEs), divides the remaining space evenly across the enabled SEs, and floors the
    // per-SE step down to the buffer alignment. Verify those invariants instead of assuming
    // the whole buffer is split evenly with no reservation.
    auto verify_step = [&](uint64_t total_buffer, uint64_t se_mask) {
        const uint64_t enabled  = builder.PopCount(se_mask);
        const uint64_t reserved = (64 - enabled) * kAlign;
        const uint64_t step     = builder.GetBaseStep(total_buffer, se_mask);

        // step must be aligned to the buffer alignment
        EXPECT_EQ(step % kAlign, 0u);
        // enabled SEs plus the space reserved for disabled SEs must fit within the buffer
        EXPECT_LE(step * enabled + reserved, total_buffer);
        // the step must be maximal: it cannot grow by another aligned block and still fit
        EXPECT_GT((step + kAlign) * enabled + reserved, total_buffer);
    };

    const uint64_t total_buffer = 1024 * 1024;  // 1MB total

    verify_step(total_buffer, 0xF);  // all 4 SEs enabled
    verify_step(total_buffer, 0x3);  // 2 SEs enabled
    verify_step(total_buffer, 0x1);  // single SE enabled
}

TEST_F(SqttBuilderTest, PopulationCount)
{
    GpuSqttBuilder<TestBuilder, TestPrimitives> builder(&agent_info);

    // Test different SE mask configurations
    EXPECT_EQ(builder.PopCount(0x1), 1);  // Single SE
    EXPECT_EQ(builder.PopCount(0x3), 2);  // Two SEs
    EXPECT_EQ(builder.PopCount(0xF), 4);  // Four SEs
    EXPECT_EQ(builder.PopCount(0x0), 1);  // No SEs (minimum is 1)
    EXPECT_EQ(builder.PopCount(0x5), 2);  // Non-contiguous SEs
}

TEST_F(SqttBuilderTest, ThreadTraceStatusMasks)
{
    GpuSqttBuilder<TestBuilder, TestPrimitives> builder(&agent_info);

    // Verify mask values
    EXPECT_EQ(builder.GetUTCErrorMask(), TestPrimitives::TT_CONTROL_UTC_ERR_MASK);
    EXPECT_EQ(builder.GetBufferFullMask(), TestPrimitives::TT_CONTROL_FULL_MASK);
    EXPECT_EQ(builder.GetWritePtrMask(), TestPrimitives::TT_WRITE_PTR_MASK);

    // Verify masks are unique
    EXPECT_NE(builder.GetUTCErrorMask(), builder.GetBufferFullMask());
    EXPECT_NE(builder.GetUTCErrorMask(), builder.GetWritePtrMask());
    EXPECT_NE(builder.GetBufferFullMask(), builder.GetWritePtrMask());
}

TEST_F(SqttBuilderTest, XCCConfiguration)
{
    GpuSqttBuilder<TestBuilder, TestPrimitives> builder(&agent_info);

    // Test XCC number configuration
    EXPECT_EQ(builder.GetXCCNumber(), agent_info.xcc_num);

    // Test with different XCC configurations
    agent_info.xcc_num = 1;
    GpuSqttBuilder<TestBuilder, TestPrimitives> single_xcc(&agent_info);
    EXPECT_EQ(single_xcc.GetXCCNumber(), 1);

    agent_info.xcc_num = 4;
    GpuSqttBuilder<TestBuilder, TestPrimitives> multi_xcc(&agent_info);
    EXPECT_EQ(multi_xcc.GetXCCNumber(), 4);
}

TEST_F(SqttBuilderTest, BufferAlignmentAndBlockSize)
{
    GpuSqttBuilder<TestBuilder, TestPrimitives> builder(&agent_info);

    // Test buffer alignment
    EXPECT_EQ(builder.BufferAlignment(), TestPrimitives::TT_BUFF_ALIGN_SHIFT);
    EXPECT_EQ(1ULL << builder.BufferAlignment(), 4096);  // 4KB alignment

    // Test write pointer block size
    EXPECT_EQ(builder.GetWritePtrBlk(), 32);  // 32-byte blocks
}

}  // namespace pm4_builder
