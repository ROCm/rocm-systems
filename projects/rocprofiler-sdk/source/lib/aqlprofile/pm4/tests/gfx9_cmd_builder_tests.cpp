// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstdint>

// Forward declarations and minimal implementations for testing
namespace pm4_builder
{
class CmdBuffer
{
public:
    virtual ~CmdBuffer()                                      = default;
    virtual void        Append(const void* data, size_t size) = 0;
    virtual size_t      Size() const                          = 0;
    virtual const void* Data() const                          = 0;
    virtual void        Clear()                               = 0;
};

// Minimal register abstraction
struct Register
{
    uint32_t addr;
    explicit Register(uint32_t a = 0)
    : addr(a)
    {}
    operator uint32_t() const { return addr; }
};

class CmdBuilder
{
public:
    explicit CmdBuilder(const void* table = nullptr) {}
    virtual ~CmdBuilder() = default;
    virtual uint32_t get_addr(Register reg) { return reg.addr; }
    bool             bUsePerfCounterMode = true;
};

class Gfx9CmdBuilder : public CmdBuilder
{
public:
    explicit Gfx9CmdBuilder(const void* table = nullptr)
    : CmdBuilder(table)
    {}
    template <typename Tp>
    static void BuildBarrierCommand(Tp* cmdBuf);
    template <typename Tp>
    static void BuildWriteWaitIdlePacket(Tp* cmdBuf);
    template <typename Tp>
    static void BuildWriteShRegPacket(Tp* cmdBuf, uint32_t addr, uint32_t value);
    template <typename Tp>
    static void BuildCacheFlushPacket(Tp* cmdBuf, size_t addr, size_t size);
    template <typename Tp>
    static void BuildNopPacket(Tp* cmdBuf, uint32_t num_dwords);
};

namespace
{
// Simple mock command buffer for testing
class TestCmdBuffer
{
public:
    TestCmdBuffer()  = default;
    ~TestCmdBuffer() = default;

    void Append(const void* data, size_t size)
    {
        const uint32_t* words      = static_cast<const uint32_t*>(data);
        size_t          word_count = size / sizeof(uint32_t);
        for(size_t i = 0; i < word_count; ++i)
        {
            commands.push_back(words[i]);
        }
    }

    size_t      Size() const { return commands.size() * sizeof(uint32_t); }
    const void* Data() const { return commands.data(); }
    void        Clear() { commands.clear(); }

    std::vector<uint32_t> commands = {};
};

class Gfx9CmdBuilderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cmd_buffer = std::make_unique<TestCmdBuffer>();
        builder    = std::make_unique<pm4_builder::Gfx9CmdBuilder>(nullptr);
    }

    void TearDown() override
    {
        cmd_buffer.reset();
        builder.reset();
    }

    // Helper to verify packet header
    void VerifyPacketHeader(uint32_t opcode, size_t packet_size_dwords)
    {
        ASSERT_FALSE(cmd_buffer->commands.empty());
        uint32_t header          = cmd_buffer->commands[0];
        uint32_t expected_count  = packet_size_dwords - 2;
        uint32_t expected_header = (3u << 30) | (opcode << 8) | expected_count;
        EXPECT_EQ(header, expected_header);
    }

    std::unique_ptr<TestCmdBuffer>               cmd_buffer = nullptr;
    std::unique_ptr<pm4_builder::Gfx9CmdBuilder> builder    = nullptr;
};

// Test barrier command generation
TEST_F(Gfx9CmdBuilderTest, BarrierCommand)
{
    builder->BuildBarrierCommand(cmd_buffer.get());

    ASSERT_EQ(cmd_buffer->commands.size(), 2u);
    VerifyPacketHeader(0x14, 2);                     // EVENT_WRITE opcode
    EXPECT_EQ(cmd_buffer->commands[1] & 0x3f, 0x4);  // CS_PARTIAL_FLUSH event type
}

// Test wait idle packet generation
TEST_F(Gfx9CmdBuilderTest, WaitIdlePacket)
{
    builder->BuildWriteWaitIdlePacket(cmd_buffer.get());

    ASSERT_EQ(cmd_buffer->commands.size(), 2u);
    VerifyPacketHeader(0x14, 2);  // EVENT_WRITE opcode
}

// Test write to shader register
TEST_F(Gfx9CmdBuilderTest, WriteShRegPacket)
{
    const uint32_t test_addr  = 0x2000;
    const uint32_t test_value = 0x12345678;

    builder->BuildWriteShRegPacket(cmd_buffer.get(), test_addr, test_value);

    ASSERT_EQ(cmd_buffer->commands.size(), 3u);
    VerifyPacketHeader(0x4, 3);  // SET_SH_REG opcode
    EXPECT_EQ(cmd_buffer->commands[2], test_value);
}

// Test cache flush packet generation
TEST_F(Gfx9CmdBuilderTest, CacheFlushPacket)
{
    const size_t test_addr = 0x1000;
    const size_t test_size = 0x100;

    builder->BuildCacheFlushPacket(cmd_buffer.get(), test_addr, test_size);

    ASSERT_EQ(cmd_buffer->commands.size(), 7u);
    VerifyPacketHeader(0x49, 7);  // ACQUIRE_MEM opcode
}

// Test NOP packet generation
TEST_F(Gfx9CmdBuilderTest, NopPacket)
{
    const uint32_t num_dwords = 3;

    builder->BuildNopPacket(cmd_buffer.get(), num_dwords);

    ASSERT_EQ(cmd_buffer->commands.size(), num_dwords);
    VerifyPacketHeader(0x10, num_dwords);  // NOP opcode

    // Verify remaining dwords are zeros
    for(uint32_t i = 1; i < num_dwords; ++i)
    {
        EXPECT_EQ(cmd_buffer->commands[i], 0u);
    }
}

}  // namespace

// Implementations for testing
template <typename Tp>
void
pm4_builder::Gfx9CmdBuilder::BuildBarrierCommand(Tp* cmdBuf)
{
    uint32_t packet[2] = {
        (3u << 30) | (0x14u << 8) | 0u,  // header: type3, EVENT_WRITE, count=0
        0x4u                             // CS_PARTIAL_FLUSH
    };
    cmdBuf->Append(packet, sizeof(packet));
}

template <typename Tp>
void
pm4_builder::Gfx9CmdBuilder::BuildWriteWaitIdlePacket(Tp* cmdBuf)
{
    BuildBarrierCommand(cmdBuf);
}

template <typename Tp>
void
pm4_builder::Gfx9CmdBuilder::BuildWriteShRegPacket(Tp* cmdBuf, uint32_t addr, uint32_t value)
{
    uint32_t packet[3] = {
        (3u << 30) | (0x4u << 8) | 1u,  // header: type3, SET_SH_REG, count=1
        addr,                           // register address
        value                           // value to write
    };
    cmdBuf->Append(packet, sizeof(packet));
}

template <typename Tp>
void
pm4_builder::Gfx9CmdBuilder::BuildCacheFlushPacket(Tp* cmdBuf, size_t addr, size_t size)
{
    uint32_t packet[7] = {
        (3u << 30) | (0x49u << 8) | 5u,  // header: type3, ACQUIRE_MEM, count=5
        0,                               // control
        uint32_t(size >> 8),             // size low
        uint32_t(size >> 40),            // size high
        uint32_t(addr >> 8),             // addr low
        uint32_t(addr >> 40),            // addr high
        0x10                             // poll interval
    };
    cmdBuf->Append(packet, sizeof(packet));
}

template <typename Tp>
void
pm4_builder::Gfx9CmdBuilder::BuildNopPacket(Tp* cmdBuf, uint32_t num_dwords)
{
    uint32_t header = (3u << 30) | (0x10u << 8) | (num_dwords - 2u);  // type3, NOP
    cmdBuf->Append(&header, sizeof(header));

    std::vector<uint32_t> nops(num_dwords - 1, 0);
    if(num_dwords > 1)
    {
        cmdBuf->Append(nops.data(), nops.size() * sizeof(uint32_t));
    }
}
}  // namespace pm4_builder
