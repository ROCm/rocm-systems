#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

// Minimal test environment
struct AgentInfo {
    uint32_t se_num;
    uint32_t xcc_num;
    uint32_t shader_arrays_per_se;
    uint32_t cu_num;
};

class CmdBuffer {
public:
    virtual ~CmdBuffer() = default;
    virtual void Append(const void* data, size_t size) = 0;
    virtual size_t Size() const = 0;
    virtual const void* Data() const = 0;
    virtual void Clear() = 0;
};

// Simple test command buffer
class TestCmdBuffer : public CmdBuffer {
public:
    void Append(const void* data, size_t size) override { commands++; }
    size_t Size() const override { return commands; }
    const void* Data() const override { return nullptr; }
    void Clear() override { commands = 0; }
    
    size_t commands = 0;
};

// Simple test PMC builder
class PmcBuilder {
public:
    virtual ~PmcBuilder() = default;
    
    void Enable(CmdBuffer* cmd_buffer) {
        if (cmd_buffer) {
            cmd_buffer->Append(nullptr, sizeof(uint32_t));
        }
    }
    
    void Disable(CmdBuffer* cmd_buffer) {
        if (cmd_buffer) {
            cmd_buffer->Append(nullptr, sizeof(uint32_t));
        }
    }
    
    int GetNumWGPs(const AgentInfo& info) {
        if (info.se_num == 0 || info.shader_arrays_per_se == 0) return 0;
        return (info.cu_num / 2) / (info.se_num * info.shader_arrays_per_se);
    }
};
// Test cases
TEST(PmcBuilderTest, BasicOperations) {
    TestCmdBuffer cmd_buffer;
    PmcBuilder builder;
    
    // Test Enable
    builder.Enable(&cmd_buffer);
    EXPECT_EQ(cmd_buffer.commands, 1);
    
    // Test Disable
    builder.Disable(&cmd_buffer);
    EXPECT_EQ(cmd_buffer.commands, 2);
}

TEST(PmcBuilderTest, WGPCalculation) {
    PmcBuilder builder;
    AgentInfo info;
    
    // Test edge case - zero CUs
    info.cu_num = 0;
    EXPECT_EQ(builder.GetNumWGPs(info), 0);
    
    // Test edge case - zero shader arrays
    info.cu_num = 64;
    info.shader_arrays_per_se = 0;
    EXPECT_EQ(builder.GetNumWGPs(info), 0);
}

TEST(PmcBuilderTest, CommandBufferOperations) {
    TestCmdBuffer cmd_buffer;
    
    // Test append
    cmd_buffer.Append(nullptr, sizeof(uint32_t));
    EXPECT_EQ(cmd_buffer.commands, 1);
    
    // Test clear
    cmd_buffer.Clear();
    EXPECT_EQ(cmd_buffer.commands, 0);
    
    // Test size
    cmd_buffer.Append(nullptr, sizeof(uint32_t));
    EXPECT_EQ(cmd_buffer.Size(), 1);
    
    // Test data
    EXPECT_EQ(cmd_buffer.Data(), nullptr);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}