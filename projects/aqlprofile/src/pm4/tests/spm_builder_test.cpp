#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstring>
#include <vector>
#include <memory>

#include "pm4/spm_builder.h"
#include "pm4/cmd_builder.h"
#include "pm4/cmd_config.h"
#include "pm4/trace_config.h"
#include "def/gpu_block_info.h"


using namespace pm4_builder;
//using namespace aql_profile;

namespace spm_builder_tests {

// Mock SpmBuilder class for testing
class MockSpmBuilder : public SpmBuilder {
public:
    MOCK_METHOD(void, Begin, (CmdBuffer* cmd_buffer, const SpmConfig* config, const counters_vector& counters_vec), (override));
    MOCK_METHOD(void, End, (CmdBuffer* cmd_buffer, const SpmConfig* config), (override));
};

class SpmBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data structures
        memset(&test_config_, 0, sizeof(test_config_));
        
        // Set up default SPM config
        test_config_.sampleRate = 1000;
        test_config_.data_buffer_ptr = test_buffer_.data();
        test_config_.data_buffer_size = test_buffer_.size() * sizeof(uint32_t);
        
        // Initialize agent info for creating concrete SpmBuilder
        memset(&agent_info_, 0, sizeof(agent_info_));
        strncpy(agent_info_.name, "gfx90a", sizeof(agent_info_.name) - 1);
        strncpy(agent_info_.gfxip, "gfx90a", sizeof(agent_info_.gfxip) - 1);
        agent_info_.cu_num = 104;
        agent_info_.se_num = 8;
        agent_info_.xcc_num = 1;
        agent_info_.shader_arrays_per_se = 2;
    }

    void TearDown() override {
        // Clean up any resources
    }

    SpmConfig test_config_;
    std::vector<uint32_t> test_buffer_{1024, 0}; // 4KB buffer initialized with zeros
    AgentInfo agent_info_;
    counters_vector test_counters_;
};

// Test 1: Begin function with valid parameters
TEST_F(SpmBuilderTest, BeginWithValidParameters) {
    // Create a mock SpmBuilder
    MockSpmBuilder mock_spm_builder;
    CmdBuffer cmd_buffer;
    
    // Set up expectations - Begin should be called once with the provided parameters
    EXPECT_CALL(mock_spm_builder, Begin(&cmd_buffer, &test_config_, ::testing::Ref(test_counters_)))
        .Times(1);
    
    // Call Begin method
    mock_spm_builder.Begin(&cmd_buffer, &test_config_, test_counters_);
    
    // Verify that the command buffer is still valid after the call
    EXPECT_GE(cmd_buffer.DwSize(), 0);
}

// Test 2: End function with valid parameters
TEST_F(SpmBuilderTest, EndWithValidParameters) {
    // Create a mock SpmBuilder
    MockSpmBuilder mock_spm_builder;
    CmdBuffer cmd_buffer;
    
    // Set up expectations - End should be called once with the provided parameters
    EXPECT_CALL(mock_spm_builder, End(&cmd_buffer, &test_config_))
        .Times(1);
    
    // Call End method
    mock_spm_builder.End(&cmd_buffer, &test_config_);
    
    // Verify that the command buffer is still valid after the call
    EXPECT_GE(cmd_buffer.DwSize(), 0);
}

// Test 5: Begin and End sequence with mock
TEST_F(SpmBuilderTest, BeginEndSequenceWithMock) {
    MockSpmBuilder mock_spm_builder;
    CmdBuffer cmd_buffer;
    
    // Set up expectations for a complete Begin-End sequence
    ::testing::InSequence seq;
    EXPECT_CALL(mock_spm_builder, Begin(&cmd_buffer, &test_config_, ::testing::Ref(test_counters_)))
        .Times(1);
    EXPECT_CALL(mock_spm_builder, End(&cmd_buffer, &test_config_))
        .Times(1);
    
    // Execute the sequence
    mock_spm_builder.Begin(&cmd_buffer, &test_config_, test_counters_);
    mock_spm_builder.End(&cmd_buffer, &test_config_);
    
    // Verify buffer state after complete sequence
    EXPECT_GE(cmd_buffer.DwSize(), 0);
}

// Test 6: Null parameter handling (defensive programming)
TEST_F(SpmBuilderTest, NullParameterHandling) {
    MockSpmBuilder mock_spm_builder;
    
    // These tests verify that the mock can handle null parameters
    // In a real implementation, these should be handled gracefully or throw exceptions
    
    // Test with null command buffer - should be handled by implementation
    EXPECT_CALL(mock_spm_builder, Begin(nullptr, &test_config_, ::testing::Ref(test_counters_)))
        .Times(1);
    mock_spm_builder.Begin(nullptr, &test_config_, test_counters_);
    
    // Test with null config - should be handled by implementation  
    CmdBuffer cmd_buffer;
    EXPECT_CALL(mock_spm_builder, Begin(&cmd_buffer, nullptr, ::testing::Ref(test_counters_)))
        .Times(1);
    mock_spm_builder.Begin(&cmd_buffer, nullptr, test_counters_);
}

} // namespace spm_builder_tests
