#include <gtest/gtest.h>
#include "../trace_config.h"

namespace pm4_builder {

class TraceConfigTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Setup default configuration
    config.sampleRate = 1000;
    config.spm_se_number_total = 4;  // Typical number of shader engines
    config.se_mask = 0x0F;  // All 4 SEs enabled
    config.capacity_per_se = 0x2000;
    config.capacity_per_disabled_se = 0x1000;
  }

  TraceConfig config;
};

TEST_F(TraceConfigTest, DefaultValues) {
  TraceConfig default_config;
  
  // Check default initialization values
  EXPECT_EQ(default_config.targetCu, 0);
  EXPECT_EQ(default_config.vmIdMask, 0);
  EXPECT_EQ(default_config.simd_sel, 0xF);
  EXPECT_EQ(default_config.sampleRate, 625);
  EXPECT_EQ(default_config.perfMASK, ~0u);
  EXPECT_TRUE(default_config.spm_sq_32bit_mode);
  EXPECT_TRUE(default_config.spm_kfd_mode);
  EXPECT_FALSE(default_config.mi100);
  EXPECT_EQ(default_config.se_mask, 0x11);
}

TEST_F(TraceConfigTest, SEConfiguration) {
  // Configure SE target CUs and base addresses
  config.target_cu_per_se[0] = 2;   // SE0: CU2
  config.target_cu_per_se[1] = -1;  // SE1: disabled
  config.target_cu_per_se[2] = 4;   // SE2: CU4
  config.target_cu_per_se[3] = 1;   // SE3: CU1

  config.se_base_addresses[0] = 0x1000;
  config.se_base_addresses[1] = 0x2000;
  config.se_base_addresses[2] = 0x3000;
  config.se_base_addresses[3] = 0x4000;

  // Test target CU retrieval
  EXPECT_EQ(config.GetTargetCU(0), 2);
  EXPECT_EQ(config.GetTargetCU(1), -1);
  EXPECT_EQ(config.GetTargetCU(2), 4);
  EXPECT_EQ(config.GetTargetCU(3), 1);

  // Test SE base address retrieval
  EXPECT_EQ(config.GetSEBaseAddr(0), 0x1000);
  EXPECT_EQ(config.GetSEBaseAddr(1), 0x2000);
  EXPECT_EQ(config.GetSEBaseAddr(2), 0x3000);
  EXPECT_EQ(config.GetSEBaseAddr(3), 0x4000);

  // Test SE capacity calculations
  EXPECT_EQ(config.GetCapacity(0), config.capacity_per_se);        // Enabled SE
  EXPECT_EQ(config.GetCapacity(1), config.capacity_per_disabled_se); // Disabled SE
  EXPECT_EQ(config.GetCapacity(2), config.capacity_per_se);        // Enabled SE
  EXPECT_EQ(config.GetCapacity(3), config.capacity_per_se);        // Enabled SE
}

TEST_F(TraceConfigTest, SEMaskConfiguration) {
  // Test different SE mask configurations
  config.se_mask = 0x5;  // Enable SE0 and SE2, disable SE1 and SE3

  // Setup target CUs
  config.target_cu_per_se[0] = 0;   // SE0 enabled
  config.target_cu_per_se[1] = -1;  // SE1 disabled
  config.target_cu_per_se[2] = 1;   // SE2 enabled
  config.target_cu_per_se[3] = -1;  // SE3 disabled

  EXPECT_EQ(config.GetSEmask(), 0x5);
  EXPECT_EQ(config.GetTargetCU(0), 0);
  EXPECT_EQ(config.GetTargetCU(1), -1);
  EXPECT_EQ(config.GetTargetCU(2), 1);
  EXPECT_EQ(config.GetTargetCU(3), -1);
}

TEST_F(TraceConfigTest, BufferConfiguration) {
  const size_t BUFFER_SIZE = 4096;
  char data_buffer[BUFFER_SIZE];
  char control_buffer[BUFFER_SIZE];

  // Configure buffers
  config.data_buffer_ptr = data_buffer;
  config.data_buffer_size = BUFFER_SIZE;
  config.control_buffer_ptr = control_buffer;
  config.control_buffer_size = BUFFER_SIZE;

  EXPECT_EQ(config.data_buffer_ptr, data_buffer);
  EXPECT_EQ(config.data_buffer_size, BUFFER_SIZE);
  EXPECT_EQ(config.control_buffer_ptr, control_buffer);
  EXPECT_EQ(config.control_buffer_size, BUFFER_SIZE);
}

TEST_F(TraceConfigTest, PerformanceConfiguration) {
  // Test performance counter configuration
  config.perfMASK = 0xF0F0;
  config.perfCTRL = 0x1234;
  
  // Add some performance counters
  config.perfcounters.push_back({0, 1});  // Counter 0, Instance 1
  config.perfcounters.push_back({2, 3});  // Counter 2, Instance 3

  EXPECT_EQ(config.perfMASK, 0xF0F0);
  EXPECT_EQ(config.perfCTRL, 0x1234);
  ASSERT_EQ(config.perfcounters.size(), 2);
  EXPECT_EQ(config.perfcounters[0].first, 0);
  EXPECT_EQ(config.perfcounters[0].second, 1);
  EXPECT_EQ(config.perfcounters[1].first, 2);
  EXPECT_EQ(config.perfcounters[1].second, 3);
}

TEST_F(TraceConfigTest, ConcurrentConfiguration) {
  // Test concurrent kernel configuration
  config.concurrent = 2;
  config.spm_kfd_mode = true;
  config.mi100 = true;

  // Configure per-SE capacities for concurrent mode
  config.capacity_per_se = 0x4000;
  config.capacity_per_disabled_se = 0x2000;

  // Setup multiple SEs with different target CUs
  for (uint32_t se = 0; se < config.spm_se_number_total; se++) {
    config.target_cu_per_se[se] = se % 2 ? -1 : se;  // Alternate between enabled/disabled
    config.se_base_addresses[se] = 0x1000 * (se + 1);
  }

  EXPECT_EQ(config.concurrent, 2);
  EXPECT_TRUE(config.spm_kfd_mode);
  EXPECT_TRUE(config.mi100);

  // Verify SE configuration in concurrent mode
  for (uint32_t se = 0; se < config.spm_se_number_total; se++) {
    if (se % 2 == 0) {
      EXPECT_EQ(config.GetTargetCU(se), se);
      EXPECT_EQ(config.GetCapacity(se), config.capacity_per_se);
    } else {
      EXPECT_EQ(config.GetTargetCU(se), -1);
      EXPECT_EQ(config.GetCapacity(se), config.capacity_per_disabled_se);
    }
    EXPECT_EQ(config.GetSEBaseAddr(se), 0x1000 * (se + 1));
  }
}

TEST_F(TraceConfigTest, ExceptionHandling) {
  // Test accessing non-existent SE configurations
  EXPECT_THROW(config.GetTargetCU(99), std::out_of_range);
  EXPECT_THROW(config.GetSEBaseAddr(99), std::out_of_range);
}

} // namespace pm4_builder
