#include <gtest/gtest.h>
#include "src/core/hardware/hsa_info.h"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  // Add line below to disable any problematic test
  hsa_init();
  testing::GTEST_FLAG(filter) =
      "-OpenMPTest.*:ProfilerSPMTest.*:ProfilerMQTest.*:ProfilerMPTest.*:MPITest.*";
  // Disable ATT test fir gfx10 GPUs until its supported
  // iterate for gpu's
  hsa_iterate_agents(
      [](hsa_agent_t agent, void*) {
        char gpu_name[64];
        hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, gpu_name);
        std::string gfx_name = gpu_name;
        if (gfx_name.find("gfx10") != std::string::npos) {
          testing::GTEST_FLAG(filter) =
              "-ATTCollection.*:OpenMPTest.*:ProfilerSPMTest*:ProfilerMQTest.*:*ProfilerMPTest.*:"
              "MPITest.*";
        }
        return HSA_STATUS_SUCCESS;
      },
    nullptr);
// Append filter above to disable any problematic test
  int res = RUN_ALL_TESTS();
  hsa_shut_down();
  return res;
}
