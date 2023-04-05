#include <gtest/gtest.h>
#include "src/core/hardware/hsa_info.h"
//#include "src/core/hsa/hsa_common.h"

// Entry Point for Gtests Infra

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  // Add line below to disable any problematic test
  testing::GTEST_FLAG(filter) =
      "-OpenMPTest.*:ProfilerSPMTest*:ProfilerMQTest*:ProfilerMPTest*:MPITest*";
  // Disable ATT test fir gfx10 GPUs until its supported
  hsa_init();
  // iterate for gpu's
  hsa_iterate_agents(
      [](hsa_agent_t agent, void*) {
        char gpu_name[64];
        hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, gpu_name);
        std::string gfx_name = gpu_name;
        if (gfx_name.find("gfx10") != std::string::npos) {
          testing::GTEST_FLAG(filter) =
              "-ATTCollection.*:OpenMPTest.*:-ProfilerSPMTest*:ProfilerMQTest:*ProfilerMPTest*:"
              "MPITest*";
        }
        return HSA_STATUS_SUCCESS;
      },
      nullptr);
  // hsa_shut_down(); // Waiting for hsa_shutdown bug to fix
  // Append filter above to disable any problematic test
  return RUN_ALL_TESTS();
}
