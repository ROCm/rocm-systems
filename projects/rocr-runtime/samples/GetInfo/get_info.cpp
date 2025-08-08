#include "get_info.h"

#include <iostream>

GetInfo::GetInfo() : HsaTest("HSA Info") {}

GetInfo::~GetInfo() {}

void GetInfo::Run() {
  std::cout << std::endl;
  std::cout << "Num CPUs in platform: " << cpus_.size() << std::endl;
  std::cout << "------------------------------------------------\n";

  for (size_t i = 0; i < cpus_.size(); ++i) {
    hsa_agent_t cpu = cpus_[i];
    std::cout << "CPU[" << i << "] properties:" << std::endl;
    std::cout << "------------------------------------------------\n";
    AgentProps prop(cpu);
    PrintAgentInfo(prop);
    PrintPeers(cpu);
    std::cout << "------------------------------------------------\n";

    hsa_amd_memory_pool_t global_fine = global_fine_[cpu.handle];
    if (global_fine.handle != 0) {
      std::cout << "CPU[" << i << "] system fine grain pool properties:\n";
      std::cout << "------------------------------------------------\n";
      PoolProps prop(global_fine);
      PrintPoolInfo(prop);
      std::cout << "------------------------------------------------\n";
    }

    hsa_amd_memory_pool_t global_coarse = global_coarse_[cpu.handle];
    if (global_coarse.handle != 0) {
      std::cout << "CPU[" << i << "] system coarse grain pool properties:\n";
      std::cout << "------------------------------------------------\n";
      PoolProps prop(global_coarse);
      PrintPoolInfo(prop);
      std::cout << "------------------------------------------------\n";
    }
  }

  std::cout << std::endl;
  std::cout << "Num GPUs in platform: " << gpus_.size() << std::endl;
  std::cout << "------------------------------------------------\n";

  for (size_t i = 0; i < gpus_.size(); ++i) {
    hsa_agent_t gpu = gpus_[i];
    std::cout << "GPU[" << i << "] properties:" << std::endl;
    std::cout << "------------------------------------------------\n";
    AgentProps prop(gpu);
    PrintAgentInfo(prop);
    PrintPeers(gpu);
    std::cout << "------------------------------------------------\n";

    hsa_amd_memory_pool_t global_coarse = global_coarse_[gpu.handle];
    if (global_coarse.handle != 0) {
      std::cout << "GPU[" << i << "] local memory pool properties:\n";
      std::cout << "------------------------------------------------\n";
      PoolProps prop(global_coarse);
      PrintPoolInfo(prop);
      std::cout << "------------------------------------------------\n";
    }

    hsa_amd_memory_pool_t group = group_[gpu.handle];
    if (group.handle != 0) {
      std::cout << "GPU[" << i << "] group memory pool properties:\n";
      std::cout << "------------------------------------------------\n";
      PoolProps prop(group);
      PrintPoolInfo(prop);
      std::cout << "------------------------------------------------\n";
    }
  }
}

int main(int argc, char* argv[]) {
  GetInfo get_info;

  get_info.Init();
  get_info.Run();
  get_info.Cleanup();

  return 0;
}