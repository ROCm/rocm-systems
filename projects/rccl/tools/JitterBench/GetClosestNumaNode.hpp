/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hsa/hsa_ext_amd.h>

// Helper macro for checking HSA calls
#define HSA_CHECK(cmd)                                                  \
  do {                                                                  \
    hsa_status_t error = (cmd);                                         \
    if (error != HSA_STATUS_SUCCESS) {                                  \
      const char* errString = NULL;                                     \
      hsa_status_string(error, &errString);                             \
      std::cerr << "Encountered HSA error (" << errString << ") at line " \
                << __LINE__ << " in file " << __FILE__ << "\n";         \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

// Structure to hold HSA agent information
#if !defined(__NVCC__)
struct AgentData
{
  bool isInitialized;
  std::vector<hsa_agent_t> cpuAgents;
  std::vector<hsa_agent_t> gpuAgents;
  std::vector<int> closestNumaNode;
};

// Simple callback function to return any memory pool for an agent
hsa_status_t MemPoolInfoCallback(hsa_amd_memory_pool_t pool, void *data)
{
  hsa_amd_memory_pool_t* poolData = reinterpret_cast<hsa_amd_memory_pool_t*>(data);

  // Check memory pool flags
  uint32_t poolFlags;
  HSA_CHECK(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &poolFlags));

  // Only consider coarse-grained pools
  if (!(poolFlags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) return HSA_STATUS_SUCCESS;

  *poolData = pool;
  return HSA_STATUS_SUCCESS;
}

// Callback function to gather HSA agent information
hsa_status_t AgentInfoCallback(hsa_agent_t agent, void* data)
{
  AgentData* agentData = reinterpret_cast<AgentData*>(data);

  // Get the device type
  hsa_device_type_t deviceType;
  HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &deviceType));
  if (deviceType == HSA_DEVICE_TYPE_CPU)
    agentData->cpuAgents.push_back(agent);
  if (deviceType == HSA_DEVICE_TYPE_GPU)
  {
    agentData->gpuAgents.push_back(agent);
    agentData->closestNumaNode.push_back(0);
  }

  return HSA_STATUS_SUCCESS;
}

AgentData& GetAgentData()
{
  static AgentData agentData = {};

  if (!agentData.isInitialized)
  {
    agentData.isInitialized = true;

    // Add all detected agents to the list
    HSA_CHECK(hsa_iterate_agents(AgentInfoCallback, &agentData));

    // Loop over each GPU
    for (uint32_t i = 0; i < agentData.gpuAgents.size(); i++)
    {
      // Collect memory pool
      hsa_amd_memory_pool_t pool;
      HSA_CHECK(hsa_amd_agent_iterate_memory_pools(agentData.gpuAgents[i], MemPoolInfoCallback, &pool));

      // Loop over each CPU agent and check distance
      int bestDistance = -1;
      for (uint32_t j = 0; j < agentData.cpuAgents.size(); j++)
      {
        // Determine number of hops from GPU memory pool to CPU agent
        uint32_t hops = 0;
        HSA_CHECK(hsa_amd_agent_memory_pool_get_info(agentData.cpuAgents[j],
                                                     pool,
                                                     HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS,
                                                     &hops));
        // Gather link info
        hsa_amd_memory_pool_link_info_t* link_info =
          (hsa_amd_memory_pool_link_info_t *)malloc(hops * sizeof(hsa_amd_memory_pool_link_info_t));
        HSA_CHECK(hsa_amd_agent_memory_pool_get_info(agentData.cpuAgents[j],
                                                     pool,
                                                     HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO,
                                                     link_info));
        int numaDist = 0;
        for (int k = 0; k < hops; k++)
        {
          numaDist += link_info[k].numa_distance;
        }
        if (bestDistance == -1 || numaDist < bestDistance)
        {
          agentData.closestNumaNode[i] = j;
          bestDistance = numaDist;
        }
        free(link_info);
      }
    }
  }
  return agentData;
}
#endif

// Returns closest CPU NUMA node to provided GPU
// NOTE: This assumes HSA GPU indexing is similar to HIP GPU indexing
int GetClosestNumaNode(int gpuIdx)
{
#if defined(__NVCC__)
  return -1;
#else
  AgentData& agentData = GetAgentData();
  if (gpuIdx < 0 || gpuIdx >= agentData.closestNumaNode.size())
  {
    printf("[ERROR] GPU index out is out of bounds\n");
    exit(1);
  }
  return agentData.closestNumaNode[gpuIdx];
#endif
}
