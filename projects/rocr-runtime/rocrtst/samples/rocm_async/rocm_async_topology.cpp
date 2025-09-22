#include "common.hpp"
#include "rocm_async.hpp"

// @brief: Helper method to iterate throught the memory pools of
// an agent and discover its properties
hsa_status_t MemPoolInfo(hsa_amd_memory_pool_t pool, void* data) {

  hsa_status_t status;
  RocmAsync* asyncDrvr = reinterpret_cast<RocmAsync*>(data);

  // Query pools' segment, report only pools from global segment
  hsa_amd_segment_t segment;
  status = hsa_amd_memory_pool_get_info(pool,
                   HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  ErrorCheck(status);
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  // Determine if allocation is allowed in this pool
  // Report only pools that allow an alloction by user
  bool alloc = false;
  status = hsa_amd_memory_pool_get_info(pool,
                   HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
  ErrorCheck(status);
  if (alloc != true) {
    return HSA_STATUS_SUCCESS;
  }

  // Query the pool size
  size_t size = 0;
  status = hsa_amd_memory_pool_get_info(pool,
                   HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
  ErrorCheck(status);

  // Query the max allocatable size
  size_t max_size = 0;
  status = hsa_amd_memory_pool_get_info(pool,
                   HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, &max_size);
  ErrorCheck(status);

  // Determine if the pools is accessible to all agents
  bool access_to_all = false;
  status = hsa_amd_memory_pool_get_info(pool,
                HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &access_to_all);
  ErrorCheck(status);

  // Determine type of access to owner agent
  hsa_amd_memory_pool_access_t owner_access;
  hsa_agent_t agent = asyncDrvr->agent_list_.back().agent_;
  status = hsa_amd_agent_memory_pool_get_info(agent, pool,
                         HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &owner_access);
  ErrorCheck(status);

  // Determine if the pool is fine-grained or coarse-grained
  uint32_t flag = 0;
  status = hsa_amd_memory_pool_get_info(pool,
                   HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
  ErrorCheck(status);
  bool is_kernarg = (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT & flag);
  bool is_fine_grained = (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED & flag);

  // Update the pool handle for system memory if kernarg is true
  if (is_kernarg) {
    asyncDrvr->sys_pool_ = pool;
  }

  // Create an instance of agent_pool_info and add it to the list
  pool_info_t pool_info(agent, asyncDrvr->agent_index_, pool,
                        segment, size, max_size, asyncDrvr->pool_index_,
                        is_fine_grained, is_kernarg,
                        access_to_all, owner_access);
  asyncDrvr->pool_list_.push_back(pool_info);

  // Create an agent_pool_infot and add it to its list
  asyncDrvr->agent_pool_list_[asyncDrvr->agent_index_].pool_list.push_back(pool_info);
  asyncDrvr->pool_index_++;

  return HSA_STATUS_SUCCESS;
}

// @brief: Helper method to iterate throught the agents of
// a system and discover its properties
hsa_status_t AgentInfo(hsa_agent_t agent, void* data) {

  RocmAsync* asyncDrvr = reinterpret_cast<RocmAsync*>(data);

  // Get the name of the agent
  char agent_name[64];
  hsa_status_t status;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);
  ErrorCheck(status);

  // Get device type
  hsa_device_type_t device_type;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  ErrorCheck(status);

  // Capture the handle of Cpu agent
  if (device_type == HSA_DEVICE_TYPE_CPU) {
    asyncDrvr->cpu_agent_ = agent;
  }

  asyncDrvr->agent_list_.push_back(agent_info(agent, asyncDrvr->agent_index_, device_type));

  // Contruct an new agent_pool_info structure and add it to the list
  agent_pool_info node;
  node.agent = asyncDrvr->agent_list_.back();
  asyncDrvr->agent_pool_list_.push_back(node);

  status = hsa_amd_agent_iterate_memory_pools(agent, MemPoolInfo, asyncDrvr);
  asyncDrvr->agent_index_++;

  return HSA_STATUS_SUCCESS;
}

void RocmAsync::DiscoverTopology() {
  err_ = hsa_iterate_agents(AgentInfo, this);
}

