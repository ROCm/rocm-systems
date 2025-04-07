#include "common.hpp"
#include "rocm_async.hpp"

// @Brief: Print Help Menu Screen
void RocmAsync::PrintHelpScreen() {

  std::cout << std::endl;
  std::cout << "Runs with following options:" << std::endl;
  std::cout << std::endl;
  std::cout << "\t -h Prints the help screen" << std::endl;
  std::cout << "\t -g Prints Gpu times for transfers" << std::endl;
  std::cout << "\t -t Prints system topology and its memory pools" << std::endl;
  std::cout << "\t -m List of buffer sizes to use, specified in Megabytes" << std::endl;
  std::cout << "\t -r List of pool,agent pairs engaged in Read operation" << std::endl;
  std::cout << "\t -w List of pool,agent pairs engaged in Write operation" << std::endl;
  std::cout << "\t -b List pools to use in bidirectional copy operations" << std::endl;
  std::cout << "\t -s List of source pools to use in copy unidirectional operations" << std::endl;
  std::cout << "\t -d List of destination pools to use in unidirectional copy operations" << std::endl;
  std::cout << "\t -a Perform Unidirectional Copy involving all pool combinations" << std::endl;
  std::cout << "\t -A Perform Bidirectional Copy involving all pool combinations" << std::endl;
  std::cout << std::endl;
  
  std::cout << std::endl;
  std::cout << "\t @note 1: Removes copyReq(srcI, dstJ) - where either Src or Dst Pool is fine-grained" << std::endl;
  std::cout << std::endl;
  std::cout << "\t @note 2: Treats copyReq(dstI, srcJ) as NOT EQUAL to copyReq(dstJ, srcI) " << std::endl;
  std::cout << "\t            Underlying copy engine could be different " << std::endl;
  std::cout << std::endl;

  /*
  std::cout << "\t @note 1: Removes copyReq(srcI, dstI) - where Src & Dst Pools are same" << std::endl;
  std::cout << std::endl;
  std::cout << "\t @note 2: Removes copyReq(srcI, dstJ) - where Src & Dst Pools are Cpu bound" << std::endl;
  std::cout << std::endl;
  std::cout << "\t @note 3: Removes copyReq(srcI, dstJ) - where either Src or Dst Pool is fine-grained" << std::endl;
  std::cout << std::endl;
  std::cout << "\t @note 4: Treats copyReq(dstI, srcJ) as NOT EQUAL to copyReq(dstJ, srcI) " << std::endl;
  std::cout << "\t            Underlying copy engine could be different " << std::endl;
  std::cout << std::endl;
  */
}

// @brief: Print the topology of Memory Pools and Agents present in system
void RocmAsync::PrintTopology() {

  size_t count = agent_pool_list_.size();
  std::cout << std::endl;
  for (uint32_t idx = 0; idx < count; idx++) {
    agent_pool_info_t node = agent_pool_list_.at(idx);

    // Print agent info
    std::cout << "Agent: " << node.agent.index_ << std::endl;
    if (HSA_DEVICE_TYPE_CPU == node.agent.device_type_)
      std::cout << "  Agent Device Type:                            CPU" << std::endl;
    else if (HSA_DEVICE_TYPE_GPU == node.agent.device_type_)
      std::cout << "  Agent Device Type:                            GPU" << std::endl;

    // Print pool info
    size_t pool_count = node.pool_list.size();
    for (uint32_t jdx = 0; jdx < pool_count; jdx++) {
      std::cout << "    Memory Pool:                                "
           << node.pool_list.at(jdx).index_ << std::endl;
      std::cout << "        max allocable size in KB:               "
           << node.pool_list.at(jdx).allocable_size_ / 1024 << std::endl;
      std::cout << "        segment id:                             "
           << node.pool_list.at(jdx).segment_ << std::endl;
      std::cout << "        is kernarg:                             "
           << node.pool_list.at(jdx).is_kernarg_ << std::endl;
      std::cout << "        is fine-grained:                        "
           << node.pool_list.at(jdx).is_fine_grained_ << std::endl;
      std::cout << "        accessible to owner:                    "
           << node.pool_list.at(jdx).owner_access_ << std::endl;
      std::cout << "        accessible to all by default:           "
           << node.pool_list.at(jdx).access_to_all_ << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// @brief: Print info on agents in system
void RocmAsync::PrintAgentsList() {

  size_t count = agent_pool_list_.size();
  for (uint32_t idx = 0; idx < count; idx++) {
    std::cout << std::endl;
    agent_pool_info_t node = agent_pool_list_.at(idx);
    std::cout << "Agent: " << node.agent.index_ << std::endl;
    if (HSA_DEVICE_TYPE_CPU == node.agent.device_type_)
      std::cout << "  Agent Device Type:            CPU" << std::endl;
    else if (HSA_DEVICE_TYPE_GPU == node.agent.device_type_)
      std::cout << "  Agent Device Type:           GPU" << std::endl;
  }
  std::cout << std::endl;
}

// @brief: Print info on memory pools in system
void RocmAsync::PrintPoolsList() {

  size_t pool_count = pool_list_.size();
  for (uint32_t jdx = 0; jdx < pool_count; jdx++) {
    std::cout << std::endl;
    std::cout << "Memory Pool Idx:                          "
         << pool_list_.at(jdx).index_ << std::endl;
    std::cout << "  max allocable size in KB:               "
         << pool_list_.at(jdx).allocable_size_ / 1024 << std::endl;
    std::cout << "  segment id:                             "
         << pool_list_.at(jdx).segment_ << std::endl;
    std::cout << "  is kernarg:                             "
         << pool_list_.at(jdx).is_kernarg_ << std::endl;
    std::cout << "  is fine-grained:                        "
         << pool_list_.at(jdx).is_fine_grained_ << std::endl;
    std::cout << "  accessible to owner:                    "
         << pool_list_.at(jdx).owner_access_ << std::endl;
    std::cout << "  accessible to all by default:           "
         << pool_list_.at(jdx).access_to_all_ << std::endl;
  }
  std::cout << std::endl;

}

// @brief: Print the list of transactions that will be executed
void RocmAsync::PrintTransList() {

  size_t count = trans_list_.size();
  for (uint32_t idx = 0; idx < count; idx++) {
    async_trans_t trans = trans_list_.at(idx);
    std::cout << std::endl;
    std::cout << "                 Transaction Id: " << idx << std::endl;
    std::cout << "               Transaction Type: " << trans.req_type_ << std::endl;
    if ((trans.req_type_ == REQ_READ) || (trans.req_type_ == REQ_WRITE)) {
      std::cout << "Rocm Kernel used by Transaction: " << trans.kernel.code_ << std::endl;
      std::cout << "Rocm Memory Pool Used by Kernel: " << trans.kernel.pool_idx_ << std::endl;
      std::cout << "  Rocm Agent used for Execution: " << trans.kernel.agent_idx_ << std::endl;
    }
    if ((trans.req_type_ == REQ_COPY_BIDIR) || (trans.req_type_ == REQ_COPY_UNIDIR)) {
      std::cout << "   Src Memory Pool used in Copy: " << trans.copy.src_idx_ << std::endl;
      std::cout << "   Dst Memory Pool used in Copy: " << trans.copy.dst_idx_ << std::endl;
    }

  }
  std::cout << std::endl;
}

// @brief: Prints error message when a request to copy between
// source pool and destination pool is not possible
void RocmAsync::PrintCopyAccessError(uint32_t src_idx, uint32_t dst_idx) {

  // Retrieve Roc runtime handles for Src memory pool and agents
  uint32_t src_dev_idx = pool_list_[src_idx].agent_index_;
  hsa_device_type_t src_dev_type = agent_list_[src_dev_idx].device_type_;
    
  // Retrieve Roc runtime handles for Dst memory pool and agents
  uint32_t dst_dev_idx = pool_list_[dst_idx].agent_index_;
  hsa_device_type_t dst_dev_type = agent_list_[dst_dev_idx].device_type_;

  std::cout << std::endl;
  std::cout << "Index of Src Pool: " << src_idx << std::endl;
  std::cout << "Index of Dst Pool: " << dst_idx << std::endl;
  std::cout << "Index of Src Pool's Agent: " << src_dev_idx << std::endl;
  std::cout << "Index of Dst Pool's Agent: " << dst_dev_idx << std::endl;
  std::cout << "Device Type of Src Pool's Agent: " << src_dev_type << std::endl;
  std::cout << "Device Type of Dst Pool's Agent: " << dst_dev_type << std::endl;
  std::cout << "Rocm Agent hosting Src Pool cannot ACCESS Dst Pool" << std::endl;
  std::cout << std::endl;
}

// @brief: Prints error message when a request to read / write from
// a pool by an agent is not possible
void RocmAsync::PrintIOAccessError(uint32_t exec_idx, uint32_t pool_idx) {

  // Retrieve device type of executing agent
  hsa_device_type_t exec_dev_type = agent_list_[exec_idx].device_type_;
    
  // Retrieve device type of memory pool's agent
  uint32_t pool_dev_idx = pool_list_[pool_idx].agent_index_;
  hsa_device_type_t pool_dev_type = agent_list_[pool_dev_idx].device_type_;

  std::cout << std::endl;
  std::cout << "Index of Executing Agent: " << exec_idx << std::endl;
  std::cout << "Device Type of Executing Agent: " << exec_dev_type << std::endl;
  
  std::cout << "Index of Buffer's Memory Pool: " << pool_idx << std::endl;
  std::cout << "Index of Buffer Memory Pool's Agent: " << pool_dev_idx << std::endl;
  std::cout << "Device Type of Buffer Memory Pool's Agent: " << pool_dev_type << std::endl;
  std::cout << "Rocm Agent executing Read / Write request cannot ACCESS Buffer's Memory Pool" << std::endl;
  std::cout << std::endl;
}
