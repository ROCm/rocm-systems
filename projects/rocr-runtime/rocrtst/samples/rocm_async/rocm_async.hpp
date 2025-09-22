#ifndef __ROCM_ASYNC_BW_H__
#define __ROCM_ASYNC_BW_H__

#include "hsa/hsa.h"
#include "base_test.hpp"
#include "hsatimer.hpp"
#include "common.hpp"
#include <vector>

using namespace std;

// Structure to encapsulate a RocR agent and its index in a list
typedef struct agent_info {

  agent_info(hsa_agent_t agent,
             uint32_t index, hsa_device_type_t device_type) {
    agent_ = agent;
    index_ = index;
    device_type_ = device_type;
  }

  agent_info() {}
  
  uint32_t index_;
  hsa_agent_t agent_;
  hsa_device_type_t device_type_;

} agent_info_t;

typedef struct pool_info {

  pool_info(hsa_agent_t agent, uint32_t agent_index,
            hsa_amd_memory_pool_t pool, hsa_amd_segment_t segment,
            size_t size, size_t alloc_max_size, uint32_t index,
            bool is_fine_grained, bool is_kernarg, bool access_to_all,
            hsa_amd_memory_pool_access_t owner_access) {

    pool_ = pool;
    index_ = index;
    segment_ = segment;
    owner_agent_ = agent;
    agent_index_ = agent_index;
    size_ = size;
    allocable_size_ = alloc_max_size;
    is_kernarg_ = is_kernarg;
    owner_access_ = owner_access;
    access_to_all_ = access_to_all;
    is_fine_grained_ = is_fine_grained;
  }

  pool_info() {}

  uint32_t index_;
  bool is_kernarg_;
  bool access_to_all_;
  bool is_fine_grained_;
  size_t size_;
  size_t allocable_size_;
  uint32_t agent_index_;
  hsa_agent_t owner_agent_;
  hsa_amd_segment_t segment_;
  hsa_amd_memory_pool_t pool_;
  hsa_amd_memory_pool_access_t owner_access_;

} pool_info_t;

// Used to print out topology info
typedef struct agent_pool_info {

  agent_pool_info() {}
  
  agent_info agent;
  
  vector<pool_info_t> pool_list;

} agent_pool_info_t;

typedef struct async_trans {

  uint32_t req_type_;
  union {
    struct {
      bool bidir_;
      bool uses_gpu_;
      uint32_t src_idx_;
      uint32_t dst_idx_;
      hsa_amd_memory_pool_t src_pool_;
      hsa_amd_memory_pool_t dst_pool_;
    } copy;
    struct {
      void* code_;
      uint32_t agent_idx_;
      hsa_agent_t agent_;
      uint32_t pool_idx_;
      hsa_amd_memory_pool_t pool_;
    } kernel;
  };

  // Cpu BenchMark average copy time
  vector<double> cpu_avg_time_;

  // Cpu Min time
  vector<double> cpu_min_time_;

  // Gpu BenchMark average copy time
  vector<double> gpu_avg_time_;

  // Gpu Min time
  vector<double> gpu_min_time_;

  // BenchMark's Average copy time and average bandwidth
  vector<double> avg_time_;
  vector<double> avg_bandwidth_;

  // BenchMark's Min copy time and peak bandwidth
  vector<double> min_time_;
  vector<double> peak_bandwidth_;

  async_trans(uint32_t req_type) { req_type_ = req_type; }
} async_trans_t;

typedef enum Request_Type {

  REQ_READ = 1,
  REQ_WRITE = 2,
  REQ_COPY_BIDIR = 3,
  REQ_COPY_UNIDIR = 4,
  REQ_COPY_ALL_BIDIR = 5,
  REQ_COPY_ALL_UNIDIR = 6,
  REQ_INVALID = 7,

} Request_Type;

class RocmAsync : public BaseTest {

 public:

  // @brief: Constructor for test case of RocmAsync
  RocmAsync(int argc, char** argv);

  // @brief: Destructor for test case of RocmAsync
  virtual ~RocmAsync();

  // @brief: Setup the environment for measurement
  virtual void SetUp();

  // @brief: Core measurement execution
  virtual void Run();

  // @brief: Clean up and retrive the resource
  virtual void Close();

  // @brief: Display the results
  virtual void Display() const;

 private:

  // @brief: Print Help Menu Screen
  void PrintHelpScreen();

  // @brief: Discover the topology of pools on Rocm Platform
  void DiscoverTopology();

  // @brief: Print topology info
  void PrintTopology();

  // @brief: Print info on agents in system
  void PrintAgentsList();

  // @brief: Print info on memory pools in system
  void PrintPoolsList();

  // @brief: Parse the arguments provided by user to
  // build list of transactions
  void ParseArguments();
  
  // @brief: Print the list of transactions
  void PrintTransList();

  // @brief: Run read/write requests of users
  void RunIOBenchmark(async_trans_t& trans);

  // @brief: Run copy requests of users
  void RunCopyBenchmark(async_trans_t& trans);

  // @brief: Get iteration number
  uint32_t GetIterationNum();

  // @brief: Get the mean copy time
  double GetMeanTime(std::vector<double>& vec);

  // @brief: Get the min copy time
  double GetMinTime(std::vector<double>& vec);

  // @brief: Dispaly Benchmark result
  void DisplayIOTime(async_trans_t& trans) const;
  void DisplayCopyTime(async_trans_t& trans) const;
  void DisplayCopyTimeMatrix() const;

  private:

  // @brief: Validate the arguments passed in by user
  bool ValidateArguments();
  bool ValidateReadReq();
  bool ValidateWriteReq();
  bool ValidateReadOrWriteReq(vector<uint32_t>& in_list);
  
  bool ValidateBidirCopyReq();
  bool ValidateUnidirCopyReq();
  bool ValidateCopyReq(vector<uint32_t>& in_list);
  void PrintIOAccessError(uint32_t agent_idx, uint32_t pool_idx);
  void PrintCopyAccessError(uint32_t src_pool_idx, uint32_t dst_pool_idx);
  
  bool PoolIsPresent(vector<uint32_t>& in_list);
  bool PoolIsDuplicated(vector<uint32_t>& in_list);

  // @brief: Builds a list of transaction per user request
  void ComputeCopyTime(async_trans_t& trans);
  bool BuildTransList();
  bool BuildReadTrans();
  bool BuildWriteTrans();
  bool BuildBidirCopyTrans();
  bool BuildUnidirCopyTrans();
  bool BuildAllPoolsBidirCopyTrans();
  bool BuildAllPoolsUnidirCopyTrans();
  bool BuildReadOrWriteTrans(uint32_t req_type,
                             vector<uint32_t>& in_list);
  bool BuildCopyTrans(uint32_t req_type,
                      vector<uint32_t>& src_list,
                      vector<uint32_t>& dst_list);

  void AllocateCopyBuffers(bool bidir, uint32_t size,
                           void*& src_fwd, hsa_amd_memory_pool_t src_pool_fwd,
                           void*& dst_fwd, hsa_amd_memory_pool_t dst_pool_fwd,
                           hsa_agent_t src_agent_fwd, hsa_agent_t dst_agent_fwd,
                           void*& src_rev, hsa_amd_memory_pool_t src_pool_rev,
                           void*& dst_rev, hsa_amd_memory_pool_t dst_pool_rev,
                           hsa_agent_t src_agent_rev, hsa_agent_t dst_agent_rev,
                           hsa_signal_t& signal_fwd, hsa_signal_t& signal_rev);
  void ReleaseBuffers(bool bidir,
                      void* src_fwd, void* src_rev,
                      void* dst_fwd, void* dst_rev,
                      hsa_signal_t signal_fwd, hsa_signal_t signal_rev);
  double GetGpuCopyTime(bool bidir, hsa_signal_t signal_fwd, hsa_signal_t signal_rev);
  void AllocateHostBuffers(bool bidir, uint32_t size,
                                    void*& src_fwd, void*& dst_fwd,
                                    void* buf_src_fwd, void* buf_dst_fwd,
                                    hsa_agent_t src_agent_fwd, hsa_agent_t dst_agent_fwd,
                                    void*& src_rev, void*& dst_rev,
                                    void* buf_src_rev, void* buf_dst_rev,
                                    hsa_agent_t src_agent_rev, hsa_agent_t dst_agent_rev,
                                    hsa_signal_t& signal_fwd, hsa_signal_t& signal_rev);
  void copy_buffer(void* dst, hsa_agent_t dst_agent,
                   void* src, hsa_agent_t src_agent,
                   size_t size, hsa_signal_t signal);

  // @brief: Check if agent and access memory pool, if so, set 
  // access to the agent, if not, exit
  void AcquireAccess(hsa_agent_t agent, void* ptr);

  // Functions to find agents and memory pools and udpate
  // relevant data structures used to maintain system topology
  friend hsa_status_t AgentInfo(hsa_agent_t agent, void* data);
  friend hsa_status_t MemPoolInfo(hsa_amd_memory_pool_t pool, void* data);

 protected:
  
  // More variables declared for testing
  // vector<transaction> tran_;

  // Used to help count agent_info
  uint32_t agent_index_;

  // List used to store agent info, indexed by agent_index_
  vector<agent_info_t> agent_list_;

  // Used to help count pool_info_t
  uint32_t pool_index_;

  // List used to store pool_info_t, indexed by pool_index_
  vector<pool_info_t> pool_list_;

  // List used to store agent_pool_info_t
  vector<agent_pool_info_t> agent_pool_list_;

  // List of agents involved in a bidrectional copy operation
  // Size of the list cannot exceed the number of agents
  // reported by the system
  vector<uint32_t> bidir_list_;

  // List of source agents in a unidrectional copy operation
  // Size of the list cannot exceed the number of agents
  // reported by the system
  vector<uint32_t> src_list_;

  // List of destination agents in a unidrectional copy operation
  // Size of the list cannot exceed the number of agents
  // reported by the system
  vector<uint32_t> dst_list_;

  // List of agents involved in read operation. Has
  // two agents, the first agent hosts the memory pool
  // while the second agent executes the read operation
  vector<uint32_t> read_list_;
  
  // List of agents involved in write operation. Has
  // two agents, the first agent hosts the memory pool
  // while the second agent executes the write operation
  vector<uint32_t> write_list_;
  
  // List of sizes to use in copy and read/write transactions
  // Size is specified in terms of Megabytes
  vector<uint32_t> size_list_;

  // Type of service requested by user
  uint32_t req_read_;
  uint32_t req_write_;
  uint32_t req_copy_bidir_;
  uint32_t req_copy_unidir_;
  uint32_t req_copy_all_bidir_;
  uint32_t req_copy_all_unidir_;

  // List used to store transactions per user request
  vector<async_trans_t> trans_list_;

  // List used to store transactions involving Cpu-Gpu pools
  vector<async_trans_t> matrix_trans_list_;

  // Variable to store argument number

  // Variable to store argument number

  // Variable to store argument number
  uint32_t usr_argc_;

  // Pointer to store address of argument text
  char** usr_argv_;

  // BenchMark copy time
  vector<double> op_time_;

  // Min time
  vector<double> min_time_;

  // Determines if user has requested verification
  bool verify_;

  // CPU agent used for verification
  hsa_agent_t cpu_agent_;

  // System region
  hsa_amd_memory_pool_t sys_pool_;
 
  static const uint32_t SIZE_LIST[4];
  //static const uint32_t SIZE_LIST[9];

};

#endif
