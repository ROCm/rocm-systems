/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#ifndef ROCRTST_SUITES_PERFORMANCE_MEMORY_ASYNC_COPY_H_
#define ROCRTST_SUITES_PERFORMANCE_MEMORY_ASYNC_COPY_H_

#include <hwloc.h>

#include <vector>
#include <algorithm>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/test_common/test_base.h"

hsa_status_t AcquireAccess(hsa_agent_t agent,
                                    hsa_amd_memory_pool_t pool, void* ptr);
typedef enum TransType
              {H2D = 0, D2H, P2P, H2DRemote, D2HRemote, P2PRemote} TransType;

typedef struct Transaction {
  int src;
  int dst;
  hsa_signal_t signal;
  size_t max_size;  // Max. amount of kBytes to copy
  TransType type;
  // BenchMark copy time
  std::vector<double> *benchmark_copy_time;
  // Min time
  std::vector<double> *min_time;
} Transaction;

class AgentInfo {
 public:
    AgentInfo(hsa_agent_t agent, int index, hsa_device_type_t device_type,
                                                        bool remote = false) {
      agent_ = agent;
      index_ = index;
      device_type_ = device_type;
      remote_ = remote;
    }
    AgentInfo() {}

    ~AgentInfo() {}
    hsa_agent_t agent(void) const {return agent_;}
    hsa_device_type_t device_type(void) const {return device_type_;}
    bool is_remote(void) const {return remote_;}
    void set_remote(bool r) {remote_ = r;}
    hsa_agent_t agent_;
    int index_;

 private:
    hsa_device_type_t device_type_;
    bool remote_;
};

class PoolInfo {
 public:
    PoolInfo(hsa_amd_memory_pool_t pool, int index,
               hsa_amd_segment_t segment, bool is_fine_grained, size_t size,
               size_t max_alloc_size, AgentInfo *agent_info) {
      pool_ = pool;
      index_ = index;
      segment_ = segment;
      is_fine_grained_ = is_fine_grained;
      size_ = size;
      allocable_size_ = max_alloc_size;
      owner_agent_info_ = agent_info;
    }
    PoolInfo() {}
    ~PoolInfo() {}
    AgentInfo* owner_agent_info(void) const {return owner_agent_info_;}
    hsa_amd_memory_pool_t pool_;
    int index_;
    hsa_amd_segment_t segment_;
    bool is_fine_grained_;
    size_t size_;
    size_t allocable_size_;
 private:
    AgentInfo *owner_agent_info_;
};


// Used to print out topology info
typedef struct NodeInfo {
  AgentInfo agent;
  std::vector<PoolInfo> pool;
} NodeInfo;


class MemoryAsyncCopy : public TestBase {
 public:
  MemoryAsyncCopy();

  // @Brief: Destructor for test case of MemoryAsyncCopy
  virtual ~MemoryAsyncCopy();

  // @Brief: Setup the environment for measurement
  virtual void SetUp();

  // @Brief: Core measurement execution
  virtual void Run();

  // @Brief: Clean up and retrive the resource
  virtual void Close();

  // @Brief: Display  results
  virtual void DisplayResults() const;

  // @Brief: Display information about what this test does
  virtual void DisplayTestInfo(void);

  // There are 3 levels of testing, from quickest/very specific to
  // longest/most complete:
  // 1. to and from a specified source to a specified target
  // 2. to and from the cpu to 1 gpu, and to/from a gpu to another gpu
  //    (if available)
  // 3. to and from the cpu to 1 gpu and, to/from every gpu to every
  //    other gpu
  // The default is #2 above. If *both* a source and dest. are set for #1
  // above, then that overides both #2 and #3
  void set_src_pool(int pool_id) {src_pool_id_ = pool_id;}
  void set_dst_pool(int pool_id) {dst_pool_id_ = pool_id;}
  int pool_index(void) const {return pool_index_;}
  void set_pool_index(int i) {pool_index_ = i;}
  int agent_index(void) const {return agent_index_;}
  void set_agent_index(int i) {agent_index_ = i;}
  std::vector<PoolInfo *> *pool_info(void) {return &pool_info_;}
  std::vector<AgentInfo *> *agent_info(void) {return &agent_info_;}
  std::vector<NodeInfo> *node_info(void) {return &node_info_;}

  hwloc_topology_t topology(void) const {return topology_;}
  void set_topology(hwloc_topology_t t) {topology_ = t;}

  hwloc_nodeset_t cpu_hwl_numa_nodeset(void) const {
                                                return cpu_hwl_numa_nodeset_;}
  void set_cpu_hwl_numa_nodeset(hwloc_nodeset_t ns) {
                                                  cpu_hwl_numa_nodeset_ = ns;}
  hsa_agent_t gpu_local_agent1() const {return gpu_local_agent1_;}
  void set_gpu_local_agent1(hsa_agent_t a) {gpu_local_agent1_ = a;}
  hsa_agent_t gpu_local_agent2() const {return gpu_local_agent2_;}
  void set_gpu_local_agent2(hsa_agent_t a) {gpu_local_agent2_ = a;}

  hsa_agent_t gpu_remote_agent() const {return gpu_remote_agent_;}
  void set_gpu_remote_agent(hsa_agent_t a) {gpu_remote_agent_ = a;}

  hsa_agent_t cpu_agent() const {return cpu_agent_;}
  void set_cpu_agent(hsa_agent_t a) {cpu_agent_ = a;}

  hsa_agent_t *
  AcquireAsyncCopyAccess(
         void *dst_ptr, hsa_amd_memory_pool_t dst_pool, hsa_agent_t *dst_ag,
         void *src_ptr, hsa_amd_memory_pool_t src_pool, hsa_agent_t *src_ag);

 protected:
  void PrintTransactionType(Transaction *t);
#if ROCRTST_EMULATOR_BUILD
  static const int kNumGranularity = 1;
  static constexpr const char* Str[kNumGranularity] = {"1k"};

  static constexpr const size_t Size[kNumGranularity] = {1024};
#else

  static const int kNumGranularity = 20;
  static constexpr const char* Str[kNumGranularity] = {
      "1k", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K",
      "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M", "256M", "512M"};

  static constexpr const size_t Size[kNumGranularity] = {
      1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024,
      256*1024, 512*1024, 1024*1024, 2048*1024, 4096*1024, 8*1024*1024,
      16*1024*1024, 32*1024*1024, 64*1024*1024, 128*1024*1024, 256*1024*1024,
      512*1024*1024};
#endif
  static constexpr const int kMaxCopySize = Size[kNumGranularity - 1];

  // @Brief: Get real iteration number
  virtual size_t RealIterationNum(void);

  // @Brief: Get the mean copy time
  double GetMeanTime(std::vector<double>* vec);

  // @Brief: Find and print out the needed topology info
  virtual void FindTopology(void);

  // @Brief: Run for Benchmark mode with verification
  virtual void RunBenchmarkWithVerification(Transaction *t);

  // @Brief: Dispaly Benchmark result
  void DisplayBenchmark(Transaction *t) const;

  // @Brief: Print topology info
  void PrintTopology(void);

  virtual void ConstructTransactionList(void);

  // @Brief: Find system region
  void FindSystemPool(void);

  // More variables declared for testing
  std::vector<Transaction> tran_;

  // Variable used to store agent info, indexed by agent_index_
  std::vector<AgentInfo *> agent_info_;

  // Variable used to store region info, indexed by pool_index_
  std::vector<PoolInfo *> pool_info_;

  // To store node info
  std::vector<NodeInfo> node_info_;

  // Variable to help count agent index
  int agent_index_;

  // Variable to help count region index
  int pool_index_;

  // Verification result
  bool verified_;

  // Should we test p2p copying?
  bool do_p2p_;

  // Store the testing level
  int src_pool_id_;
  int dst_pool_id_;
  // System region
  hsa_amd_memory_pool_t sys_pool_;

  // CPU agent used for verification
  hsa_agent_t cpu_agent_;

  rocrtst::PerfTimer copy_timer_;

  hwloc_topology_t topology_;
  hwloc_nodeset_t cpu_hwl_numa_nodeset_;

  // hsa_agent_t cpu_agent_; use one in base class
  hsa_agent_t gpu_local_agent1_;
  hsa_agent_t gpu_local_agent2_;
  hsa_agent_t gpu_remote_agent_;  // Not associated with cpu_agent_
};


#endif  // ROCRTST_SUITES_PERFORMANCE_MEMORY_ASYNC_COPY_H_
