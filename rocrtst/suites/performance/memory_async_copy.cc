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

#include <hwloc.h>
#include <hwloc/linux-libnuma.h>
#include <numa.h>

#include <vector>
#include <algorithm>

#include "common/base_rocr.h"
#include "suites/test_common/test_base.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/performance/memory_async_copy.h"
#include "common/base_rocr_utils.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"

#define RET_IF_HSA_ERR(err)                                                                        \
  {                                                                                                \
    if ((err) != HSA_STATUS_SUCCESS) {                                                             \
      const char* msg = 0;                                                                         \
      hsa_status_string(err, &msg);                                                                \
      EXPECT_EQ(HSA_STATUS_SUCCESS, err) << msg;                                                   \
      return (err);                                                                                \
    }                                                                                              \
  }

/* PCIE BDF ID: 0xC81407 is specific to DTIF platform */
static const uint32_t kDtifBdfId = 0xC81407;

constexpr const size_t MemoryAsyncCopy::Size[kNumGranularity];
constexpr const char* MemoryAsyncCopy::Str[kNumGranularity];
constexpr const int MemoryAsyncCopy::kMaxCopySize;

MemoryAsyncCopy::MemoryAsyncCopy(void) :
    TestBase() {
  static_assert(sizeof(Size)/sizeof(size_t) == kNumGranularity,
      "kNumGranularity does not match size of arrays");

  cpu_agent_.handle = 0;  // Ignore any previous initialization
  gpu_local_agent1_.handle = 0;
  gpu_local_agent2_.handle = 0;
  gpu_remote_agent_.handle = 0;
  topology_ = nullptr;
  cpu_hwl_numa_nodeset_ = nullptr;
  agent_index_ = 0;
  pool_index_ = 0;
  tran_.clear();
  agent_info()->clear();
  pool_info()->clear();
  node_info()->clear();
  verified_ = true;
  do_p2p_ = true;
  src_pool_id_ = -1;
  dst_pool_id_ = -1;
  set_num_iteration(10);  // Default value
  set_title("Asynchronous Memory Copy Bandwidth");
  set_description("This test measures bandwidth to/from Host from/to GPU "
      "and Peer to Peer using hsa_amd_memory_async_copy() to copy buffers "
      "of various length from memory pool to another.");
}

MemoryAsyncCopy::~MemoryAsyncCopy(void) {
  for (PoolInfo *p : pool_info_) {
    delete p;
  }

  for (AgentInfo *a : agent_info_) {
    delete a;
  }
}

void MemoryAsyncCopy::SetUp(void) {
  TestBase::SetUp();

  hwloc_topology_init(&topology_);

  FindTopology();

  if (verbosity() >= VERBOSE_STANDARD) {
    PrintTopology();
  }
  ConstructTransactionList();
  return;
}

void MemoryAsyncCopy::Run(void) {
  TestBase::Run();

  for (Transaction t : tran_) {
    this->RunBenchmarkWithVerification(&t);
  }
}

void MemoryAsyncCopy::FindSystemPool(void) {
  hsa_status_t err;

//  err = hsa_iterate_agents(rocrtst::FindCPUDevice, &cpu_agent_);
//  ASSERT_EQ(HSA_STATUS_INFO_BREAK, err);

  err = hsa_amd_agent_iterate_memory_pools(cpu_agent_, rocrtst::FindGlobalPool,
        &sys_pool_);
  ASSERT_EQ(HSA_STATUS_INFO_BREAK, err);
}

hsa_status_t AcquireAccess(hsa_agent_t agent,
                                    hsa_amd_memory_pool_t pool, void* ptr) {
  hsa_status_t err;

  hsa_amd_memory_pool_access_t access;
  err = hsa_amd_agent_memory_pool_get_info(agent, pool,
                              HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);

  RET_IF_HSA_ERR(err);

  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    return HSA_STATUS_ERROR;
  }

  if (access == HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT) {
    err = hsa_amd_agents_allow_access(1, &agent, NULL, ptr);
    RET_IF_HSA_ERR(err);
  }

  return err;
}

// Provided a destination pointer, pool and agent, and a source ptr, pool,
// and agent, get access for one of the 2 agents to the other agent's pool.
// Return the selected agent. This function will first attempt to gain access
// for the first agent to the second pool. If that succeeds, it will return a
// pointer to the first agent. Otherwise, the function will attempt to again
// access to the first pool by the second agent. If that succeeds a pointer to
// the second agent will be returned. If it fails, nullptr will be returned.
// We prefer to use GPU agents over CPU agents to avoid poor copy performance
// due to reading of uncached device memory by CPU.
hsa_agent_t *
MemoryAsyncCopy::AcquireAsyncCopyAccess(
         void *dst_ptr, hsa_amd_memory_pool_t dst_pool, hsa_agent_t *dst_ag,
         void *src_ptr, hsa_amd_memory_pool_t src_pool, hsa_agent_t *src_ag) {
  hsa_status_t err;
  bool can_use_src_agent = false;
  hsa_device_type_t type = HSA_DEVICE_TYPE_CPU;

  err = AcquireAccess(*src_ag, dst_pool, dst_ptr);
  if (err == HSA_STATUS_SUCCESS) {
    can_use_src_agent = true;

    if (hsa_agent_get_info(*src_ag, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
      return NULL;

    // We prefer GPU agents over CPU agents, so if this is not a GPU agent,
    // try using the destination agent
    if (type == HSA_DEVICE_TYPE_GPU) return src_ag;
  }

  err = AcquireAccess(*dst_ag, src_pool, src_ptr);
  if (err == HSA_STATUS_SUCCESS) return dst_ag;

  if (can_use_src_agent) return src_ag;
  return NULL;
}

void MemoryAsyncCopy::PrintTransactionType(Transaction *t) {
  if (verbosity() >= VERBOSE_STANDARD) {
    printf("Executing Copy Path: From Pool %d To Pool %d ", t->src, t->dst);
    switch (t->type) {
      case H2D:
        printf("(Host-To-Device)\n");
        break;

      case D2H:
        printf("(Device-To-Host)\n");
        break;

      case P2P:
        printf("(Peer-To-Peer)\n");
        break;

      case H2DRemote:
        printf("(Host To Remote Device)\n");
        break;

      case D2HRemote:
        printf("(Remote Device To Host)\n");
        break;

      case P2PRemote:
        printf("(Peer To Remote Peer)\n");
        break;

      default:
        printf("**Unexpected path**\n");
        return;
    }
  }
}
void MemoryAsyncCopy::RunBenchmarkWithVerification(Transaction *t) {
  hsa_status_t err;
  void* ptr_src;
  void* ptr_dst;
  size_t src_alloc_size;
  size_t dst_alloc_size;
  size_t max_alloc_size;
  size_t size;
  hsa_device_type_t ag_type;


  size_t max_trans_size = t->max_size * 1024;

  hsa_amd_memory_pool_t src_pool =  pool_info_[t->src]->pool_;
  hsa_agent_t dst_agent = pool_info_[t->dst]->owner_agent_info()->agent();
  hsa_amd_memory_pool_t dst_pool = pool_info_[t->dst]->pool_;
  hsa_agent_t src_agent = pool_info_[t->src]->owner_agent_info()->agent();

  PrintTransactionType(t);

  err = hsa_amd_memory_pool_get_info(src_pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                      &src_alloc_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_agent_get_info(src_agent, HSA_AGENT_INFO_DEVICE, &ag_type);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  if (src_alloc_size <= 536870912 && ag_type == HSA_DEVICE_TYPE_GPU) {
    err = hsa_agent_get_info(src_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
                              &src_alloc_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  err = hsa_amd_memory_pool_get_info(dst_pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                      &dst_alloc_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_agent_get_info(dst_agent, HSA_AGENT_INFO_DEVICE, &ag_type);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  if (dst_alloc_size <= 536870912 && ag_type == HSA_DEVICE_TYPE_GPU) {
    err = hsa_agent_get_info(dst_agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
                              &dst_alloc_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  max_alloc_size = (src_alloc_size < dst_alloc_size) ? src_alloc_size: dst_alloc_size;

  if (dst_alloc_size <= 536870912 && ag_type == HSA_DEVICE_TYPE_GPU)
    size = (max_alloc_size/3 <= max_trans_size) ? max_alloc_size/3: max_trans_size;
  else
    size = (max_alloc_size/2 <= max_trans_size) ? max_alloc_size/2: max_trans_size;

  err = hsa_amd_memory_pool_allocate(src_pool, size, 0,
				      &ptr_src);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_pool_allocate(dst_pool, size, 0,
				      &ptr_dst);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);


  // rocrtst::CommonCleanUp data
  void* host_ptr_src = NULL;
  void* host_ptr_dst = NULL;
  err = hsa_amd_memory_pool_allocate(sys_pool_, size, 0,
                                     reinterpret_cast<void**>(&host_ptr_src));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_amd_memory_pool_allocate(sys_pool_, size, 0,
                                     reinterpret_cast<void**>(&host_ptr_dst));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_fill(host_ptr_src, 1, size/sizeof(uint32_t));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = hsa_amd_memory_fill(host_ptr_dst, 0, size/sizeof(uint32_t));
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  hsa_signal_t s;
  err = hsa_signal_create(1, 0, NULL, &s);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);


  // Deallocate resources...
  MAKE_SCOPE_GUARD([&]() {
    err = hsa_amd_memory_pool_free(ptr_src);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_amd_memory_pool_free(ptr_dst);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_amd_memory_pool_free(host_ptr_src);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_amd_memory_pool_free(host_ptr_dst);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_signal_destroy(s);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  });

  // **** First copy from the system buffer source to the test source pool
  // Acquire the appropriate access; prefer GPU agent over CPU where there
  // is a choice.
  hsa_agent_t *cpy_ag = nullptr;
  cpy_ag = AcquireAsyncCopyAccess(ptr_src, src_pool, &src_agent, host_ptr_src,
                                                     sys_pool_, &cpu_agent_);
  if (cpy_ag == nullptr) {
    std::cout << "Agents " << t->src << " and " << t->dst <<
                              "cannot access each other's pool." << std::endl;
    std::cout << "Skipping..." << std::endl;
    return;
  }

  err = hsa_amd_memory_async_copy(ptr_src, *cpy_ag, host_ptr_src, *cpy_ag,
                                                            size, 0, NULL, s);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1, uint64_t(-1),
                                   HSA_WAIT_STATE_ACTIVE))
  {}

  int iterations = RealIterationNum();

  // **** Next, copy from the test source pool to the test destination pool
  // Prefer a gpu agent to a cpu agent

  cpy_ag = AcquireAsyncCopyAccess(ptr_dst, dst_pool, &dst_agent, ptr_src,
                                                        src_pool, &src_agent);
  if (cpy_ag == nullptr) {
    std::cout << "Owner agents for pools" << t->src << " and " <<
                   t->dst << " cannot access each other's pool." << std::endl;
    std::cout << "Skipping..." << std::endl;
    return;
  }

  for (int i = 0; i < kNumGranularity; i++) {
    if (Size[i] > size) {
      printf("Skip test with block size %s\n", Str[i]);
      break;
    }
    printf("Start test with block size %s\n",Str[i]);

    std::vector<double> time;

    for (int it = 0; it < iterations; it++) {
      if (verbosity() >= VERBOSE_PROGRESS) {
        std::cout << ".";
        std::cout.flush();
      }

      hsa_signal_store_relaxed(t->signal, 1);

      rocrtst::PerfTimer copy_timer;
      int index = copy_timer.CreateTimer();

      copy_timer.StartTimer(index);
      err = hsa_amd_memory_async_copy(ptr_dst, *cpy_ag, ptr_src, *cpy_ag,
                                                 Size[i], 0, NULL, t->signal);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      while (hsa_signal_wait_scacquire(t->signal, HSA_SIGNAL_CONDITION_LT, 1,
                                         uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
      {}

      copy_timer.StopTimer(index);

      hsa_signal_store_relaxed(s, 1);

      err = AcquireAccess(dst_agent, sys_pool_,
                    host_ptr_dst);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);


      err = hsa_amd_memory_async_copy(host_ptr_dst, cpu_agent_, ptr_dst,
                                                 dst_agent, Size[i], 0, NULL, s);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      while (hsa_signal_wait_scacquire(s, HSA_SIGNAL_CONDITION_LT, 1,
                                       uint64_t(-1), HSA_WAIT_STATE_ACTIVE))
      {}

      err = AcquireAccess(cpu_agent_, sys_pool_, host_ptr_dst);
      ASSERT_EQ(HSA_STATUS_SUCCESS, err);

      if (memcmp(host_ptr_src, host_ptr_dst, Size[i])) {
        verified_ = false;
      }
      // Push the result back to vector time

      time.push_back(copy_timer.ReadTimer(index));
    }

    if (verbosity() >= VERBOSE_PROGRESS) {
      std::cout << std::endl;
    }

    // Get Min copy time
    t->min_time->push_back(*std::min_element(time.begin(), time.end()));
    // Get mean copy time and store to the array
    t->benchmark_copy_time->push_back(GetMeanTime(&time));
  }
}

size_t MemoryAsyncCopy::RealIterationNum(void) {
  return num_iteration() * 1.2 + 1;
}

double MemoryAsyncCopy::GetMeanTime(std::vector<double> *vec) {
  std::sort(vec->begin(), vec->end());

  vec->erase(vec->begin());
  vec->erase(vec->begin(), vec->begin() + num_iteration() * 0.1);
  vec->erase(vec->begin() + num_iteration(), vec->end());

  double mean = 0.0;
  int num = vec->size();

  for (int it = 0; it < num; it++) {
    mean += (*vec)[it];
  }

  mean /= num;
  return mean;
}

void MemoryAsyncCopy::DisplayResults(void) const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::DisplayResults();
  hsa_status_t err;
  for (Transaction t : tran_) {
    DisplayBenchmark(&t);
    err = hsa_signal_destroy(t.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    delete t.benchmark_copy_time;
    delete t.min_time;
  }

  return;
}

void MemoryAsyncCopy::DisplayBenchmark(Transaction *t) const {
  hsa_status_t err;
  size_t src_alloc_size;
  size_t dst_alloc_size;
  size_t max_alloc_size;
  size_t size;

  size_t max_trans_size = t->max_size * 1024;
  hsa_amd_memory_pool_t src_pool =  pool_info_[t->src]->pool_;
  hsa_amd_memory_pool_t dst_pool = pool_info_[t->dst]->pool_;

  err = hsa_amd_memory_pool_get_info(src_pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                    &src_alloc_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_get_info(dst_pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                    &dst_alloc_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  max_alloc_size = (src_alloc_size < dst_alloc_size) ? src_alloc_size: dst_alloc_size;

  size = (max_alloc_size/2 <= max_trans_size) ? max_alloc_size/2: max_trans_size;

  printf("=========================== PATH: From Pool %d To Pool %d (",
                                                              t->src, t->dst);

  switch (t->type) {
    case H2D:
      printf("Host-To-Device) ===========================\n");
      break;

    case D2H:
      printf("Device-To-Host) ===========================\n");
      break;

    case P2P:
      printf("Peer-To-Peer) =============================\n");
      break;

    case P2PRemote:
      printf("(Peer-To-Remote-Peer) =====================\n");
      break;

    case H2DRemote:
      printf("(Host-To-Remote-Device) ===================\n");
      break;

    case D2HRemote:
      printf("(Device-To-Remote-Host) ===================\n");
      break;

    default:
      ASSERT_TRUE(false) << "Unexpected Transaction value:" << t->type <<
                                                                    std::endl;
  }

  if ((*t->benchmark_copy_time).size() == 0) {
    printf("Skipped...\n");
    return;
  }
  if (verified_) {
    std::cout << "Verification: Pass" << std::endl;
  } else {
    std::cout << "Verification: Fail" << std::endl;
  }

  if (verbosity() < VERBOSE_STANDARD) {
    return;
  }

  printf("Data Size             Avg Time(us)         Avg BW(GB/s)"
                           "          Min Time(us)          Peak BW(GB/s)\n");

  for (int i = 0; i < kNumGranularity; i++) {

    if (Size[i] > size) {
      printf(
         "Notice: Data Size >= %s is skipped due to hard limit of 1/2 vram size \n\n",
         Str[i]
      );
      break;
    }

    double band_width =
    static_cast<double>(Size[i]/(*(t->benchmark_copy_time))[i]/1024/1024/1024);
    double peak_band_width =
       static_cast<double>(Size[i] / (*(t->min_time))[i]/ 1024 / 1024 / 1024);
    printf(
        "  %4s            %14lf        %14lf         %14lf         %14lf\n",
       Str[i], (*(t->benchmark_copy_time))[i] * 1e6, band_width,
                                  (*(t->min_time))[i] * 1e6, peak_band_width);
  }

  return;
}

void MemoryAsyncCopy::Close() {
  if (cpu_hwl_numa_nodeset_ != nullptr) {
    hwloc_bitmap_free(cpu_hwl_numa_nodeset_);
    cpu_hwl_numa_nodeset_ = nullptr;
  }
  hwloc_topology_destroy(topology_);

  // hwloc hack - hwloc uses OpenCL which loads ROCr.  As OpenCL does not have a shutdown routine it
  // can not free HSA state.  This will leak resources but is the only option short of isolating
  // hwloc in it's own process.
  while (hsa_shut_down() == HSA_STATUS_SUCCESS)
    ;
  hsa_init();

  TestBase::Close();
}

static hsa_status_t GetPoolInfo(hsa_amd_memory_pool_t pool, void* data) {
  hsa_status_t err;
  MemoryAsyncCopy* ptr = reinterpret_cast<MemoryAsyncCopy*>(data);
  // Query pool segment, only report global one
  hsa_amd_segment_t region_segment;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                     &region_segment);
  RET_IF_HSA_ERR(err);

  if (region_segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  // Check if the pool is alloc allowed, if not, discard this pool
  bool alloc_allowed = false;
  err = hsa_amd_memory_pool_get_info(pool,
              HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  RET_IF_HSA_ERR(err);

  if (alloc_allowed != true) {
    return HSA_STATUS_SUCCESS;
  }

  // Query the pool size
  size_t size = 0;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                     &size);
  RET_IF_HSA_ERR(err);

  // Query the max allocable size
  size_t alloc_max_size = 0;
  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                     &alloc_max_size);
  RET_IF_HSA_ERR(err);

  // Check if the pool is fine-grained or coarse-grained
  uint32_t global_flag = 0;
  err = hsa_amd_memory_pool_get_info(pool,
                        HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
  RET_IF_HSA_ERR(err);

  bool is_fine_grained = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED
                         & global_flag;

  int pool_i = ptr->pool_index();
  int ag_ind = ptr->agent_index();
  ptr->pool_info()->push_back(
    new PoolInfo(pool, pool_i, region_segment, is_fine_grained, size,
                                  alloc_max_size, ptr->agent_info()->back()));

  // Construct node_info and push back to agent_info_
  (*ptr->node_info())[ag_ind].pool.push_back(*ptr->pool_info()->back());
  ptr->set_pool_index(pool_i + 1);

  return HSA_STATUS_SUCCESS;
}

static hsa_status_t GetGPUAgents(hsa_agent_t agent, void* data) {
  hsa_status_t err;
  MemoryAsyncCopy* ptr = reinterpret_cast<MemoryAsyncCopy*>(data);

  hsa_device_type_t device_type;
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  RET_IF_HSA_ERR(err);

  if (device_type != HSA_DEVICE_TYPE_GPU) {
    return HSA_STATUS_SUCCESS;
  }

  uint32_t agent_bdf_id;
  err = hsa_agent_get_info(agent,
                (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &agent_bdf_id);
  RET_IF_HSA_ERR(err);

  uint8_t bus = (agent_bdf_id & (0xFF << 8)) >> 8;
  uint8_t device = (agent_bdf_id & (0x1F << 3)) >> 3;

  // The function part of the location_id hasn't been used yet
  // and may not contain a valid function number.
  uint8_t function = 0; //(agent_bdf_id & 0x07);

  if (ptr->verbosity() >  MemoryAsyncCopy::VERBOSE_STANDARD) {
    char name[64];
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
    RET_IF_HSA_ERR(err);

    const char* name2 = (HSA_DEVICE_TYPE_GPU == device_type) ? "GPU" : "CPU";

    printf("The %s agent name located at PCIe Bus %x, Device %x, "
                                                     "Function %x, is %s.\n",
                                          name2, bus, device, function, name);
  }

  uint32_t pci_domain_id = 0;
  err = hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &pci_domain_id);
  RET_IF_HSA_ERR(err);

  hwloc_obj_t gpu_numa_node = nullptr;
  if (agent_bdf_id != kDtifBdfId) {
    hwloc_obj_t gpu_hwl_dev;
    gpu_hwl_dev = hwloc_get_pcidev_by_busid(ptr->topology(), pci_domain_id, bus, device,
                                                                      function);

    if (gpu_hwl_dev == nullptr) {
      return HSA_STATUS_ERROR;
    }

    gpu_numa_node = hwloc_get_ancestor_obj_by_type(ptr->topology(),
                                              HWLOC_OBJ_NUMANODE, gpu_hwl_dev);
  }

  if (gpu_numa_node != nullptr) {
    char s1[256], s2[256];
    hwloc_bitmap_snprintf(s1, sizeof(s1), gpu_numa_node->nodeset);
    hwloc_bitmap_snprintf(s2, sizeof(s2), ptr->cpu_hwl_numa_nodeset());
    printf("gpu nodeset: %s\n", s1);
    printf("cpu nodeset: %s\n", s2);
    if (!hwloc_bitmap_isequal(gpu_numa_node->nodeset,
                                              ptr->cpu_hwl_numa_nodeset())) {
      if (ptr->gpu_remote_agent().handle == 0) {
        ptr->set_gpu_remote_agent(agent);
      }

      if (ptr->gpu_local_agent1().handle != 0 &&
                                          ptr->gpu_local_agent2().handle != 0) {
        return HSA_STATUS_INFO_BREAK;
      } else {
        return HSA_STATUS_SUCCESS;
      }
    } else {
      if (ptr->gpu_local_agent1().handle == 0) {
        ptr->set_gpu_local_agent1(agent);
      } else if (ptr->gpu_local_agent2().handle == 0) {
        ptr->set_gpu_local_agent2(agent);
      }
      if (ptr->gpu_local_agent1().handle != 0 &&
                                     ptr->gpu_local_agent2().handle != 0 &&
                                        ptr->gpu_remote_agent().handle != 0) {
        return HSA_STATUS_INFO_BREAK;
      } else {
        return HSA_STATUS_SUCCESS;
      }
    }

    if (!hwloc_bitmap_isequal(gpu_numa_node->nodeset,
                                               ptr->cpu_hwl_numa_nodeset())) {
      std::cout << "ASSERT: Unexpected unequal nodesets" << std::endl;
      return HSA_STATUS_ERROR;
    }
  } else if (ptr->verbosity() >= MemoryAsyncCopy::VERBOSE_STANDARD) {
    std::cout << "Only 1 NUMA node found.\n" << std::endl;
  }

  if (ptr->gpu_local_agent1().handle != 0) {
    if (ptr->gpu_local_agent2().handle != 0) {
      if (gpu_numa_node == nullptr) {
        return HSA_STATUS_INFO_BREAK;
      } else if (ptr->gpu_remote_agent().handle == 0) {
        return HSA_STATUS_SUCCESS;
      } else {
        return HSA_STATUS_INFO_BREAK;
      }
    } else {
      ptr->set_gpu_local_agent2(agent);
      if (ptr->gpu_remote_agent().handle == 0) {
        return (gpu_numa_node == nullptr ?
                  HSA_STATUS_INFO_BREAK : HSA_STATUS_SUCCESS);
      } else {
        return HSA_STATUS_INFO_BREAK;
      }
    }
  } else {
    ptr->set_gpu_local_agent1(agent);
  }

  return HSA_STATUS_SUCCESS;
}

static hsa_status_t GetAgentInfo(hsa_agent_t agent, void* data) {
  MemoryAsyncCopy* ptr = reinterpret_cast<MemoryAsyncCopy*>(data);

  hsa_status_t err;
  int ret;

  if (ptr->cpu_agent().handle != 0) {
    return HSA_STATUS_ERROR;
  }


  // Get device type
  hsa_device_type_t device_type;
  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  RET_IF_HSA_ERR(err);

  // First thing is to find CPU agent
  if (device_type != HSA_DEVICE_TYPE_CPU) {
    return HSA_STATUS_SUCCESS;
  }

  ptr->set_cpu_agent(agent);
  uint32_t cpu_numa_node_id;
//  hwloc_obj_t cpu_numa;
  hwloc_nodeset_t cpu_nodeset;

  err = hsa_agent_get_info(ptr->cpu_agent(), HSA_AGENT_INFO_NODE,
                                                           &cpu_numa_node_id);
  RET_IF_HSA_ERR(err);

  struct bitmask *numa_node_mask = numa_allocate_nodemask();
  cpu_nodeset = hwloc_bitmap_alloc();

  numa_bitmask_setbit(numa_node_mask, cpu_numa_node_id);

  ret = hwloc_nodeset_from_linux_libnuma_bitmask(ptr->topology(),
      cpu_nodeset, numa_node_mask);
  numa_free_nodemask(numa_node_mask);

  if (ret == -1) {
    hwloc_bitmap_free(cpu_nodeset);
    return HSA_STATUS_ERROR;
  }

  ptr->set_cpu_hwl_numa_nodeset(cpu_nodeset);

  err = hsa_iterate_agents(GetGPUAgents, data);

  if (err != HSA_STATUS_INFO_BREAK && err != HSA_STATUS_SUCCESS) {
    return err;
  }

  if (ptr->gpu_local_agent1().handle == 0) {
    hwloc_bitmap_free(ptr->cpu_hwl_numa_nodeset());
    ptr->set_cpu_hwl_numa_nodeset(nullptr);

    if (ptr->gpu_local_agent2().handle != 0) {
      std::cout << "Unexpected value set for gpu_local_agent2" << std::endl;
      return HSA_STATUS_ERROR;
    }
    // In this case, the CPU and at least 1 GPU are not on the same NUMA node;
    // try another CPU
    hsa_agent_t t;
    t.handle = 0;
    ptr->set_gpu_local_agent1(t);
    ptr->set_cpu_agent(t);
    ptr->set_gpu_remote_agent(t);
    return HSA_STATUS_SUCCESS;
  }
  auto add_agent = [&](hsa_agent_t ag, hsa_device_type_t dev_type,
                                                                bool remote) {
    if (ag.handle == 0) {
      return;
    }
    ptr->agent_info()->push_back(
            new AgentInfo(ag, ptr->agent_index(), dev_type, remote));

    // Contruct a new NodeInfo structure and push back to agent_info_
    NodeInfo node;
    node.agent = *ptr->agent_info()->back();
    ptr->node_info()->push_back(node);

    err = hsa_amd_agent_iterate_memory_pools(ag, GetPoolInfo, data);
    ptr->set_agent_index(ptr->agent_index() + 1);
  };

  add_agent(ptr->cpu_agent(), HSA_DEVICE_TYPE_CPU, false);
  add_agent(ptr->gpu_local_agent1(), HSA_DEVICE_TYPE_GPU, false);
  add_agent(ptr->gpu_local_agent2(), HSA_DEVICE_TYPE_GPU, false);
  add_agent(ptr->gpu_remote_agent(), HSA_DEVICE_TYPE_GPU, true);

  return HSA_STATUS_INFO_BREAK;
}

void MemoryAsyncCopy::FindTopology() {
  hsa_status_t err;

  hwloc_topology_set_flags(topology_, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM |
                                         HWLOC_TOPOLOGY_FLAG_IO_DEVICES);

  hwloc_topology_load(topology_);

  err = hsa_iterate_agents(GetAgentInfo, this);

  if (gpu_local_agent1_.handle == 0) {
    std::cout << "**** No GPU found in same NUMA node as a CPU ****"
                                                                 << std::endl;
  }
  ASSERT_EQ(HSA_STATUS_INFO_BREAK, err);

  FindSystemPool();
}

void MemoryAsyncCopy::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryAsyncCopy::ConstructTransactionList(void) {
  hsa_status_t err;

  tran_.clear();

  int cpu_pool_indx = -1;
  int gpu_local1_pool_indx = -1;
  int gpu_local2_pool_indx = -1;
  int gpu_remote_pool_indx = -1;

  auto push_trans = [&](int from_indx, int to_indx, TransType type) {
    Transaction t;
    t.src = from_indx;
    t.dst = to_indx;
    t.max_size = kMaxCopySize/1024;
    t.type = type;
    t.benchmark_copy_time = new  std::vector<double>;
    t.min_time = new std::vector<double>;
    err = hsa_signal_create(1, 0, NULL, &t.signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    tran_.push_back(t);
  };

  // Find the CPU Node and pool
  for (NodeInfo n : *node_info()) {
    if (cpu_pool_indx == -1 && n.agent.device_type() == HSA_DEVICE_TYPE_CPU) {
      cpu_pool_indx = n.pool[0].index_;
      continue;
    }

    if (n.agent.device_type() == HSA_DEVICE_TYPE_GPU) {
      if (!n.agent.is_remote()) {
        if (gpu_local1_pool_indx == -1) {
          gpu_local1_pool_indx = n.pool[0].index_;
          continue;
        }
        if (gpu_local2_pool_indx == -1) {
          gpu_local2_pool_indx = n.pool[0].index_;
        }
      } else if (gpu_remote_pool_indx == -1) {
        gpu_remote_pool_indx = n.pool[0].index_;
      }
    }
  }

  ASSERT_NE(cpu_pool_indx, -1);
  ASSERT_NE(gpu_local1_pool_indx, -1);

  push_trans(cpu_pool_indx, gpu_local1_pool_indx, H2D);
  push_trans(gpu_local1_pool_indx, cpu_pool_indx, D2H);

  if (do_p2p_ && gpu_local2_pool_indx != -1) {
    push_trans(gpu_local1_pool_indx, gpu_local2_pool_indx, P2P);
    push_trans(gpu_local2_pool_indx, gpu_local1_pool_indx, P2P);
  }

  if (gpu_remote_pool_indx != -1) {
    push_trans(cpu_pool_indx, gpu_remote_pool_indx, H2DRemote);
    push_trans(gpu_remote_pool_indx, cpu_pool_indx, D2HRemote);
    if (do_p2p_) {
      push_trans(gpu_local1_pool_indx, gpu_remote_pool_indx, P2PRemote);
      push_trans(gpu_remote_pool_indx, gpu_local1_pool_indx, P2PRemote);
    }
  }
}

void MemoryAsyncCopy::PrintTopology(void) {
  size_t node_num = node_info()->size();

  for (uint32_t i = 0; i < node_num; i++) {
    NodeInfo node = node_info()->at(i);
    // Print agent info
    std::cout << std::endl;
    std::cout << "Agent #" << node.agent.index_ << ":" << std::endl;

    if (HSA_DEVICE_TYPE_CPU == node.agent.device_type())
      std::cout << "Agent Device Type:                             CPU"
                << std::endl;
    else if (HSA_DEVICE_TYPE_GPU == node.agent.device_type())
      std::cout << "Agent Device Type:                             GPU"
                << std::endl;

    // Print pool info
    size_t pool_num = node.pool.size();

    for (uint32_t j = 0; j < pool_num; j++) {
      std::cout << "    Memory Pool#" << node.pool.at(j).index_ << ":"
                << std::endl;
      std::cout << "        max allocable size in KB: \t\t"
                << node.pool.at(j).allocable_size_ / 1024 << std::endl;
      std::cout << "        is fine-grained: \t\t\t"
                << node.pool.at(j).is_fine_grained_ << std::endl;
    }
  }
}

#undef RET_IF_HSA_ERR
