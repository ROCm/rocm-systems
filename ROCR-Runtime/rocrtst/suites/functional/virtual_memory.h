/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2022, Advanced Micro Devices, Inc.
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
#ifndef ROCRTST_SUITES_FUNCTIONAL_VIRTUAL_MEMORY_H_
#define ROCRTST_SUITES_FUNCTIONAL_VIRTUAL_MEMORY_H_

#include <atomic>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class VirtMemoryTestBasic : public TestBase {
 public:
  VirtMemoryTestBasic();

  // @Brief: Destructor for test case of VirtMemoryTestBasic
  virtual ~VirtMemoryTestBasic();

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

  void TestCreateDestroy(void);
  void TestRefCount(void);
  void TestPartialMapping(void);
  void NonContiguousChunks(void);
  void GPUAccessToCPUMemoryTest(void);
  void CPUAccessToGPUMemoryTest(void);
  void GPUAccessToGPUMemoryTest(void);

 private:
  void TestCreateDestroy(hsa_agent_t agent, hsa_amd_memory_pool_t pool);
  void TestRefCount(hsa_agent_t agent, hsa_amd_memory_pool_t pool);
  void TestPartialMapping(hsa_agent_t agent, hsa_amd_memory_pool_t pool);
  void NonContiguousChunks(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                           hsa_amd_memory_pool_t pool);

  void GPUAccessToCPUMemoryTest(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                                hsa_amd_memory_pool_t pool);
  void CPUAccessToGPUMemoryTest(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                                hsa_amd_memory_pool_t pool);
  void GPUAccessToGPUMemoryTest(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                                hsa_amd_memory_pool_t pool);
};

struct SharedVirtMem {
  std::atomic<int> token;
  std::atomic<int> count;
  std::atomic<size_t> size;
  std::atomic<int> child_status;
  std::atomic<int> parent_status;

  int sv[2];
};

class VirtMemoryTestInterProcess : public TestBase {
 public:
  VirtMemoryTestInterProcess();

  // @Brief: Destructor for test case of VirtMemoryTest
  virtual ~VirtMemoryTestInterProcess();

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

  void ParentProcessImpl();
  void ChildProcessImpl();


 private:
  int SendDmaBufFd(int socket, int dmabuf_fd);
  int ReceiveDmaBufFd(int socket);

  int child_;
  SharedVirtMem* shared_;
  bool parentProcess_;
  size_t min_gpu_mem_granule; /* Minimum granularity */
  size_t rec_gpu_mem_granule; /* Recommented granularity */
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_VIRTUAL_MEMORY_H_
