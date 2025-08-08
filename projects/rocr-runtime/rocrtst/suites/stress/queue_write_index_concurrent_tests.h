/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
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
#ifndef ROCRTST_SUITES_STRESS_QUEUE_WRITE_INDEX_CONCURRENT_TESTS_H_
#define ROCRTST_SUITES_STRESS_QUEUE_WRITE_INDEX_CONCURRENT_TESTS_H_


#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"


class QueueWriteIndexConcurrentTest : public TestBase {
 public:
    QueueWriteIndexConcurrentTest(bool launch_Concurrent_AddWriteIndex,
                         bool launch_Concurrent_CasWriteIndex ,
                         bool launch_Concurrent_LoadStoreWriteIndex);

  // @Brief: Destructor for test case of MemoryTest
  virtual ~QueueWriteIndexConcurrentTest();

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


  // @Brief: Verifies that the hsa_queue_write_index_add operations is atomic,
  // and 'torn' adds do not occur when this API is executed concurrently.
  void QueueAddWriteIndexAtomic(void);

  // @Brief:  Verifies that the hsa_queue_cas_write_index operations is atomic,
  // and 'torn' compare and swaps do not occur when this API is executed
  // concurrently.
  void QueueCasWriteIndexAtomic(void);

  // @Brief: Verifies that the hsa_queue_write_index_load and store operations
  // are atomic, and 'torn' loads or stores do not occur when these APIs are executed
  // concurrently.
  void QueueLoadStoreWriteIndexAtomic(void);

 private:
  void QueueAddWriteIndexAtomic(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent);
  void QueueCasWriteIndexAtomic(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent);
  void QueueLoadStoreWriteIndexAtomic(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent);
};

#endif  // ROCRTST_SUITES_STRESS_QUEUE_WRITE_INDEX_CONCURRENT_TESTS_H_
