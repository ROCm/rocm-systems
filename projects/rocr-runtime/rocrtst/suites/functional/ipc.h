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

#ifndef ROCRTST_SUITES_FUNCTIONAL_IPC_H_
#define ROCRTST_SUITES_FUNCTIONAL_IPC_H_

#include <sys/types.h>
#include <unistd.h>
#include <atomic>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

struct Shared {
  std::atomic<int> token;
  std::atomic<int> count;
  std::atomic<size_t> size;
  std::atomic<int> child_status;
  std::atomic<int> parent_status;
  hsa_amd_ipc_memory_t handle;
  hsa_amd_ipc_signal_t signal_handle;
};

class IPCTest : public TestBase {
 public:
    IPCTest();

  // @Brief: Destructor for test case of TestExample
  virtual ~IPCTest();

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

  // @Brief: Implements child process exclusive logic
  void ChildProcessImpl();

  // @Brief: Implements parent process exclusive logic
  void ParentProcessImpl();

  // @Brief: Implements the check to see if buffer has expected
  // value if so updates it with new values
  void CheckAndFillBuffer(void* gpu_src_ptr, uint32_t exp_cur_val, uint32_t new_val);

 private:
  // @Brief: Bind number of iterations to run per user specification
  uint32_t RealIterationNum(void);

  // @Brief: Collect and print verbose messages to enable debugging
  void PrintVerboseMesg(void);

  // @Brief: Values used to initialize framebuffer that is shared
  uint32_t first_val_ = 0x01;
  uint32_t second_val_ = 0x02;
  uint32_t third_val_ = 0x03;
  uint32_t fourth_val_ = 0x04;
  uint32_t fifth_val_ = 0x05;

  int child_;
  Shared* shared_;
  bool parentProcess_;
  size_t gpu_mem_granule;

  // Supports user triggered failure
  int32_t usr_fail_val_ = 0xFFFFFFFF;

  // Specifies timeout period for parent/child processes
  int32_t timeout_ = 0x20000;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_IPC_H_
