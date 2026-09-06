/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_IPC_SDMA_H_
#define ROCRTST_SUITES_FUNCTIONAL_IPC_SDMA_H_

#include <sys/types.h>
#include <unistd.h>
#include <atomic>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

struct SharedIpcSdma {
  std::atomic<int> token;
  std::atomic<int> child_status;
  std::atomic<int> parent_status;
  std::atomic<size_t> size;
  hsa_amd_ipc_memory_t handle;
};

class IPCSDMATest : public TestBase {
 public:
  IPCSDMATest();
  virtual ~IPCSDMATest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  void ChildProcessImpl();
  void ParentProcessImpl();

 private:
  int child_;
  SharedIpcSdma* shared_;
  bool parentProcess_;
  size_t gpu_mem_granule_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_IPC_SDMA_H_
