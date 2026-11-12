/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SIGNAL_BATCH_TEST_H_
#define ROCRTST_SUITES_FUNCTIONAL_SIGNAL_BATCH_TEST_H_

#include <string>
#include <vector>

#include "suites/test_common/test_case_template.h"
#include "suites/test_common/test_base.h"

// Tests hsa_amd_signal_batch_create/destroy, HSA_AMD_MEMORY_POOL_INFO_SIGNAL_CAPABLE, the
// HSA_AMD_AGENT_INFO_PCIE_STORES_NEED_FLUSH / HOST_ATOMICS_SUPPORTED agent queries and the
// inline signal handle hint helpers.
class SignalBatchTest : public TestBase {
 public:
  SignalBatchTest();
  virtual ~SignalBatchTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  void HostPoolBatch();
  void DevicePoolsFollowQuery();
  void AsyncCopyCompletesMember();
  void NegativeArguments();
  void AgentQueries();
  void HandleHints();

 private:
  std::vector<std::string> resultList_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SIGNAL_BATCH_TEST_H_
