/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_MEMORY_ASYNC_BATCH_COPY_H_
#define ROCRTST_SUITES_FUNCTIONAL_MEMORY_ASYNC_BATCH_COPY_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// Functional tests for hsa_amd_memory_async_batch_copy. They mirror what is
// checked for hsa_amd_memory_async_copy: data lands where it should, nothing
// past the requested size is touched, the completion signal reaches exactly
// zero, dependent signals hold the copy back, and bad arguments are rejected
// without starting any work.
//
// Linear copies run on every platform. The other operation types need SDMA and
// some of them a particular gfx target; those tests skip the data checks where
// the runtime is expected to refuse the operation, and fail when the runtime
// refuses an operation the target supports.
class MemoryAsyncBatchCopy : public TestBase {
 public:
  MemoryAsyncBatchCopy();
  virtual ~MemoryAsyncBatchCopy();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  // HSA_AMD_MEMORY_COPY_OP_LINEAR, one copy per op, several ops per batch.
  void LinearSingleTest();
  // HSA_AMD_MEMORY_COPY_OP_LINEAR with num_entries > 0.
  void LinearMultiTest();
  // LINEAR between two GPUs. Skips on single-GPU systems.
  void LinearPeerTest();
  // HSA_AMD_MEMORY_COPY_OP_LINEAR_BROADCAST.
  void BroadcastTest();
  // HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP.
  void SwapTest();
  // HSA_AMD_MEMORY_COPY_OP_RECT.
  void RectTest();
  // HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_{SRC,DST,SRCDST}.
  void IndirectTest();
  // num_dep_signals / dep_signals hold back every op of the batch.
  void DependencyTest();
  // Several operation types in one call sharing one completion signal.
  void MixedOpsTest();
  // Invalid arguments are rejected and no copy is started.
  void NegativeTest();
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_MEMORY_ASYNC_BATCH_COPY_H_
