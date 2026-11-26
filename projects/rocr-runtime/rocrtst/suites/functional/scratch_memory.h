/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_SCRATCH_MEMORY_H_
#define ROCRTST_SUITES_FUNCTIONAL_SCRATCH_MEMORY_H_

#include <cstddef>
#include <cstdint>

#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

/// Functional tests for the scratch (private segment) memory that ROCr
/// manages for a compute queue. A dispatch whose private_segment_size is not
/// zero makes the command processor stop the queue until the runtime has
/// attached enough scratch memory to it. These tests dispatch with a range of
/// private segment sizes and grid sizes so that the runtime has to allocate,
/// grow, switch and release scratch, and check that every dispatch still
/// produces the right result.
class ScratchMemoryTest : public TestBase {
 public:
  ScratchMemoryTest();
  virtual ~ScratchMemoryTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo(void);

  /// @brief Dispatch with increasing private segment sizes on a queue that
  /// was created without scratch, so scratch is allocated and grown on demand.
  void DynamicScratchGrowthTest();

  /// @brief Dispatch with a grid that covers every scratch slot of the
  /// device, then with a small grid again, so that the runtime moves between
  /// the main and the alternate scratch allocation.
  void ScratchFullDeviceAndSmallGridTest();

  /// @brief Create queues with and without a private segment size hint and
  /// dispatch on each of them.
  void QueueCreateScratchHintTest();

  /// @brief Several queues on one GPU each get their own scratch, and
  /// dispatches from all of them are in flight together.
  void MultiQueueScratchTest();

  /// @brief Change the async scratch reclaim threshold with
  /// hsa_amd_agent_set_async_scratch_limit while queues hold scratch.
  void AsyncScratchLimitTest();

 private:
  // One dispatch of the 'square' kernel: its own output buffer, kernel
  // arguments and completion signal, so several can be in flight at once.
  struct Job {
    void* dst = nullptr;
    void* kernarg = nullptr;
    hsa_signal_t done = {0};
    size_t elements = 0;
  };

  // Allocates the buffers of @p job and writes the AQL packet to @p queue.
  bool Submit(Job* job, hsa_queue_t* queue, size_t elements, uint32_t private_segment_size,
              const char* label);

  // Waits for @p job, checks its output and releases its resources. With
  // @p dispatched false nothing was submitted, so it only releases resources.
  bool Finish(Job* job, const char* label, bool dispatched = true);

  // Submit() and Finish() one dispatch.
  bool DispatchSquare(hsa_queue_t* queue, size_t elements, uint32_t private_segment_size,
                      const char* label);

  // Number of compute units of the GPU under test.
  uint32_t ComputeUnits();

  // Grid size, in elements, that is large enough to use every scratch slot.
  size_t FullDeviceElements();

  // Create a queue on the GPU under test.
  hsa_status_t CreateQueue(uint32_t private_segment_size, hsa_queue_t** queue);

  // Allocates the input shared by all dispatches, for up to @p max_elements.
  void AllocateInput(size_t max_elements);
  void FreeInput();

  void* src_ = nullptr;
  size_t src_elements_ = 0;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_SCRATCH_MEMORY_H_
