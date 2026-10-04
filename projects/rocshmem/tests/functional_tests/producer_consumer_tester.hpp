/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef _PRODUCER_CONSUMER_TESTER_HPP_
#define _PRODUCER_CONSUMER_TESTER_HPP_

#include "tester.hpp"

/* Producer-consumer signaling round-trip exercising the common pattern
 *   put -> fence -> atomic_set -> wait_until -> load
 * in two modes selected by the ROCSHMEM_PC_RELAXED environment variable:
 *   0 (default) = standard ordering  (release fence + L2 flush, acquire + L2 inval)
 *   1           = targeted ordering  (cache-bypassing put, waitcnt-only fence,
 *                                     relaxed system-scope atomic_set, uncached load)
 * The relaxed mode avoids the L2 flush/invalidate that the standard ordering
 * requires, which is the performance benefit measured by the heatmap. */
class ProducerConsumerTester : public Tester {
 public:
  explicit ProducerConsumerTester(TesterArguments args);
  virtual ~ProducerConsumerTester();

 protected:
  virtual void resetBuffers(uint64_t size) override;
  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) override;
  virtual void verifyResults(uint64_t size) override;

 private:
  bool relaxed_{false};
  char *src_buf{nullptr};   // local source payload (all 'a')
  char *dst_buf{nullptr};   // symmetric destination (written by peer)
  uint64_t *flag{nullptr};  // symmetric completion flag (set by peer)
};

#endif /* _PRODUCER_CONSUMER_TESTER_HPP_ */
