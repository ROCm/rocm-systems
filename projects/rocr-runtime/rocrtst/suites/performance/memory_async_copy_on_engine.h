/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates. 
 * 
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_PERFORMANCE_MEMORY_ASYNC_COPY_ON_ENGINE_H_
#define ROCRTST_SUITES_PERFORMANCE_MEMORY_ASYNC_COPY_ON_ENGINE_H_

#include <hwloc.h>

#include <vector>
#include <algorithm>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "suites/test_common/test_base.h"
#include "suites/performance/memory_async_copy.h"

class MemoryAsyncCopyOnEngine : public MemoryAsyncCopy {
 public:
  MemoryAsyncCopyOnEngine();

 protected:
  // @Brief: Run for Benchmark mode with verification
  virtual void RunBenchmarkWithVerification(Transaction *t);
};

#endif // ROCRTST_SUITES_PERFORMANCE_MEMORY_ASYNC_COPY_ON_ENGINE_H_
