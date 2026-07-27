// Minimal enqueue.h stub for CPU-only RCCL unit tests.
// Contains only the inline count functions used by EnqueueCountTests.
#pragma once

#include <cstddef>
#include "nccl_common.h"

#ifndef rccl_static
#define rccl_static static
#endif

static inline size_t ncclFuncSendCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncReduceScatter ? nRanks * count : count;
}
static inline size_t ncclFuncRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather ? nRanks * count : count;
}
rccl_static inline size_t ncclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather || func == ncclFuncReduceScatter ? nRanks * count : count;
}
