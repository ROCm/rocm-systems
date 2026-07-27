// Minimal comm.h stub for CPU-only RCCL unit tests.
// Shadows the real comm.h (which pulls in the entire HIP/GPU header chain).
// Contains only declarations needed by: MiscTests, TimeoutTests.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>

#include "nccl.h"
#include "nccl_common.h"
#include "bitops.h"

// From real comm.h line 213
struct ncclTaskColl {
  struct ncclTaskColl* next;
  ncclFunc_t func;
  void const* sendbuff;
  void* recvbuff;
  size_t count;
};

// From real comm.h lines 392-410
struct ncclTaskCollSorter {
  static constexpr int UnitLog2 = 10;
  static constexpr size_t UnitSize = 1 << UnitLog2;
  static constexpr int MaxLog2 = 30;
  static constexpr size_t MaxSize = 1ull << MaxLog2;
  static constexpr int BitsPerPow2 = 2;
  static constexpr int BinsPerPow2 = 1 << BitsPerPow2;
  static constexpr int BinCount = 1 + (MaxLog2 - UnitLog2) * BinsPerPow2;

  struct ncclTaskColl* head;
  struct ncclTaskColl* tail;
  int binEdge;
  struct ncclTaskColl** bins[BinCount];
};

// From real comm.h lines 412-440
inline void ncclTaskCollSorterInsert(struct ncclTaskCollSorter* me, struct ncclTaskColl* x, size_t size) {
  constexpr int UnitLog2 = ncclTaskCollSorter::UnitLog2;
  constexpr size_t MaxSize = ncclTaskCollSorter::MaxSize;
  constexpr int BitsPerPow2 = ncclTaskCollSorter::BitsPerPow2;
  constexpr int BinCount = ncclTaskCollSorter::BinCount;
  int bin = u32fpEncode(static_cast<uint32_t>(std::min(MaxSize, size) >> UnitLog2), BitsPerPow2);
  bin = BinCount - 1 - bin;

  if (me->bins[bin] == nullptr) {
    if (me->binEdge <= bin) {
      me->binEdge = bin + 1;
      me->bins[bin] = me->tail ? &me->tail->next : &me->head;
      me->tail = x;
    } else {
      int succ = bin + 1;
      while (me->bins[succ] == nullptr) succ++;
      me->bins[bin] = me->bins[succ];
      me->bins[succ] = &x->next;
    }
  }
  x->next = *me->bins[bin];
  *me->bins[bin] = x;
}

// From real comm.h lines 442-444
inline bool ncclTaskCollSorterEmpty(struct ncclTaskCollSorter* me) {
  return me->head == nullptr;
}

// From real comm.h lines 447-451
inline struct ncclTaskColl* ncclTaskCollSorterDequeueAll(struct ncclTaskCollSorter* me) {
  struct ncclTaskColl* head = me->head;
  if (head != nullptr) memset(me, 0, sizeof(*me));
  return head;
}

// From real comm.h line 538
#define NCCL_MAGIC 0x0280028002800280

// Minimal ncclComm struct — only fields used by TimeoutTests.
// The real struct is ~400 lines; we only need magic sentinels + asyncResult.
struct ncclComm {
  uint64_t startMagic;
  ncclResult_t asyncResult;
  uint64_t endMagic;
};

// From real comm.h line 1063
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState);
