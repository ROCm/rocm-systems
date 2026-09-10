/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/misc/utils.cc, for the microtest binaries that do NOT
// compile the real utils.cc.
//
// Kept out of comm_fakes.cc for exactly that reason: the init and enqueue
// targets link the real hipified utils.cc as an oracle TU, so a fake definition
// there is a duplicate symbol at link time rather than an unused one.

#include <cstdlib>

#include "utils.h"

// Per-thread wait signal referenced by the inline MPSC-callback drain helpers
// in utils.h.
thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance = {};

// ncclMemoryStackConstruct is inline and leaves bumper == end == 0, so the first
// ncclMemoryStackAlloc on a constructed stack always lands here, and callers
// memset the result -- a nullptr-returning stub would fault.
//
// Simpler than production (src/misc/utils.cc): one fresh hunk per spill, no hunk
// reuse and no out-of-band Unhunk path. It keeps the invariant teardown relies
// on -- every hunk reachable from me->stub.above via ->above -- so Destruct below
// reclaims all of it.
void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align) {
  constexpr size_t kMinHunkSize = 64 << 10;  // matches production's growth step
  const size_t need = sizeof(struct Hunk) + align + size;
  const size_t hunkSize = need < kMinHunkSize ? kMinHunkSize : need;

  struct Hunk* hunk = static_cast<struct Hunk*>(malloc(hunkSize));
  if (hunk == nullptr) return nullptr;
  hunk->size = hunkSize;

  // Push onto the chain rooted at the stub rather than above the current top:
  // ordering is irrelevant to Destruct, and this stays correct on a stack that
  // was only zero-initialised (topFrame.hunk == nullptr).
  hunk->above = me->stub.above;
  me->stub.above = hunk;

  const uintptr_t obj =
    (reinterpret_cast<uintptr_t>(hunk) + sizeof(struct Hunk) + align - 1) & -uintptr_t(align);
  me->topFrame.hunk = hunk;
  me->topFrame.end = reinterpret_cast<uintptr_t>(hunk) + hunkSize;
  me->topFrame.bumper = obj + size;
  return reinterpret_cast<void*>(obj);
}

// Mirrors src/misc/utils.cc. Unhunks are freed first because in production the
// proxies live inside the hunks; this fake never makes one, but the loop is kept
// so a stack built elsewhere still tears down fully.
void ncclMemoryStackDestruct(struct ncclMemoryStack* me) {
  for (struct ncclMemoryStack::Frame* f = &me->topFrame; f != nullptr; f = f->below) {
    for (struct ncclMemoryStack::Unhunk* u = f->unhunks; u != nullptr; u = u->next) {
      free(u->obj);
    }
  }
  struct ncclMemoryStack::Hunk* h = me->stub.above;
  while (h != nullptr) {
    struct ncclMemoryStack::Hunk* above = h->above;
    free(h);
    h = above;
  }
  me->stub.above = nullptr;
}
