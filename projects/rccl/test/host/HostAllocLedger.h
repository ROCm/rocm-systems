/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tracks host memory a unit allocates with ncclCalloc and frees with free(),
// through macro shims the test defines around its #include of the unit:
//
//   #define ncclCalloc(...) g_hostAllocs.Calloc(__VA_ARGS__, __FILE__, __LINE__, __func__)
//   #define free(ptr) g_hostAllocs.Free(ptr)
//
// Clean() is true when everything was freed and nothing was refused.

#ifndef RCCL_TEST_HOST_HOSTALLOCLEDGER_H_
#define RCCL_TEST_HOST_HOSTALLOCLEDGER_H_

#include <cstdlib>
#include <map>
#include <string>

#include "HipVmmLedger.h"
#include "alloc.h"

class HostAllocLedger {
 public:
  template <typename T>
  ncclResult_t Calloc(T** ptr, size_t nelem, const char* file, int line, const char* fn) {
    const size_t bytes = nelem * sizeof(T);
    if (failBytes != 0 && bytes == failBytes) {
      failBytes = 0;
      return ncclSystemError;
    }
    ncclResult_t ret = ncclCallocDebug(ptr, nelem, file, line, fn, true);
    if (ret == ncclSuccess && *ptr != nullptr) live[*ptr] = bytes;
    return ret;
  }

  void Free(void* ptr) {
    if (ptr != nullptr && live.erase(ptr) == 0) {
      ++badFrees;
      return;
    }
    std::free(ptr);
  }

  bool Covers(const void* dst, size_t n) const { return RangeCovers(live, dst, n); }
  template <typename CopyFn>
  auto GuardHostCopies(CopyFn copy) {
    return [this, copy](void* dst, const void* src, size_t n, hipMemcpyKind kind) mutable {
      if (kind == hipMemcpyHostToHost && !Covers(dst, n)) {
        ++refusedCopies;
        return hipErrorInvalidValue;
      }
      return copy(dst, src, n, kind);
    };
  }
  bool Clean() const { return live.empty() && badFrees == 0 && refusedCopies == 0; }
  std::string Summary() const {
    return std::to_string(live.size()) + " host allocations live; " + std::to_string(badFrees) + " bad frees; " +
           std::to_string(refusedCopies) + " host copies refused";
  }
  void Reset() {
    live.clear();
    badFrees = 0;
    refusedCopies = 0;
    failBytes = 0;
  }

  std::map<void*, size_t> live;  // live allocations -> size
  int badFrees = 0;
  int refusedCopies = 0;  // bumped by GuardHostCopies when a destination is outside `live`
  size_t failBytes = 0;   // fail the next ncclCalloc of exactly this many bytes
};

inline HostAllocLedger g_hostAllocs;

#endif  // RCCL_TEST_HOST_HOSTALLOCLEDGER_H_
