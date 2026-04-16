//===-- aegisbit/PersistentBufferCache.h - Persistent buffer cache *- C++ -*===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Per-kernel cache of GPU-visible trace buffers used by the instrumented
/// path. Buffer addresses are baked into each kernel's trampoline, so we reuse
/// the same buffer across dispatches of the same kernel; the counter is reset
/// per dispatch by callers.
///
/// Extracted from TracingEngine. The cache does not free HSA memory on
/// `clear()` because `TracingEngine::finalize()` runs during __cxa_finalize,
/// when calling `hsa_amd_memory_pool_free` corrupts glibc heap metadata.
/// The OS reclaims GPU allocations at process exit.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_PERSISTENT_BUFFER_CACHE_H
#define AEGISBIT_PERSISTENT_BUFFER_CACHE_H

#include "aegisbit/Types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace aegisbit {

/// GPU-visible trace buffer that persists across dispatches of the same
/// kernel. Buffer and counter pointers are baked into the patched kernel's
/// trampoline.
struct PersistentTraceBuffer {
  void *BufferPtr = nullptr;  ///< GPU-visible trace record buffer
  void *CounterPtr = nullptr; ///< GPU-visible atomic counter
  size_t BufferSize = 0;
  TraceConfig Config; ///< Config passed to patcher
};

class PersistentBufferCache {
public:
  /// Callable that allocates a brand-new `PersistentTraceBuffer` for the
  /// given kernel name. Called only on cache miss. Must be safe to invoke
  /// while the cache's internal mutex is held — matching the historical
  /// `TracingEngine::getOrAllocPersistentBuffer` behavior.
  using Allocator =
      std::function<PersistentTraceBuffer(const std::string &KernelName)>;

  /// Get the persistent buffer for `(KernelName, VariantID)`, allocating via
  /// `Alloc` on miss.  `VariantID` selects among complementary patched-ELF
  /// variants built by the instrumentation-replay path; single-variant
  /// callers pass `0` (the overload without `VariantID` does this for them).
  ///
  /// Returns a stable reference: internally each variant is owned via
  /// `unique_ptr` so growing the per-kernel vector cannot invalidate prior
  /// pointers.  Stability matters because the trampoline bakes
  /// `Buffer.BufferPtr` / `Buffer.CounterPtr` as 64-bit immediates.
  PersistentTraceBuffer &getOrAlloc(const std::string &KernelName,
                                    uint32_t VariantID,
                                    const Allocator &Alloc);

  /// Single-variant convenience wrapper (VariantID=0).  Matches the
  /// pre-replay API so existing call sites keep compiling.
  PersistentTraceBuffer &getOrAlloc(const std::string &KernelName,
                                    const Allocator &Alloc) {
    return getOrAlloc(KernelName, 0u, Alloc);
  }

  /// Thread-safe lookup for a specific variant.  Returns nullptr if no
  /// entry exists for that `(KernelName, VariantID)` pair.
  const PersistentTraceBuffer *find(const std::string &KernelName,
                                    uint32_t VariantID) const;
  PersistentTraceBuffer *find(const std::string &KernelName,
                              uint32_t VariantID);

  /// Single-variant convenience overloads (VariantID=0).
  const PersistentTraceBuffer *find(const std::string &KernelName) const {
    return find(KernelName, 0u);
  }
  PersistentTraceBuffer *find(const std::string &KernelName) {
    return find(KernelName, 0u);
  }

  /// Number of variants currently cached under `KernelName` (0 if absent).
  size_t variantCount(const std::string &KernelName) const;

  /// Drop all entries without freeing HSA memory. Used during finalize where
  /// HSA calls are unsafe.
  void clear();

  /// Total entry count across all kernels and variants (for stats/tests).
  size_t size() const;

private:
  mutable std::mutex Mutex;
  // Per-kernel vector of variant-owning unique_ptrs; `unique_ptr` gives
  // pointer stability even when the vector grows past its initial capacity.
  std::unordered_map<std::string,
                     std::vector<std::unique_ptr<PersistentTraceBuffer>>>
      Buffers;
};

} // namespace aegisbit

#endif // AEGISBIT_PERSISTENT_BUFFER_CACHE_H
