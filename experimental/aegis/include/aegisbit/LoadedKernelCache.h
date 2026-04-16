//===-- aegisbit/LoadedKernelCache.h - Loaded kernel cache ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Thread-safe cache of `(PatchCacheKey -> LoadedKernel)` used by the tracing
/// engine to avoid re-loading the same patched ELF into the HSA runtime for
/// every dispatch. Extracted from TracingEngine.
///
/// Note: `LoadedKernel` is a plain value (handles only) with no RAII, so
/// clearing this cache does **not** call into the HSA runtime. That is
/// important because `TracingEngine::finalize()` runs during __cxa_finalize,
/// when HSA internals may be partially torn down.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_LOADED_KERNEL_CACHE_H
#define AEGISBIT_LOADED_KERNEL_CACHE_H

#include "aegisbit/KernelLauncher.h"
#include "aegisbit/KernelPatcher.h"

#include <mutex>
#include <unordered_map>

namespace aegisbit {

class LoadedKernelCache {
public:
  /// Look up a cached kernel. Returns nullptr if not present.
  const LoadedKernel *find(const PatchCacheKey &Key) const;

  /// Insert a loaded kernel into the cache. If an entry for `Key` already
  /// exists, the existing entry is kept and a pointer to it is returned.
  /// Otherwise the new entry is inserted and a pointer to the stored value
  /// is returned.
  const LoadedKernel *insert(const PatchCacheKey &Key, LoadedKernel Loaded);

  /// Drop all cache entries. Safe to call during shutdown.
  void clear();

  /// Current entry count (for stats/tests).
  size_t size() const;

private:
  mutable std::mutex Mutex;
  std::unordered_map<PatchCacheKey, LoadedKernel> Cache;
};

} // namespace aegisbit

#endif // AEGISBIT_LOADED_KERNEL_CACHE_H
