//===-- LoadedKernelCache.cpp - Loaded kernel cache implementation --------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/LoadedKernelCache.h"

namespace aegisbit {

const LoadedKernel *LoadedKernelCache::find(const PatchCacheKey &Key) const {
  std::lock_guard<std::mutex> Lock(Mutex);
  auto It = Cache.find(Key);
  if (It == Cache.end())
    return nullptr;
  return &It->second;
}

const LoadedKernel *LoadedKernelCache::insert(const PatchCacheKey &Key,
                                              LoadedKernel Loaded) {
  std::lock_guard<std::mutex> Lock(Mutex);
  auto [It, Inserted] = Cache.emplace(Key, std::move(Loaded));
  (void)Inserted;
  return &It->second;
}

void LoadedKernelCache::clear() {
  std::lock_guard<std::mutex> Lock(Mutex);
  Cache.clear();
}

size_t LoadedKernelCache::size() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return Cache.size();
}

} // namespace aegisbit
