//===-- PersistentBufferCache.cpp - Persistent buffer cache impl ----------===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/PersistentBufferCache.h"

namespace aegisbit {

PersistentTraceBuffer &
PersistentBufferCache::getOrAlloc(const std::string &KernelName,
                                  uint32_t VariantID, const Allocator &Alloc) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto &Vec = Buffers[KernelName];
  if (VariantID < Vec.size() && Vec[VariantID])
    return *Vec[VariantID];

  // Cache miss for this variant.  Grow the vector if needed and allocate
  // the new buffer while still holding the mutex — this matches the
  // historical single-variant behavior of
  // `TracingEngine::getOrAllocPersistentBuffer` and serializes concurrent
  // first-use for the same `(name, variant)` pair.
  if (VariantID >= Vec.size())
    Vec.resize(static_cast<size_t>(VariantID) + 1);
  Vec[VariantID] =
      std::make_unique<PersistentTraceBuffer>(Alloc(KernelName));
  return *Vec[VariantID];
}

const PersistentTraceBuffer *
PersistentBufferCache::find(const std::string &KernelName,
                            uint32_t VariantID) const {
  std::lock_guard<std::mutex> Lock(Mutex);
  auto It = Buffers.find(KernelName);
  if (It == Buffers.end() || VariantID >= It->second.size())
    return nullptr;
  const auto &Slot = It->second[VariantID];
  return Slot ? Slot.get() : nullptr;
}

PersistentTraceBuffer *
PersistentBufferCache::find(const std::string &KernelName,
                            uint32_t VariantID) {
  std::lock_guard<std::mutex> Lock(Mutex);
  auto It = Buffers.find(KernelName);
  if (It == Buffers.end() || VariantID >= It->second.size())
    return nullptr;
  auto &Slot = It->second[VariantID];
  return Slot ? Slot.get() : nullptr;
}

size_t
PersistentBufferCache::variantCount(const std::string &KernelName) const {
  std::lock_guard<std::mutex> Lock(Mutex);
  auto It = Buffers.find(KernelName);
  if (It == Buffers.end())
    return 0;
  return It->second.size();
}

void PersistentBufferCache::clear() {
  std::lock_guard<std::mutex> Lock(Mutex);
  Buffers.clear();
}

size_t PersistentBufferCache::size() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  size_t Total = 0;
  for (const auto &[_, Vec] : Buffers)
    for (const auto &P : Vec)
      if (P)
        ++Total;
  return Total;
}

} // namespace aegisbit
