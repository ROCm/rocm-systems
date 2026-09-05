// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/hsa_code_object_reader_registry.h"

#include <mutex>
#include <new>
#include <utility>

namespace rocjitsu::hooks {

bool HsaCodeObjectReaderRegistry::store(uint64_t handle, const uint8_t *bytes, size_t size,
                                        std::shared_ptr<const std::vector<uint8_t>> owned) {
  std::unique_lock lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    auto *entry = static_cast<Entry *>(it.node_pointer());
    if (entry->handle == handle) {
      entry->bytes = bytes;
      entry->size = size;
      entry->owned = std::move(owned);
      return true;
    }
  }

  void *storage = entry_pool_.try_allocate(sizeof(Entry));
  if (storage == nullptr)
    return false;
  entries_.push_front(*new (storage) Entry(handle, bytes, size, std::move(owned)));
  return true;
}

HsaCodeObjectReaderRegistry::ReaderBytes HsaCodeObjectReaderRegistry::lookup(uint64_t handle) {
  std::shared_lock lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    auto *entry = static_cast<Entry *>(it.node_pointer());
    if (entry->handle == handle)
      return {entry->bytes, entry->size, entry->owned};
  }
  return {};
}

void HsaCodeObjectReaderRegistry::remove(uint64_t handle) {
  std::unique_lock lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    auto *entry = static_cast<Entry *>(it.node_pointer());
    if (entry->handle != handle)
      continue;
    entries_.erase(it);
    destroy_entry(entry);
    return;
  }
}

void HsaCodeObjectReaderRegistry::clear() {
  std::unique_lock lock(mutex_);
  while (!entries_.empty()) {
    auto it = entries_.begin();
    auto *entry = static_cast<Entry *>(it.node_pointer());
    entries_.erase(it);
    destroy_entry(entry);
  }
}

void HsaCodeObjectReaderRegistry::destroy_entry(Entry *entry) {
  entry->~Entry();
  entry_pool_.deallocate(entry);
}

} // namespace rocjitsu::hooks
