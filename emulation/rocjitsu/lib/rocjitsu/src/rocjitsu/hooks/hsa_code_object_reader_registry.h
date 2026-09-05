// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/arena_alloc.h"
#include "util/intrusive_list.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace rocjitsu::hooks {

/// Process-local map from opaque HSA code-object reader handles to ELF bytes.
///
/// Application-created memory readers remain non-owning. Readers created by a
/// hook may retain shared storage so lookup snapshots keep their bytes alive
/// across a concurrent reader destruction.
class HsaCodeObjectReaderRegistry {
public:
  struct ReaderBytes {
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> owned;

    [[nodiscard]] explicit operator bool() const { return bytes != nullptr; }
  };

  [[nodiscard]] bool store(uint64_t handle, const uint8_t *bytes, size_t size,
                           std::shared_ptr<const std::vector<uint8_t>> owned = {});
  [[nodiscard]] ReaderBytes lookup(uint64_t handle);
  void remove(uint64_t handle);
  void clear();

private:
  struct Entry : util::IListNode<Entry> {
    Entry(uint64_t handle, const uint8_t *bytes, size_t size,
          std::shared_ptr<const std::vector<uint8_t>> owned)
        : handle(handle), bytes(bytes), size(size), owned(std::move(owned)) {}

    uint64_t handle = 0;
    const uint8_t *bytes = nullptr;
    size_t size = 0;
    std::shared_ptr<const std::vector<uint8_t>> owned;
  };

  void destroy_entry(Entry *entry);

  mutable std::shared_mutex mutex_;
  util::ArenaAlloc<sizeof(Entry), 256, alignof(Entry)> entry_pool_;
  util::IntrusiveList<Entry> entries_;
};

} // namespace rocjitsu::hooks
