// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "types.h"

namespace hazard_core {

template <class T> inline void hash_combine(std::size_t &seed, const T &value) {
  std::hash<T> hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct ExecutionKey {
  EntityId dispatch_id = 0;
  EntityId cluster_id = 0;
  EntityId workgroup_id = 0;
  EntityId wavegroup_id = 0;
  EntityId wave_id = 0;

  bool operator==(const ExecutionKey &o) const noexcept {
    return dispatch_id == o.dispatch_id && cluster_id == o.cluster_id &&
           workgroup_id == o.workgroup_id && wavegroup_id == o.wavegroup_id && wave_id == o.wave_id;
  }
};

struct ExecutionKeyHash {
  size_t operator()(const ExecutionKey &k) const noexcept {
    size_t seed = 0;
    hash_combine(seed, k.dispatch_id);
    hash_combine(seed, k.cluster_id);
    hash_combine(seed, k.workgroup_id);
    hash_combine(seed, k.wavegroup_id);
    hash_combine(seed, k.wave_id);
    return seed;
  }
};

struct InstructionKey {
  ExecutionKey execution;
  EntityId instruction_id = 0;

  bool operator==(const InstructionKey &other) const noexcept {
    return execution == other.execution && instruction_id == other.instruction_id;
  }
};

struct InstructionKeyHash {
  size_t operator()(const InstructionKey &key) const noexcept {
    ExecutionKeyHash execution_hash;
    size_t seed = execution_hash(key.execution);
    hash_combine(seed, key.instruction_id);
    return seed;
  }
};

} // namespace hazard_core
