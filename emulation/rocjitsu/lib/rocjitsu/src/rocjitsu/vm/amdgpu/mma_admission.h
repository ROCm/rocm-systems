// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mma_admission.h
/// @brief Bounded decoded lookahead for profitable asynchronous MMA issue.

#pragma once

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_cache.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <format>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace rocjitsu::amdgpu {

/// @brief Bounded per-CU cache for speculative MMA admission.
/// @details Plans retain code snapshots and an issuer offset, never decoded
/// instructions. Temporary decodes are destroyed before inspection returns.
class MmaAdmissionCache {
public:
  using Words = std::array<uint32_t, 4>;
  struct Counters {
    uint64_t decodes = 0;
    uint64_t plans = 0;
    uint64_t hits = 0;
    uint64_t validations = 0;
    uint64_t accept = 0;
    uint64_t reject = 0;
    uint64_t issuer = 0;
    uint64_t evictions = 0;
  };
  explicit MmaAdmissionCache(unsigned limit = 8, size_t capacity = 16384)
      : limit_(std::clamp(limit, 1u, 16u)), capacity_(std::max(size_t{1}, capacity)) {}
  size_t size() const { return plans_.size(); }
  void flush() {
    if (!(stats.accept + stats.reject + stats.issuer))
      return;
    util::Logger::vm([&](auto &os) {
      os << std::format("RJ_ADMISSION decodes={} plans={} hits={} validations={} "
                        "accept={} reject={} issuer={} entries={} evictions={}",
                        stats.decodes, stats.plans, stats.hits, stats.validations, stats.accept,
                        stats.reject, stats.issuer, plans_.size(), stats.evictions);
    });
    stats = {};
  }

  /// @brief Find an independent MMA to keep on the issuer.
  /// @details Lookahead never executes, notifies observers, changes the PC,
  /// or emits a decode diagnostic.
  std::optional<uint64_t> inspect(Decoder &decoder, InstructionCache &icache,
                                  const GpuMemory &memory, uint64_t pc, uint32_t vmid,
                                  uint32_t num_vgprs, bool has_accvgprs, const Words &first) {
    return inspect(decoder, icache, memory, nullptr, pc, vmid, num_vgprs, has_accvgprs, first);
  }

  /// @brief Inspect through the same retained VM snapshot as ordinary issue.
  std::optional<uint64_t> inspect(Decoder &decoder, InstructionCache &icache,
                                  const GpuMemory &memory, const GpuVmAccess *vm_access,
                                  uint64_t pc, uint32_t vmid, uint32_t num_vgprs, bool has_accvgprs,
                                  const Words &first) {
    const VmCacheNamespace cache_namespace =
        vm_access ? vm_access->cache_namespace() : VmCacheNamespace{};
    const Key key{pc, vmid, num_vgprs, has_accvgprs, cache_namespace};
    const auto fetch_words = [&](uint64_t address, Words &words) {
      if (vm_access)
        return icache.fetch(*vm_access, address, reinterpret_cast<uint8_t *>(words.data())) ==
               VmAccessOutcome::Complete;
      icache.fetch(memory, address, reinterpret_cast<uint8_t *>(words.data()));
      return true;
    };
    auto it = plans_.find(key);
    bool valid = it != plans_.end() && it->second.first == first;
    if (valid && it->second.epoch != icache.epoch()) {
      auto &plan = it->second;
      ++stats.validations;
      // I$ invalidation rechecks bytes. Unchanged code retains successful and
      // rejected plans across dispatches without decoding again.
      for (const auto &snapshot : plan.lookahead) {
        Words words;
        if (!fetch_words(pc + snapshot.offset, words) || words != snapshot.words) {
          valid = false;
          break;
        }
      }
      plan.epoch = icache.epoch();
    }
    if (!valid) {
      // Cache hits and replacement of an existing key need no eviction.
      if (it == plans_.end() && plans_.size() == capacity_) {
        plans_.clear();
        it = plans_.end();
        ++stats.evictions;
      }
      if (it == plans_.end())
        it = plans_.try_emplace(key).first;
      auto &plan = it->second;
      plan = {};
      plan.first = first;
      plan.epoch = icache.epoch();
      ++stats.plans;
      auto initial = decode(decoder, first);
      const auto initial_access =
          initial ? async_execution::footprint(*initial, num_vgprs, has_accvgprs) : std::nullopt;
      if (initial_access) {
        auto pending = *initial_access;
        uint64_t offset = initial->size();
        for (unsigned count = 0; count != limit_; ++count) {
          // Stay in the already fetchable code page. Failure rejects speculation
          // without changing the ordinary execution fault boundary.
          if ((pc % GpuMemory::PAGE_SIZE) + offset + sizeof(Words) > GpuMemory::PAGE_SIZE)
            break;
          Words words;
          if (!fetch_words(pc + offset, words))
            break;
          plan.lookahead.push_back({offset, words});
          auto next = decode(decoder, words);
          if (!next || !async_execution::safe_inline(*next))
            break;
          const auto access = async_execution::footprint(*next, num_vgprs, has_accvgprs);
          if (!access || pending.conflicts(*access))
            break;
          if (matrix_coexecution::async_candidate(next->mnemonic())) {
            plan.issuer_offset = offset;
            // Preserve wider independent groups: offload their prefix and
            // reserve the final MMA for useful work on the issuing thread.
            pending.reads |= access->reads;
            pending.writes |= access->writes;
          }
          offset += next->size();
        }
      }
    } else {
      ++stats.hits;
    }
    if (it->second.issuer_offset) {
      ++stats.accept;
      return pc + it->second.issuer_offset;
    }
    ++stats.reject;
    return std::nullopt;
  }

  Counters stats;

private:
  struct Key {
    uint64_t pc;
    uint32_t vmid, vgprs;
    bool acc;
    VmCacheNamespace cache_namespace;
    bool operator==(const Key &) const = default;
  };
  struct Hash {
    size_t operator()(const Key &key) const {
      size_t hash = std::hash<uint64_t>{}(key.pc ^ (uint64_t{key.vmid} << 32));
      hash ^= size_t{key.vgprs} << 1;
      hash ^= key.acc;
      hash ^= std::hash<uint64_t>{}(key.cache_namespace.address_space.generation);
      hash ^= size_t{key.cache_namespace.address_space.slot} << 2;
      hash ^= std::hash<uint64_t>{}(key.cache_namespace.translation_epoch);
      return hash;
    }
  };
  struct Snapshot {
    uint64_t offset;
    Words words;
  };
  struct Plan {
    uint64_t epoch = 0, issuer_offset = 0;
    Words first{};
    std::vector<Snapshot> lookahead;
  };
  std::unique_ptr<Instruction> decode(Decoder &decoder, const Words &words) {
    ++stats.decodes;
    auto result = decoder.decode(words.data());
    return result.succeeded() ? std::move(result).value() : nullptr;
  }
  unsigned limit_;
  size_t capacity_;
  std::unordered_map<Key, Plan, Hash> plans_;
};
} // namespace rocjitsu::amdgpu
