// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/instruction_cache.h"
#include "util/log.h"
#include <unordered_map>

namespace rocjitsu::amdgpu {

// Read-only decode objects belong to the optional CU adapter. Executing
// instructions retain their separate per-issue state and pipeline ownership.
class MmaAdmissionCache {
public:
  using Words = std::array<uint32_t, 4>;
  static int configured_mode() {
    static const int value = [] {
      const char *text = std::getenv("RJ_MMA_ADMISSION");
      return std::clamp(text ? std::atoi(text) : 0, 0, 3);
    }();
    return value;
  }
  static unsigned configured_limit() {
    static const unsigned value = [] {
      const char *text = std::getenv("RJ_MMA_LOOKAHEAD");
      return unsigned(std::clamp(text ? std::atoi(text) : 8, 1, 16));
    }();
    return value;
  }
  /// Retain bounded per-CU code history; pressure discards both caches at entry.
  struct Capacity {
    size_t plans = 16384;
    size_t decodes = 65536;
  };
  explicit MmaAdmissionCache(int mode, unsigned limit = 8)
      : MmaAdmissionCache(mode, limit, Capacity{}) {}
  MmaAdmissionCache(int mode, unsigned limit, Capacity capacity)
      : mode_(mode), limit_(std::clamp(limit, 1u, 16u)), capacity_(capacity) {
    capacity_.plans = std::max(size_t{1}, capacity_.plans);
    // One inspection may decode its initial instruction and the entire window.
    capacity_.decodes = std::max(size_t{limit_} + 1, capacity_.decodes);
  }
  /// Number of retained plans and unique decoding snapshots, respectively.
  Capacity size() const { return {plans_.size(), decoded_.size()}; }
  bool applies(const Instruction &inst) const {
    return async_mma_policy::needs_admission(mode_, inst.mnemonic());
  }
  bool observe_only() const { return mode_ == 1; }
  struct Counters {
    uint64_t decodes = 0, decode_hits = 0, plans = 0, hits = 0, validations = 0;
    uint64_t accept = 0, reject = 0, issuer = 0, evictions = 0;
  } stats;
  void flush() {
    if (!(stats.accept + stats.reject + stats.issuer))
      return;
    util::Logger::warn(
        std::format("RJ_ADMISSION decodes={} decode_hits={} plans={} hits={} validations={} "
                    "accept={} reject={} issuer={} entries={} evictions={}",
                    stats.decodes, stats.decode_hits, stats.plans, stats.hits, stats.validations,
                    stats.accept, stats.reject, stats.issuer, decoded_.size(), stats.evictions));
    stats = {};
  }

  // Find an independent MMA to keep on the issuer. Lookahead never executes,
  // notifies observers, changes the PC, or emits a decode diagnostic.
  std::optional<uint64_t> inspect(Decoder &decoder, InstructionCache &icache,
                                  const GpuMemory &memory, uint64_t pc, uint32_t vmid,
                                  uint32_t num_vgprs, bool has_accvgprs, const Words &first) {
    const Key key{pc, vmid, num_vgprs, has_accvgprs};
    auto it = plans_.find(key);
    bool valid = it != plans_.end() && it->second.first == first;
    if (valid && it->second.epoch != icache.epoch()) {
      auto &plan = it->second;
      ++stats.validations;
      // I$ invalidation rechecks bytes. Unchanged code retains its decoded
      // objects and both successful and rejected plans across dispatches.
      for (const auto &snapshot : plan.lookahead) {
        Words words;
        icache.fetch(memory, pc + snapshot.offset, vmid, reinterpret_cast<uint8_t *>(words.data()));
        if (words != snapshot.words) {
          valid = false;
          break;
        }
      }
      plan.epoch = icache.epoch();
    }
    if (!valid) {
      // Reserve for the worst-case scan before retaining references or decoded
      // pointers. Cache hits need no capacity check or eviction bookkeeping.
      if ((it == plans_.end() && plans_.size() == capacity_.plans) ||
          decoded_.size() > capacity_.decodes - (size_t{limit_} + 1)) {
        plans_.clear();
        decoded_.clear();
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
      const Instruction *initial = decode(decoder, first);
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
          icache.fetch(memory, pc + offset, vmid, reinterpret_cast<uint8_t *>(words.data()));
          plan.lookahead.push_back({offset, words});
          const Instruction *next = decode(decoder, words);
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

private:
  struct Key {
    uint64_t pc;
    uint32_t vmid, vgprs;
    bool acc;
    bool operator==(const Key &) const = default;
  };
  struct Hash {
    size_t operator()(const Key &key) const {
      return std::hash<uint64_t>{}(key.pc ^ (uint64_t{key.vmid} << 32)) ^ (size_t{key.vgprs} << 1) ^
             key.acc;
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
  struct Decoded {
    // raw_encoding() can point here. Preserve the backing bytes' lifetime.
    Words words;
    std::unique_ptr<Instruction> inst;
    ~Decoded() {
      Instruction::ScopedHeapAllocation heap;
      inst.reset();
    }
  };
  struct WordsHash {
    size_t operator()(const Words &words) const {
      size_t hash = 0;
      for (uint32_t word : words)
        hash ^= size_t{word} + 0x9e3779b9u + (hash << 6) + (hash >> 2);
      return hash;
    }
  };
  const Instruction *decode(Decoder &decoder, const Words &words) {
    auto it = decoded_.find(words);
    if (it == decoded_.end()) {
      auto entry = std::make_unique<Decoded>();
      entry->words = words;
      Instruction::ScopedHeapAllocation heap;
      auto result = decoder.decode(entry->words.data());
      if (result.succeeded())
        entry->inst = std::move(result).value();
      it = decoded_.emplace(words, std::move(entry)).first;
      ++stats.decodes;
    } else {
      ++stats.decode_hits;
    }
    return it->second->inst.get();
  }
  int mode_;
  unsigned limit_;
  Capacity capacity_;
  std::unordered_map<Words, std::unique_ptr<Decoded>, WordsHash> decoded_;
  std::unordered_map<Key, Plan, Hash> plans_;
};
} // namespace rocjitsu::amdgpu
