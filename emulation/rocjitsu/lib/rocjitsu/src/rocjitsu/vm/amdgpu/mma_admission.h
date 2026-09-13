// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/async_scoreboard.h"
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
  explicit MmaAdmissionCache(int mode, unsigned limit = 8) : mode_(mode), limit_(limit) {}
  bool applies(const Instruction &inst) const {
    return mode_ == 3 || inst.mnemonic() == "v_wmma_f32_16x16x32_f16" ||
           inst.mnemonic() == "v_wmma_f32_16x16x32_bf16";
  }
  bool observe_only() const { return mode_ == 1; }
  struct Counters {
    uint64_t decodes = 0, decode_hits = 0, plans = 0, hits = 0, validations = 0;
    uint64_t accept = 0, reject = 0, issuer = 0;
  } stats;
  void flush() {
    if (!(stats.accept + stats.reject + stats.issuer))
      return;
    std::fprintf(stderr,
                 "RJ_ADMISSION decodes=%llu decode_hits=%llu plans=%llu hits=%llu "
                 "validations=%llu accept=%llu reject=%llu issuer=%llu entries=%zu\n",
                 (unsigned long long)stats.decodes, (unsigned long long)stats.decode_hits,
                 (unsigned long long)stats.plans, (unsigned long long)stats.hits,
                 (unsigned long long)stats.validations, (unsigned long long)stats.accept,
                 (unsigned long long)stats.reject, (unsigned long long)stats.issuer,
                 decoded_.size());
    stats = {};
  }

  // Find an independent MMA to keep on the issuer. Lookahead never executes,
  // notifies observers, changes the PC, or emits a decode diagnostic.
  std::optional<uint64_t> inspect(Decoder &decoder, InstructionCache &icache,
                                  const GpuMemory &memory, uint64_t pc, uint32_t vmid,
                                  uint32_t num_vgprs, bool has_accvgprs, const Words &first) {
    auto [it, inserted] = plans_.try_emplace(Key{pc, vmid, num_vgprs, has_accvgprs});
    Plan &plan = it->second;
    bool valid = !inserted && plan.first == first;
    if (valid && plan.epoch != icache.epoch()) {
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
    if (plan.issuer_offset) {
      ++stats.accept;
      return pc + plan.issuer_offset;
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
  std::unordered_map<Words, std::unique_ptr<Decoded>, WordsHash> decoded_;
  std::unordered_map<Key, Plan, Hash> plans_;
};
} // namespace rocjitsu::amdgpu
