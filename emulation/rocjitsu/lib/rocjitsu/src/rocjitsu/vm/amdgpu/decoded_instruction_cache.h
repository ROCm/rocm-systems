// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Reuse completed instructions on one CU's issue path.
///
/// @details Fetchability and the byte I$ remain authoritative: lookups compare
/// all fetched words, including literals, so code maintenance and debugger
/// writes cannot reuse stale decodes.
/// Instructions that borrow the fetch buffer cannot be retained. Entries use
/// heap storage because a CU can run on different workers. Re-execution follows
/// the stateless Instruction::execute contract. The decoder consumes bytes, not
/// their PC: identical fetched windows can share a decode at different addresses.
/// PC-relative execution still reads the issuing wave's current PC.
/// Up to kNumEntries consecutive four-byte instructions occupy distinct primary
/// slots, so content-hash collisions cannot evict them from a contiguous loop.
/// A secondary content-to-slot hint permits reuse after relocation without
/// changing primary placement. Hint collisions only lose sharing opportunities;
/// both lookups validate the full fetched window and address-space namespace.
class DecodedInstructionCache {
public:
  // Bounded host-speed choice; direct indexing requires a power of two.
  static constexpr size_t kNumEntries = 256;
  static_assert(kNumEntries != 0 && (kNumEntries & (kNumEntries - 1)) == 0);
  // Match the four-word CU instruction-fetch window, including literals.
  static constexpr size_t kNumWords = 4;
  using Words = std::span<const uint32_t, kNumWords>;

  ~DecodedInstructionCache() { clear(); }

  /// @brief Decode from freshly fetched words, or take the matching cached object.
  ///
  /// @details The CU uses heap storage for every miss, including uncached
  /// memory operations and debugger execution. The allocation guard also
  /// bypasses an unrelated decoder pool installed by the caller.
  /// @param enabled False bypasses reuse and table allocation while debugging.
  [[gnu::always_inline]] DecodeResult decode(Decoder &decoder, uint64_t pc, uint32_t vmid,
                                             Words words, const DecodeErrorEmitter &emit_error,
                                             bool enabled = true) {
    if (enabled && entries_) {
      auto &entry = entries_->slots[pc_index(pc)];
      if (entry.instruction && entry.vmid == vmid &&
          std::equal(words.begin(), words.end(), entry.words.begin()))
        return std::move(entry.instruction);
      auto &shared = entries_->slots[entries_->content_slots[content_index(words)]];
      if (shared.instruction && shared.vmid == vmid &&
          std::equal(words.begin(), words.end(), shared.words.begin()))
        return std::move(shared.instruction);
    }
    return decode_miss(decoder, words, emit_error, enabled);
  }

  /// @brief Return only completed, stateless instructions.
  ///
  /// @details Construct immediately after decode() with the same key and policy.
  /// Memory pipelines and async execution take ownership by releasing/moving
  /// the caller's unique_ptr, so their in-flight instructions cannot become
  /// cache entries.
  class ScopedReturn {
  public:
    ScopedReturn(DecodedInstructionCache &cache, std::unique_ptr<Instruction> &instruction,
                 uint64_t pc, uint32_t vmid, Words words, bool enabled = true)
        : cache_(cache), instruction_(instruction), pc_(pc), vmid_(vmid), words_(words),
          enabled_(enabled), exceptions_(std::uncaught_exceptions()) {}

    [[gnu::always_inline]] ~ScopedReturn() {
      if (enabled_ && instruction_ && !instruction_->is_memory_op() && !instruction_->data() &&
          !borrows_fetch_buffer(*instruction_, words_) && std::uncaught_exceptions() == exceptions_)
        cache_.put(pc_, vmid_, words_, instruction_);
    }

    ScopedReturn(const ScopedReturn &) = delete;
    ScopedReturn &operator=(const ScopedReturn &) = delete;

    /// @brief Exclude a failed execution from reuse.
    void discard() { enabled_ = false; }

  private:
    DecodedInstructionCache &cache_;
    std::unique_ptr<Instruction> &instruction_;
    uint64_t pc_;
    uint32_t vmid_;
    Words words_;
    bool enabled_;
    int exceptions_;
  };

  /// @brief Return a memory instruction whose pipeline state was already cleared.
  ///
  /// @details Only an instruction that completed synchronously inside the caller's
  /// own issue may come back, with the key and policy it was decoded under. It
  /// must hold no dynamic state and no borrow of the fetch buffer; otherwise it
  /// is left in @p instruction for the caller to destroy.
  void put_completed(uint64_t pc, uint32_t vmid, Words words,
                     std::unique_ptr<Instruction> &instruction) {
    if (instruction && !instruction->data() && !borrows_fetch_buffer(*instruction, words))
      put(pc, vmid, words, instruction);
  }

  /// @brief Drop entries before replacing the decoder/target, or destroying this CU.
  void clear() {
    Instruction::ScopedHeapAllocation heap_allocation;
    entries_.reset();
  }

private:
  struct Entry {
    uint32_t vmid = 0;
    std::array<uint32_t, kNumWords> words{};
    std::unique_ptr<Instruction> instruction;
  };

  static_assert(kNumEntries <= 256, "Content hints store eight-bit slot indices");

  struct Table {
    std::array<Entry, kNumEntries> slots;
    std::array<uint8_t, kNumEntries> content_slots{};
  };

  static size_t pc_index(uint64_t pc) { return (pc / sizeof(uint32_t)) & (kNumEntries - 1); }

  static size_t content_index(Words words) {
    const uint64_t low = uint64_t{words[0]} | (uint64_t{words[1]} << 32);
    const uint64_t high = uint64_t{words[2]} | (uint64_t{words[3]} << 32);
    uint64_t hash = low ^ std::rotl(high, 21);
    hash ^= hash >> 32;
    hash *= 0xd6e8feb86659fd93ULL;
    hash ^= hash >> 32;
    return hash & (kNumEntries - 1);
  }

  // Keep allocation and decoder setup out of the per-instruction cache-hit path.
  RJ_NOINLINE DecodeResult decode_miss(Decoder &decoder, Words words,
                                       const DecodeErrorEmitter &emit_error, bool enabled) {
    if (enabled && !entries_)
      entries_ = std::make_unique<Table>();
    Instruction::ScopedHeapAllocation heap_allocation;
    return decoder.decode(words.data(), emit_error);
  }

  static bool borrows_fetch_buffer(const Instruction &instruction, Words words) {
    const auto raw = reinterpret_cast<uintptr_t>(instruction.raw_encoding());
    const auto begin = reinterpret_cast<uintptr_t>(words.data());
    return raw >= begin && raw - begin < words.size_bytes();
  }

  [[gnu::always_inline]] void put(uint64_t pc, uint32_t vmid, Words words,
                                  std::unique_ptr<Instruction> &instruction) {
    // A decode miss allocates the table before execution; returning an
    // instruction during stack unwinding must not allocate.
    if (!entries_)
      return;
    const auto slot = pc_index(pc);
    auto &entry = entries_->slots[slot];
    if (entry.instruction)
      evict(entry);
    entry.vmid = vmid;
    std::copy(words.begin(), words.end(), entry.words.begin());
    entry.instruction = std::move(instruction);
    entries_->content_slots[content_index(words)] = static_cast<uint8_t>(slot);
  }

  RJ_NOINLINE static void evict(Entry &entry) {
    Instruction::ScopedHeapAllocation heap_allocation;
    entry.instruction.reset();
  }

  std::unique_ptr<Table> entries_;
};

} // namespace rocjitsu::amdgpu
