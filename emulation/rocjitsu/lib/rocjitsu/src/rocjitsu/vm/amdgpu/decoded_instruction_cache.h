// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
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
/// the stateless Instruction::execute contract.
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
  DecodeResult decode(Decoder &decoder, uint64_t pc, uint32_t vmid, Words words,
                      const DecodeErrorEmitter &emit_error, bool enabled = true) {
    if (enabled && !entries_)
      entries_ = std::make_unique<std::array<Entry, kNumEntries>>();
    if (enabled && entries_) {
      auto &entry = (*entries_)[index(pc)];
      if (entry.instruction && entry.pc == pc && entry.vmid == vmid &&
          std::equal(words.begin(), words.end(), entry.words.begin()))
        return std::move(entry.instruction);
    }
    Instruction::ScopedHeapAllocation heap_allocation;
    return decoder.decode(words.data(), emit_error);
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

    ~ScopedReturn() {
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

  /// @brief Drop entries before replacing the decoder/target, or destroying this CU.
  void clear() {
    Instruction::ScopedHeapAllocation heap_allocation;
    entries_.reset();
  }

private:
  struct Entry {
    uint64_t pc = 0;
    uint32_t vmid = 0;
    std::array<uint32_t, kNumWords> words{};
    std::unique_ptr<Instruction> instruction;
  };

  static size_t index(uint64_t pc) { return (pc / sizeof(uint32_t)) & (kNumEntries - 1); }

  static bool borrows_fetch_buffer(const Instruction &instruction, Words words) {
    const auto raw = reinterpret_cast<uintptr_t>(instruction.raw_encoding());
    const auto begin = reinterpret_cast<uintptr_t>(words.data());
    return raw >= begin && raw - begin < words.size_bytes();
  }

  void put(uint64_t pc, uint32_t vmid, Words words, std::unique_ptr<Instruction> &instruction) {
    // A decode miss allocates the table before execution; returning an
    // instruction during stack unwinding must not allocate.
    if (!entries_)
      return;
    auto &entry = (*entries_)[index(pc)];
    if (entry.instruction) {
      Instruction::ScopedHeapAllocation heap_allocation;
      entry.instruction.reset();
    }
    entry.pc = pc;
    entry.vmid = vmid;
    std::copy(words.begin(), words.end(), entry.words.begin());
    entry.instruction = std::move(instruction);
  }

  std::unique_ptr<std::array<Entry, kNumEntries>> entries_;
};

} // namespace rocjitsu::amdgpu
