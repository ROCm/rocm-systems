// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_file.h
/// @brief Physical register file with block-granularity allocation tracking.

#ifndef SIMDOJO_COMPONENTS_REGISTER_FILE_H_
#define SIMDOJO_COMPONENTS_REGISTER_FILE_H_

#include "simdojo/sim/component.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace simdojo {

/// @brief Physical register file with block-granularity allocation tracking.
///
/// @details Templated on the register type: use uint32_t for scalar files or
/// VectorReg<NumElems, Elem> for vector files. The file is divided into
/// fixed-size blocks (one per hardware context slot). Allocation finds a
/// free block and returns its base register index.
///
/// This component is not thread-safe. Const reads may materialize memoized
/// zero-initialized storage and therefore must run on the owning simulation
/// partition's engine thread, just like mutable accesses.
///
/// @tparam RegType Register element type (default: uint32_t).
template <typename RegType = uint32_t> class RegisterFile : public Component {
public:
  explicit RegisterFile(std::string name) : Component(std::move(name)) {}

  /// @brief Initialize the register file.
  /// @param total_regs Total number of registers in the file.
  /// @param regs_per_block Registers per allocation block (granularity).
  void init(uint32_t total_regs, uint32_t regs_per_block) {
    assert(total_regs_ == 0 && "RegisterFile already initialized");
    total_regs_ = total_regs;
    regs_per_block_ = regs_per_block;
    data_.assign(total_regs, RegType{});
    // data_ was just value-initialized. Treat it as materialized until a block
    // allocation marks that block lazy for a subsequent occupant.
    initialized_.assign(total_regs, 1);
    uint32_t num_blocks = (regs_per_block > 0) ? (total_regs / regs_per_block) : 0;
    free_block_bits_.assign((num_blocks + 63u) / 64u, ~uint64_t{0});
    if (!free_block_bits_.empty() && num_blocks % 64u != 0)
      free_block_bits_.back() = (uint64_t{1} << (num_blocks % 64u)) - 1u;
    free_block_count_ = num_blocks;
  }

  /// @brief Try to allocate a contiguous block of registers.
  /// @param count Number of registers needed (must be <= regs_per_block).
  /// @returns Base register index, or -1 if no free block.
  int32_t allocate(uint32_t count) {
    if (count == 0 || regs_per_block_ == 0)
      return -1;
    assert(count <= regs_per_block_ && "requested register count exceeds block size");
    for (size_t word = 0; word < free_block_bits_.size(); ++word) {
      if (free_block_bits_[word] != 0) {
        uint32_t block =
            static_cast<uint32_t>(word * 64u + std::countr_zero(free_block_bits_[word]));
        free_block_bits_[word] &= ~(uint64_t{1} << (block % 64u));
        --free_block_count_;
        uint32_t base = block * regs_per_block_;
        // Preserve zero-initialized allocation semantics without eagerly
        // clearing every lane of every physical register. A register is
        // materialized on its first access after this allocation.
        std::fill(initialized_.begin() + base, initialized_.begin() + base + regs_per_block_, 0);
        return static_cast<int32_t>(base);
      }
    }
    return -1;
  }

  /// @brief Free a previously allocated block.
  /// @param base Base register index returned by allocate().
  void free(uint32_t base) {
    if (regs_per_block_ == 0)
      return;
    if (base % regs_per_block_ != 0)
      return;
    uint32_t block = base / regs_per_block_;
    if (block >= (total_regs_ / regs_per_block_))
      return;
    const uint64_t mask = uint64_t{1} << (block % 64u);
    if (free_block_bits_[block / 64u] & mask) {
      assert(false && "double-free of register block");
      return;
    }
    free_block_bits_[block / 64u] |= mask;
    ++free_block_count_;
  }

  /// @brief Access a register by index.
  /// @param idx Register index.
  /// @returns Mutable reference to the register.
  RegType &operator[](uint32_t idx) {
    assert(idx < total_regs_);
    materialize(idx);
    return data_[idx];
  }

  /// @brief Access a register by index (const).
  /// @param idx Register index.
  /// @returns Const reference to the register.
  const RegType &operator[](uint32_t idx) const {
    assert(idx < total_regs_);
    materialize(idx);
    return data_[idx];
  }

  /// @brief Materialize zero-initialized storage for a contiguous register range.
  /// @details Raw bulk consumers must call this before reading or overwriting a
  /// range through a pointer obtained from the first register.
  void materialize_range(uint32_t base, uint32_t count) const {
    assert(base <= total_regs_ && count <= total_regs_ - base);
    for (uint32_t idx = base; idx < base + count; ++idx)
      materialize(idx);
  }

  /// @brief Return the total number of registers.
  /// @returns Total register count.
  uint32_t total_regs() const { return total_regs_; }

  /// @brief Return the number of registers per allocation block.
  /// @returns Registers per block.
  uint32_t regs_per_block() const { return regs_per_block_; }

  /// @brief Count the number of free allocation blocks.
  /// @returns Number of blocks available for allocation.
  uint32_t free_block_count() const { return free_block_count_; }

private:
  void materialize(uint32_t idx) const {
    if (initialized_[idx] == 0) {
      data_[idx] = RegType{};
      initialized_[idx] = 1;
    }
  }

  mutable std::vector<RegType> data_;        ///< One RegType per register.
  mutable std::vector<uint8_t> initialized_; ///< One byte per materialized register.
  uint32_t total_regs_ = 0;                  ///< Total registers.
  uint32_t regs_per_block_ = 0;              ///< Registers per block.
  std::vector<uint64_t> free_block_bits_;    ///< One bit per free block.
  uint32_t free_block_count_ = 0;            ///< Number of free blocks.
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_REGISTER_FILE_H_
