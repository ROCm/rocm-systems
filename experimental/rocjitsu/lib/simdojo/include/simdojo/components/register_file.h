// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_file.h
/// @brief Physical register file with block-granularity allocation tracking.

#ifndef SIMDOJO_COMPONENTS_REGISTER_FILE_H_
#define SIMDOJO_COMPONENTS_REGISTER_FILE_H_

#include "simdojo/sim/component.h"

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
    uint32_t num_blocks = (regs_per_block > 0) ? (total_regs / regs_per_block) : 0;
    free_blocks_.assign(num_blocks, true);
  }

  /// @brief Try to allocate a contiguous block of registers.
  /// @param count Number of registers needed (must be <= regs_per_block).
  /// @returns Base register index, or -1 if no free block.
  int32_t allocate(uint32_t count) {
    if (count == 0 || regs_per_block_ == 0)
      return -1;
    assert(count <= regs_per_block_ && "requested register count exceeds block size");
    for (size_t i = 0; i < free_blocks_.size(); ++i) {
      if (free_blocks_[i]) {
        free_blocks_[i] = false;
        uint32_t base = static_cast<uint32_t>(i * regs_per_block_);
        for (uint32_t r = base; r < base + regs_per_block_; ++r)
          data_[r] = RegType{};
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
    if (block >= free_blocks_.size())
      return;
    assert(!free_blocks_[block] && "double-free of register block");
    free_blocks_[block] = true;
  }

  /// @brief Access a register by index.
  /// @param idx Register index.
  /// @returns Mutable reference to the register.
  RegType &operator[](uint32_t idx) {
    assert(idx < total_regs_);
    return data_[idx];
  }

  /// @brief Access a register by index (const).
  /// @param idx Register index.
  /// @returns Const reference to the register.
  const RegType &operator[](uint32_t idx) const {
    assert(idx < total_regs_);
    return data_[idx];
  }

  /// @brief Return a pointer to the underlying register storage.
  /// @returns Mutable pointer to the first register.
  RegType *data() { return data_.data(); }

  /// @brief Return a pointer to the underlying register storage (const).
  /// @returns Const pointer to the first register.
  const RegType *data() const { return data_.data(); }

  /// @brief Return the total number of registers.
  /// @returns Total register count.
  uint32_t total_regs() const { return total_regs_; }

  /// @brief Return the number of registers per allocation block.
  /// @returns Registers per block.
  uint32_t regs_per_block() const { return regs_per_block_; }

  /// @brief Count the number of free allocation blocks.
  /// @returns Number of blocks available for allocation.
  uint32_t free_block_count() const {
    uint32_t count = 0;
    for (bool b : free_blocks_)
      if (b)
        ++count;
    return count;
  }

private:
  std::vector<RegType> data_;     ///< One RegType per register.
  uint32_t total_regs_ = 0;       ///< Total registers.
  uint32_t regs_per_block_ = 0;   ///< Registers per block.
  std::vector<bool> free_blocks_; ///< One bit per block (true = free).
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_REGISTER_FILE_H_
