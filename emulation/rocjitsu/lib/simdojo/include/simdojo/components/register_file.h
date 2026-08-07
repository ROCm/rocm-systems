// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_file.h
/// @brief Physical register file with block-granularity allocation tracking.

#ifndef SIMDOJO_COMPONENTS_REGISTER_FILE_H_
#define SIMDOJO_COMPONENTS_REGISTER_FILE_H_

#include "simdojo/sim/component.h"
#include "util/reclaimable_buffer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace simdojo {

template <size_t NUM_ELEMS, typename VecElem> class VectorReg;

/// @brief Backing-store policy for a physical register file.
enum class RegisterFileStorage {
  EAGER,        ///< Heap storage committed and initialized up front.
  DEMAND_PAGED, ///< Demand-paged on Linux; eager storage on other platforms.
};

namespace detail {

template <typename RegType> inline constexpr bool has_zero_bit_pattern_v = false;
template <> inline constexpr bool has_zero_bit_pattern_v<uint32_t> = true;
template <size_t NUM_ELEMS>
inline constexpr bool has_zero_bit_pattern_v<VectorReg<NUM_ELEMS, uint32_t>> = true;

template <typename RegType> class EagerRegisterStorage {
public:
  void init(uint32_t count) { data_.assign(count, RegType{}); }

  void reset(uint32_t base, uint32_t count) {
    std::fill(data_.begin() + base, data_.begin() + base + count, RegType{});
  }

  RegType &operator[](uint32_t idx) { return data_[idx]; }
  const RegType &operator[](uint32_t idx) const { return data_[idx]; }
  RegType *data() { return data_.data(); }
  const RegType *data() const { return data_.data(); }

private:
  std::vector<RegType> data_;
};

/// @brief Contiguous register storage with zero-and-reclaim semantics.
///
/// The register types used by rocjitsu are implicit-lifetime, trivially
/// copyable values. ReclaimableBuffer therefore gives them their required
/// all-zero initial representation without eagerly touching storage on systems
/// with demand-paged backing. Resetting a reused allocation restores that same
/// zero-filled state and recycles physical pages when the platform supports it.
template <typename RegType> class DemandPagedRegisterStorage {
public:
  static_assert(std::is_trivially_copyable_v<RegType>);
  static_assert(std::is_trivially_destructible_v<RegType>);
  static_assert(has_zero_bit_pattern_v<RegType>,
                "demand-paged registers must use all-zero bytes for RegType{}");

  DemandPagedRegisterStorage() = default;
  DemandPagedRegisterStorage(const DemandPagedRegisterStorage &) = delete;
  DemandPagedRegisterStorage &operator=(const DemandPagedRegisterStorage &) = delete;
  DemandPagedRegisterStorage(DemandPagedRegisterStorage &&) = delete;
  DemandPagedRegisterStorage &operator=(DemandPagedRegisterStorage &&) = delete;

  void init(uint32_t count) {
    assert(data_ == nullptr && "DemandPagedRegisterStorage already initialized");
    if (count == 0)
      return;
    if (count > std::numeric_limits<size_t>::max() / sizeof(RegType))
      throw std::bad_alloc();

    const size_t bytes = static_cast<size_t>(count) * sizeof(RegType);
    storage_.allocate(bytes, DATA_ALIGNMENT);
    data_ = std::launder(reinterpret_cast<RegType *>(storage_.data()));
  }

  void reset(uint32_t base, uint32_t count) {
    storage_.zero_and_reclaim(static_cast<size_t>(base) * sizeof(RegType),
                              static_cast<size_t>(count) * sizeof(RegType));
  }

  [[nodiscard]] static bool can_reclaim_independently(uint32_t count) noexcept {
    const size_t bytes = static_cast<size_t>(count) * sizeof(RegType);
    return bytes % util::ReclaimableBuffer::reclamation_granularity() == 0;
  }

  RegType &operator[](uint32_t idx) { return data_[idx]; }
  const RegType &operator[](uint32_t idx) const { return data_[idx]; }
  RegType *data() { return data_; }
  const RegType *data() const { return data_; }

private:
  // Preserve the register type's alignment; ReclaimableBuffer strengthens it
  // to the platform reclamation granularity when needed.
  static constexpr size_t DATA_ALIGNMENT = std::max(alignof(RegType), alignof(std::max_align_t));

  util::ReclaimableBuffer storage_;
  RegType *data_ = nullptr;
};

template <typename RegType, RegisterFileStorage Storage>
using RegisterStorage =
    std::conditional_t<Storage == RegisterFileStorage::EAGER, EagerRegisterStorage<RegType>,
                       DemandPagedRegisterStorage<RegType>>;

} // namespace detail

/// @brief Physical register file with block-granularity allocation tracking.
///
/// @details Templated on the register type: use uint32_t for scalar files or
/// VectorReg<NumElems, Elem> for vector files. The file is divided into
/// fixed-size blocks (one per hardware context slot). Allocation finds a
/// free block and returns its base register index. References and pointers are
/// allocation-scoped: callers must not read or write through them before
/// allocation or after freeing their block. Freeing a block invalidates every
/// handle into that block.
///
/// @tparam RegType Register element type (default: uint32_t).
/// @tparam Storage Backing-store policy (default: eager heap storage).
template <typename RegType = uint32_t, RegisterFileStorage Storage = RegisterFileStorage::EAGER>
class RegisterFile : public Component {
public:
  explicit RegisterFile(std::string name) : Component(std::move(name)) {}

  /// @brief Initialize the register file.
  /// @param total_regs Total number of registers in the file.
  /// @param regs_per_block Registers per allocation block (granularity).
  void init(uint32_t total_regs, uint32_t regs_per_block) {
    assert(total_regs_ == 0 && "RegisterFile already initialized");
    total_regs_ = total_regs;
    regs_per_block_ = regs_per_block;
    data_.init(total_regs);
    uint32_t num_blocks = (regs_per_block > 0) ? (total_regs / regs_per_block) : 0;
    free_blocks_.assign(num_blocks, true);
    needs_reset_.assign(num_blocks, false);
  }

  /// @brief Try to allocate a contiguous block of registers.
  /// @param count Number of registers needed (must be <= regs_per_block).
  /// @returns Base register index, or -1 if no free block.
  /// @post On success, every register in the returned allocation block is zero.
  int32_t allocate(uint32_t count) {
    if (count == 0 || regs_per_block_ == 0)
      return -1;
    assert(count <= regs_per_block_ && "requested register count exceeds block size");
    for (size_t i = 0; i < free_blocks_.size(); ++i) {
      if (free_blocks_[i]) {
        free_blocks_[i] = false;
        uint32_t base = static_cast<uint32_t>(i * regs_per_block_);
        if (needs_reset_[i]) {
          data_.reset(base, regs_per_block_);
          needs_reset_[i] = false;
        }
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
    if constexpr (Storage == RegisterFileStorage::DEMAND_PAGED) {
      // Immediately restore the retired block's zero state and let the backing
      // store release physical storage when the platform supports it. Generic
      // layouts whose blocks share reclamation units receive one final
      // whole-file reset so their boundary storage can also be released.
      const bool independently_reclaimable = data_.can_reclaim_independently(regs_per_block_);
      const bool all_blocks_free =
          !independently_reclaimable && std::all_of(free_blocks_.begin(), free_blocks_.end(),
                                                    [](bool is_free) { return is_free; });
      if (all_blocks_free) {
        data_.reset(0, total_regs_);
      } else {
        data_.reset(base, regs_per_block_);
      }
      needs_reset_[block] = false;
    } else {
      needs_reset_[block] = true;
    }
  }

  /// @brief Access a register by index.
  /// @param idx Register index.
  /// @pre The allocation block containing @p idx is currently allocated.
  /// @returns Mutable reference to the register.
  RegType &operator[](uint32_t idx) {
    assert(idx < total_regs_);
    assert(is_allocated(idx) && "mutable access to a free register block");
    return data_[idx];
  }

  /// @brief Access a register by index (const).
  /// @param idx Register index.
  /// @pre The allocation block containing @p idx is currently allocated.
  /// @returns Const reference to the register.
  const RegType &operator[](uint32_t idx) const {
    assert(idx < total_regs_);
    assert(is_allocated(idx) && "const access to a free register block");
    return data_[idx];
  }

  /// @brief Return a pointer to the underlying register storage.
  /// @pre Callers may dereference the returned pointer only within currently
  /// allocated blocks and must stop accessing each block when it is freed.
  /// @returns Mutable pointer to the first register.
  RegType *data() { return data_.data(); }

  /// @brief Return a pointer to the underlying register storage (const).
  /// @pre Callers may dereference the returned pointer only within currently
  /// allocated blocks and must stop accessing each block when it is freed.
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
  bool is_allocated(uint32_t idx) const {
    if (regs_per_block_ == 0)
      return false;
    const size_t block = idx / regs_per_block_;
    return block < free_blocks_.size() && !free_blocks_[block];
  }

  detail::RegisterStorage<RegType, Storage> data_; ///< Register backing storage.
  uint32_t total_regs_ = 0;                        ///< Total registers.
  uint32_t regs_per_block_ = 0;                    ///< Registers per block.
  std::vector<bool> free_blocks_;                  ///< One bit per block (true = free).
  std::vector<bool> needs_reset_;                  ///< Blocks dirtied by prior allocation.
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_REGISTER_FILE_H_
