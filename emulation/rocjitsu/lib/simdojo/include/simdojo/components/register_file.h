// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_file.h
/// @brief Physical register file with block-granularity allocation tracking.

#ifndef SIMDOJO_COMPONENTS_REGISTER_FILE_H_
#define SIMDOJO_COMPONENTS_REGISTER_FILE_H_

#include "simdojo/sim/component.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

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

#if defined(__linux__)
/// @brief Anonymous, contiguous storage whose untouched pages consume no RSS.
///
/// The register types used by rocjitsu are implicit-lifetime, trivially
/// copyable values. Anonymous mmap storage therefore gives them their required
/// all-zero initial representation without faulting in every page. Resetting a
/// reused allocation with MADV_DONTNEED both recycles its physical pages and
/// restores the same zero-filled state as eager initialization.
template <typename RegType> class DemandPagedRegisterStorage {
public:
  static_assert(std::is_trivially_copyable_v<RegType>);
  static_assert(std::is_trivially_destructible_v<RegType>);
  static_assert(alignof(RegType) <= alignof(std::max_align_t));
  static_assert(has_zero_bit_pattern_v<RegType>,
                "demand-paged registers must use all-zero bytes for RegType{}");

  DemandPagedRegisterStorage() = default;
  DemandPagedRegisterStorage(const DemandPagedRegisterStorage &) = delete;
  DemandPagedRegisterStorage &operator=(const DemandPagedRegisterStorage &) = delete;

  ~DemandPagedRegisterStorage() {
    if (mapping_ != nullptr)
      munmap(mapping_, mapped_bytes_);
  }

  void init(uint32_t count) {
    assert(data_ == nullptr && "DemandPagedRegisterStorage already initialized");
    if (count == 0)
      return;

    const size_t bytes = static_cast<size_t>(count) * sizeof(RegType);
    const size_t page_size = system_page_size();
    mapped_bytes_ = ((bytes + DATA_OFFSET + page_size - 1) / page_size) * page_size;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    // A topology can reserve many sparse register files. Avoid having strict
    // overcommit policies charge untouched virtual pages up front; a later
    // page fault can still fail if the host exhausts physical memory and swap.
    flags |= MAP_NORESERVE;
#endif
    void *mapping = mmap(nullptr, mapped_bytes_, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (mapping == MAP_FAILED) {
      data_ = nullptr;
      mapped_bytes_ = 0;
      throw std::bad_alloc();
    }
    mapping_ = mapping;
    // Creating a byte array starts the lifetime of suitable implicit-lifetime
    // objects within its storage without initializing (and faulting in) pages.
    auto *storage = ::new (mapping) std::byte[mapped_bytes_];
    data_ = std::launder(reinterpret_cast<RegType *>(storage + DATA_OFFSET));

#ifdef MADV_NOHUGEPAGE
    // Register accesses are sparse for many kernels. Avoid turning the first
    // touched 4 KiB page into a 2 MiB physical commitment under transparent
    // huge-page policies.
    (void)madvise(mapping_, mapped_bytes_, MADV_NOHUGEPAGE);
#endif
  }

  void reset(uint32_t base, uint32_t count) {
    auto *first = reinterpret_cast<std::byte *>(data_ + base);
    auto *last = reinterpret_cast<std::byte *>(data_ + base + count);
    const uintptr_t page_mask = system_page_size() - 1;
    auto *page_first = reinterpret_cast<std::byte *>(
        (reinterpret_cast<uintptr_t>(first) + page_mask) & ~page_mask);
    auto *page_last = reinterpret_cast<std::byte *>(reinterpret_cast<uintptr_t>(last) & ~page_mask);

    if (page_first >= page_last) {
      std::memset(first, 0, static_cast<size_t>(last - first));
      return;
    }

    std::memset(first, 0, static_cast<size_t>(page_first - first));
    if (madvise(page_first, static_cast<size_t>(page_last - page_first), MADV_DONTNEED) != 0) {
      std::memset(page_first, 0, static_cast<size_t>(page_last - page_first));
    }
    std::memset(page_last, 0, static_cast<size_t>(last - page_last));
  }

  RegType &operator[](uint32_t idx) { return data_[idx]; }
  const RegType &operator[](uint32_t idx) const { return data_[idx]; }
  RegType *data() { return data_; }
  const RegType *data() const { return data_; }

private:
  // Match the offset of glibc's max-aligned large allocations instead of
  // forcing every VGPR file to the same L1 cache-index alignment. The extra
  // bytes remain inside the anonymous mapping and do not consume RSS.
  static constexpr size_t DATA_OFFSET = alignof(std::max_align_t);

  static size_t system_page_size() {
    static const size_t value = [] {
      const long result = sysconf(_SC_PAGESIZE);
      return result > 0 ? static_cast<size_t>(result) : size_t{4096};
    }();
    return value;
  }

  void *mapping_ = nullptr;
  RegType *data_ = nullptr;
  size_t mapped_bytes_ = 0;
};
#else
/// @brief Portable eager fallback for platforms without Linux demand paging.
template <typename RegType> using DemandPagedRegisterStorage = EagerRegisterStorage<RegType>;
#endif

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
/// free block and returns its base register index. Mutable references and
/// pointers are allocation-scoped: callers must not use them before allocation
/// or after freeing their block. Freeing a block invalidates every mutable
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
      // Return physical pages as soon as a hardware context becomes idle. The
      // next allocation reads the anonymous mapping's zero-filled pages.
      const bool all_blocks_free = std::all_of(free_blocks_.begin(), free_blocks_.end(),
                                               [](bool is_free) { return is_free; });
      if (all_blocks_free)
        data_.reset(0, total_regs_);
      else
        data_.reset(base, regs_per_block_);
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
  /// @returns Const reference to the register.
  const RegType &operator[](uint32_t idx) const {
    assert(idx < total_regs_);
    return data_[idx];
  }

  /// @brief Return a pointer to the underlying register storage.
  /// @pre Callers may mutate only registers in currently allocated blocks, and
  /// must stop using each mutable pointer when its block is freed.
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
