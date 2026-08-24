// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_file.h
/// @brief Physical register file with block-granularity allocation tracking.

#ifndef SIMDOJO_COMPONENTS_REGISTER_FILE_H_
#define SIMDOJO_COMPONENTS_REGISTER_FILE_H_

#include "simdojo/sim/component.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace simdojo {

/// @brief Backing-store layout for a physical register file.
enum class RegisterFileStorage {
  CONTIGUOUS,    ///< Contiguous storage used by the small scalar register file.
  SOFTWARE_LAZY, ///< Portable chunk storage allocated on first mutable access.
};

namespace detail {

template <typename RegType, size_t MaxRegisters> class ContiguousRegisterStorage {
public:
  void init(uint32_t count) { data_.assign(count, RegType{}); }

  void reset(uint32_t base, uint32_t count) {
    std::fill(data_.begin() + base, data_.begin() + base + count, RegType{});
  }

  RegType &operator[](uint32_t idx) { return data_[idx]; }
  const RegType &operator[](uint32_t idx) const { return data_[idx]; }

  template <typename Function> void for_each(uint32_t base, uint32_t count, Function &&function) {
    for (uint32_t idx = base; idx < base + count; ++idx)
      function(data_[idx]);
  }

  template <typename Function>
  void for_each(uint32_t base, uint32_t count, Function &&function) const {
    for (uint32_t idx = base; idx < base + count; ++idx)
      function(data_[idx]);
  }

  void copy_to(uint32_t base, std::span<std::byte> destination) const {
    if (destination.empty())
      return;
    std::memcpy(destination.data(), reinterpret_cast<const std::byte *>(data_.data() + base),
                destination.size());
  }

  void copy_from(uint32_t base, std::span<const std::byte> source) {
    if (source.empty())
      return;
    std::memcpy(reinterpret_cast<std::byte *>(data_.data() + base), source.data(), source.size());
  }

  void copy_nonzero_from(uint32_t base, std::span<const std::byte> source) {
    copy_from(base, source);
  }

  RegType *data() { return data_.data(); }
  const RegType *data() const { return data_.data(); }

private:
  std::vector<RegType> data_;
};

/// @brief Portable sparse register storage allocated in fixed-size chunks.
///
/// Const access to an absent chunk observes immutable zero storage without
/// allocating. Mutable access materializes and zero-initializes the containing
/// chunk. The storage retains CU-global register indices but deliberately does
/// not expose a single contiguous pointer spanning multiple chunks.
template <typename RegType, size_t MaxRegisters> class SoftwareLazyRegisterStorage {
public:
  static_assert(MaxRegisters > 0);
  static_assert(std::is_trivially_copyable_v<RegType>);
  static_assert(std::is_trivially_destructible_v<RegType>);

  SoftwareLazyRegisterStorage() = default;
  SoftwareLazyRegisterStorage(const SoftwareLazyRegisterStorage &) = delete;
  SoftwareLazyRegisterStorage &operator=(const SoftwareLazyRegisterStorage &) = delete;
  SoftwareLazyRegisterStorage(SoftwareLazyRegisterStorage &&) = delete;
  SoftwareLazyRegisterStorage &operator=(SoftwareLazyRegisterStorage &&) = delete;

  void init(uint32_t count) {
    assert(total_regs_ == 0 && "SoftwareLazyRegisterStorage already initialized");
    assert(count <= MaxRegisters && "register count exceeds lazy storage capacity");
    if (count > MaxRegisters)
      std::abort();
    total_regs_ = count;
  }

  void reset(uint32_t base, uint32_t count) {
    assert(base <= total_regs_ && count <= total_regs_ - base && "range exceeds storage");
    if (count == 0)
      return;

    const size_t first_chunk = base / REGS_PER_CHUNK;
    const size_t last_chunk = (static_cast<size_t>(base) + count - 1) / REGS_PER_CHUNK;
    const size_t range_end = static_cast<size_t>(base) + count;
    for (size_t chunk_idx = first_chunk; chunk_idx <= last_chunk; ++chunk_idx) {
      auto &chunk = chunks_[chunk_idx];
      if (!chunk)
        continue;

      const size_t chunk_base = chunk_idx * REGS_PER_CHUNK;
      const size_t valid_chunk_end =
          std::min(chunk_base + REGS_PER_CHUNK, static_cast<size_t>(total_regs_));
      const size_t clear_begin = std::max(static_cast<size_t>(base), chunk_base);
      const size_t clear_end = std::min(range_end, valid_chunk_end);
      if (clear_begin == chunk_base && clear_end == valid_chunk_end) {
        chunk.reset();
      } else {
        std::fill(chunk->registers.begin() + static_cast<ptrdiff_t>(clear_begin - chunk_base),
                  chunk->registers.begin() + static_cast<ptrdiff_t>(clear_end - chunk_base),
                  RegType{});
      }
    }
  }

  [[nodiscard]] static constexpr bool can_reclaim_independently(uint32_t count) noexcept {
    return count % REGS_PER_CHUNK == 0;
  }

  [[nodiscard]] static constexpr uint32_t registers_per_chunk() noexcept { return REGS_PER_CHUNK; }

  /// @brief Count chunks with materialized register storage.
  /// @returns Number of currently materialized chunks.
  [[nodiscard]] size_t materialized_chunk_count() const noexcept {
    return static_cast<size_t>(std::count_if(chunks_.begin(), chunks_.end(),
                                             [](const auto &chunk) { return chunk != nullptr; }));
  }

  RegType &operator[](uint32_t idx) {
    assert(idx < total_regs_);
    auto &chunk = chunks_[idx / REGS_PER_CHUNK];
    if (!chunk)
      chunk = std::make_unique<Chunk>();
    return chunk->registers[idx % REGS_PER_CHUNK];
  }

  const RegType &operator[](uint32_t idx) const {
    assert(idx < total_regs_);
    const auto &chunk = chunks_[idx / REGS_PER_CHUNK];
    if (!chunk)
      return zero_register_;
    return chunk->registers[idx % REGS_PER_CHUNK];
  }

  template <typename Function> void for_each(uint32_t base, uint32_t count, Function &&function) {
    size_t local_idx = base % REGS_PER_CHUNK;
    size_t chunk_idx = base / REGS_PER_CHUNK;
    while (count != 0) {
      const size_t chunk_count = std::min<size_t>(REGS_PER_CHUNK - local_idx, count);
      const size_t chunk_end = local_idx + chunk_count;

      auto &chunk = chunks_[chunk_idx];
      if (!chunk)
        chunk = std::make_unique<Chunk>();

      for (; local_idx < chunk_end; ++local_idx)
        function(chunk->registers[local_idx]);
      count -= chunk_count;
      ++chunk_idx;
      local_idx = 0;
    }
  }

  template <typename Function>
  void for_each(uint32_t base, uint32_t count, Function &&function) const {
    size_t local_idx = base % REGS_PER_CHUNK;
    size_t chunk_idx = base / REGS_PER_CHUNK;
    while (count != 0) {
      const size_t chunk_count = std::min<size_t>(REGS_PER_CHUNK - local_idx, count);
      const size_t chunk_end = local_idx + chunk_count;

      const auto &chunk = chunks_[chunk_idx];
      if (chunk) {
        for (; local_idx < chunk_end; ++local_idx)
          function(chunk->registers[local_idx]);
      } else {
        for (; local_idx < chunk_end; ++local_idx)
          function(zero_register_);
      }
      count -= chunk_count;
      ++chunk_idx;
      local_idx = 0;
    }
  }

  void copy_to(uint32_t base, std::span<std::byte> destination) const {
    visit_byte_runs(
        base, destination.size(),
        [&](size_t chunk_idx, size_t chunk_offset, size_t run_size, size_t destination_offset) {
          const auto &chunk = chunks_[chunk_idx];
          if (!chunk) {
            std::memset(destination.data() + destination_offset, 0, run_size);
            return;
          }
          const auto *source = reinterpret_cast<const std::byte *>(chunk->registers.data());
          std::memcpy(destination.data() + destination_offset, source + chunk_offset, run_size);
        });
  }

  void copy_from(uint32_t base, std::span<const std::byte> source) {
    visit_byte_runs(
        base, source.size(),
        [&](size_t chunk_idx, size_t chunk_offset, size_t run_size, size_t source_offset) {
          auto &chunk = chunks_[chunk_idx];
          if (!chunk)
            chunk = std::make_unique<Chunk>();
          auto *destination = reinterpret_cast<std::byte *>(chunk->registers.data());
          std::memcpy(destination + chunk_offset, source.data() + source_offset, run_size);
        });
  }

  /// Copy into storage known to be logically zero, preserving absent chunks for
  /// source runs that contain only zero bytes.
  void copy_nonzero_from(uint32_t base, std::span<const std::byte> source) {
    visit_byte_runs(
        base, source.size(),
        [&](size_t chunk_idx, size_t chunk_offset, size_t run_size, size_t source_offset) {
          const auto run = source.subspan(source_offset, run_size);
          if (std::all_of(run.begin(), run.end(),
                          [](std::byte value) { return value == std::byte{}; }))
            return;
          auto &chunk = chunks_[chunk_idx];
          if (!chunk)
            chunk = std::make_unique<Chunk>();
          auto *destination = reinterpret_cast<std::byte *>(chunk->registers.data());
          std::memcpy(destination + chunk_offset, run.data(), run.size());
        });
  }

private:
  static constexpr size_t TARGET_CHUNK_BYTES = 4096;
  static constexpr uint32_t REGS_PER_CHUNK =
      static_cast<uint32_t>(std::max<size_t>(1, TARGET_CHUNK_BYTES / sizeof(RegType)));
  static constexpr size_t MAX_CHUNKS = (MaxRegisters + REGS_PER_CHUNK - 1) / REGS_PER_CHUNK;

  struct Chunk {
    std::array<RegType, REGS_PER_CHUNK> registers{};
  };

  template <typename Function>
  static void visit_byte_runs(uint32_t base, size_t byte_count, Function &&function) {
    constexpr size_t CHUNK_BYTES = REGS_PER_CHUNK * sizeof(RegType);
    size_t storage_offset = static_cast<size_t>(base) * sizeof(RegType);
    size_t range_offset = 0;
    while (range_offset < byte_count) {
      const size_t chunk_idx = storage_offset / CHUNK_BYTES;
      const size_t chunk_offset = storage_offset % CHUNK_BYTES;
      const size_t run_size = std::min(byte_count - range_offset, CHUNK_BYTES - chunk_offset);
      function(chunk_idx, chunk_offset, run_size, range_offset);
      storage_offset += run_size;
      range_offset += run_size;
    }
  }

  inline static const RegType zero_register_{};
  std::array<std::unique_ptr<Chunk>, MAX_CHUNKS> chunks_{};
  uint32_t total_regs_ = 0;
};

template <typename RegType, RegisterFileStorage Storage, size_t MaxRegisters>
using RegisterStorage = std::conditional_t<Storage == RegisterFileStorage::CONTIGUOUS,
                                           ContiguousRegisterStorage<RegType, MaxRegisters>,
                                           SoftwareLazyRegisterStorage<RegType, MaxRegisters>>;

} // namespace detail

/// @brief Physical register file with block-granularity allocation tracking.
///
/// @details Templated on the register type: use uint32_t for scalar files or
/// VectorReg<NumElems, Elem> for vector files. The file is divided into
/// fixed-size blocks (one per hardware context slot). Allocation finds a
/// free block and returns its base register index. References and pointers are
/// allocation-scoped: callers must not read or write through them before
/// allocation or after freeing their block. Freeing a block invalidates every
/// handle into that block. With software-lazy storage, const access to an
/// unmaterialized register is an ephemeral observation of its logical zero
/// value and may use shared immutable backing. Such a handle is not a persistent
/// storage identity and need not observe a later write through a mutable handle.
/// Mutable handles, and const handles into already materialized storage, remain
/// stable until their allocation is freed.
/// A pointer or reference to one register addresses only that register and must
/// not be incremented to traverse adjacent registers.
///
/// @tparam RegType Register element type (default: uint32_t).
/// @tparam Storage Backing-store layout (default: contiguous storage).
/// @tparam MaxRegisters Maximum logical register count for fixed-capacity storage.
template <typename RegType = uint32_t,
          RegisterFileStorage Storage = RegisterFileStorage::CONTIGUOUS, size_t MaxRegisters = 0>
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
    if constexpr (Storage == RegisterFileStorage::SOFTWARE_LAZY) {
      // Immediately restore the retired block's zero state and release wholly
      // covered chunks. Layouts whose blocks share chunks receive one final
      // whole-file reset so their boundary chunks can also be released.
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

  /// @brief Visit each logical register in an allocated range in index order.
  template <typename Function> void for_each(uint32_t base, uint32_t count, Function &&function) {
    assert_allocated_range(base, count);
    data_.for_each(base, count, std::forward<Function>(function));
  }

  /// @brief Visit each logical register in an allocated range in index order.
  template <typename Function>
  void for_each(uint32_t base, uint32_t count, Function &&function) const {
    assert_allocated_range(base, count);
    data_.for_each(base, count, std::forward<Function>(function));
  }

  /// @brief Copy bytes from an allocated logical register range.
  /// @details The destination may end within the final register.
  void copy_to(uint32_t base, uint32_t count, std::span<std::byte> destination) const
    requires std::is_trivially_copyable_v<RegType>
  {
    assert_allocated_range(base, count);
    assert(destination.size() <= static_cast<size_t>(count) * sizeof(RegType));
    data_.copy_to(base, destination);
  }

  /// @brief Copy bytes into an allocated logical register range.
  /// @details The source may end within the final register.
  void copy_from(uint32_t base, uint32_t count, std::span<const std::byte> source)
    requires std::is_trivially_copyable_v<RegType>
  {
    assert_allocated_range(base, count);
    assert(source.size() <= static_cast<size_t>(count) * sizeof(RegType));
    data_.copy_from(base, source);
  }

  /// @brief Copy nonzero source runs into a logically zero allocated range.
  /// @details This preserves sparse backing when restoring zero-heavy state.
  void copy_nonzero_from(uint32_t base, uint32_t count, std::span<const std::byte> source)
    requires std::is_trivially_copyable_v<RegType>
  {
    assert_allocated_range(base, count);
    assert(source.size() <= static_cast<size_t>(count) * sizeof(RegType));
    data_.copy_nonzero_from(base, source);
  }

  /// @brief Return a pointer to the underlying register storage.
  /// @pre Callers may dereference the returned pointer only within currently
  /// allocated blocks and must stop accessing each block when it is freed.
  /// @returns Mutable pointer to the first register.
  RegType *data()
    requires requires(detail::RegisterStorage<RegType, Storage, MaxRegisters> &storage) {
      storage.data();
    }
  {
    return data_.data();
  }

  /// @brief Return a pointer to the underlying register storage (const).
  /// @pre Callers may dereference the returned pointer only within currently
  /// allocated blocks and must stop accessing each block when it is freed.
  /// @returns Const pointer to the first register.
  const RegType *data() const
    requires requires(const detail::RegisterStorage<RegType, Storage, MaxRegisters> &storage) {
      storage.data();
    }
  {
    return data_.data();
  }

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

  /// @brief Count chunks with materialized backing storage.
  /// @returns Number of currently materialized chunks.
  [[nodiscard]] size_t materialized_chunk_count() const
    requires requires(const detail::RegisterStorage<RegType, Storage, MaxRegisters> &storage) {
      storage.materialized_chunk_count();
    }
  {
    return data_.materialized_chunk_count();
  }

private:
  void assert_allocated_range(uint32_t base, uint32_t count) const {
    assert(base <= total_regs_ && count <= total_regs_ - base && "range exceeds register file");
    if (count == 0)
      return;
    assert(regs_per_block_ != 0);
    const size_t first_block = base / regs_per_block_;
    const size_t last_block = (static_cast<size_t>(base) + count - 1) / regs_per_block_;
    for (size_t block = first_block; block <= last_block; ++block)
      assert(block < free_blocks_.size() && !free_blocks_[block] &&
             "range includes a free register block");
  }

  bool is_allocated(uint32_t idx) const {
    if (regs_per_block_ == 0)
      return false;
    const size_t block = idx / regs_per_block_;
    return block < free_blocks_.size() && !free_blocks_[block];
  }

  detail::RegisterStorage<RegType, Storage, MaxRegisters> data_; ///< Register backing storage.
  uint32_t total_regs_ = 0;                                      ///< Total registers.
  uint32_t regs_per_block_ = 0;                                  ///< Registers per block.
  std::vector<bool> free_blocks_; ///< One bit per block (true = free).
  std::vector<bool> needs_reset_; ///< Blocks dirtied by prior allocation.
};

} // namespace simdojo

#endif // SIMDOJO_COMPONENTS_REGISTER_FILE_H_
