// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_LDS_H_
#define ROCJITSU_VM_AMDGPU_LDS_H_

#include "simdojo/components/memory_interface.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// Reserved value used when an instruction lane has no valid LDS address.
constexpr uint32_t kInvalidLdsAddress = UINT32_MAX;

/// @brief Local Data Share (LDS) memory backing for a dispatch placement.
///
/// @details CU-mode workgroups use the backing owned by one ComputeUnitCore.
/// RDNA WGP-mode workgroups use a backing owned by the sibling-CU pair. The
/// latter has the combined capacity of both physical CUs. Addresses are
/// byte-granularity and local to the selected placement (not globally visible).
/// Logical capacity is independent of host storage: unmaterialized bytes read
/// as zero, while writes and workgroup reservations grow a contiguous,
/// prefix in fixed 4 KiB backing granules. Clearing LDS retains the materialized
/// prefix for reuse.
class Lds : public simdojo::MemoryInterface {
public:
  /// @brief Construct LDS with the given size in kilobytes.
  explicit Lds(uint32_t size_kb) : capacity_bytes_(static_cast<size_t>(size_kb) * 1024) {}

  /// @brief Return the total size in bytes.
  size_t size_bytes() const { return capacity_bytes_; }

  /// @brief Return the bytes currently backed by host storage.
  ///
  /// @details The remaining logical capacity reads as zero and is materialized
  /// on the first write or workgroup allocation that reaches it. This accessor
  /// is intended for diagnostics and allocation tests.
  size_t materialized_size_bytes() const { return data_.size(); }

  /// @brief Read a single byte from LDS. OOB returns 0.
  uint8_t read8(uint32_t addr) const {
    if (addr >= capacity_bytes_ || addr >= data_.size())
      return 0;
    return data_[addr];
  }

  /// @brief Write a single byte to LDS. OOB writes are dropped.
  void write8(uint32_t addr, uint8_t val) {
    if (addr >= capacity_bytes_)
      return;
    ensure_materialized(static_cast<size_t>(addr) + 1);
    data_[addr] = val;
  }

  /// @brief Read 16 bits (little-endian) from LDS. OOB returns 0.
  uint16_t read16(uint32_t addr) const {
    if (!contains(addr, 2))
      return 0;
    uint16_t val = 0;
    read_backing(addr, reinterpret_cast<uint8_t *>(&val), sizeof(val));
    return val;
  }

  /// @brief Write 16 bits (little-endian) to LDS. OOB writes are dropped.
  void write16(uint32_t addr, uint16_t val) {
    if (!contains(addr, 2))
      return;
    write_backing(addr, reinterpret_cast<const uint8_t *>(&val), sizeof(val));
  }

  /// @brief Read 32 bits (little-endian) from LDS. OOB returns 0.
  uint32_t read32(uint32_t addr) const {
    if (!contains(addr, 4))
      return 0;
    uint32_t val = 0;
    read_backing(addr, reinterpret_cast<uint8_t *>(&val), sizeof(val));
    return val;
  }

  /// @brief Write 32 bits (little-endian) to LDS. OOB writes are dropped.
  void write32(uint32_t addr, uint32_t val) {
    if (!contains(addr, 4))
      return;
    write_backing(addr, reinterpret_cast<const uint8_t *>(&val), sizeof(val));
  }

  /// @brief Read 64 bits (little-endian) from LDS. OOB returns 0.
  uint64_t read64(uint32_t addr) const {
    if (!contains(addr, 8))
      return 0;
    uint64_t val = 0;
    read_backing(addr, reinterpret_cast<uint8_t *>(&val), sizeof(val));
    return val;
  }

  /// @brief Write 64 bits (little-endian) to LDS. OOB writes are dropped.
  void write64(uint32_t addr, uint64_t val) {
    if (!contains(addr, 8))
      return;
    write_backing(addr, reinterpret_cast<const uint8_t *>(&val), sizeof(val));
  }

  /// @brief Bulk read of arbitrary size from LDS. OOB returns 0.
  void read(uint32_t addr, uint8_t *dst, uint32_t size) const {
    if (size == 0)
      return;
    if (!contains(addr, size)) {
      std::memset(dst, 0, size);
      return;
    }
    read_backing(addr, dst, size);
  }

  /// @brief Bulk write of arbitrary size to LDS. OOB writes are dropped.
  void write(uint32_t addr, const uint8_t *src, uint32_t size) {
    if (size == 0 || !contains(addr, size))
      return;
    write_backing(addr, src, size);
  }

  /// @brief MemoryInterface read (truncates addr to 32-bit local address).
  void read(uint64_t addr, uint8_t *dst, uint32_t size) override {
    if (size == 0)
      return;
    auto a = static_cast<uint32_t>(addr);
    assert(contains(a, size));
    read_backing(a, dst, size);
  }

  /// @brief MemoryInterface write (truncates addr to 32-bit local address).
  void write(uint64_t addr, const uint8_t *src, uint32_t size) override {
    if (size == 0)
      return;
    auto a = static_cast<uint32_t>(addr);
    assert(contains(a, size));
    write_backing(a, src, size);
  }

  /// @brief Per-lane vector load from LDS.
  ///
  /// Reads `num_elems` elements of `elem_size` bytes for each active lane.
  /// Output layout: `dst[lane * stride + elem * elem_size]` where
  /// `stride = num_elems * elem_size`.
  /// @param addrs Per-lane LDS byte addresses (64 entries).
  /// @param lane_mask Bitmask of active lanes.
  /// @param elem_size Size of each element in bytes.
  /// @param num_elems Number of elements per lane.
  /// @param[out] dst Destination buffer.
  /// @param base_offset Per-workgroup LDS base offset added to each lane's address.
  ///        Callers should set this from the dispatch packet's LDS allocation
  ///        to enable per-workgroup LDS partitioning. Defaults to 0.
  void vector_load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                   uint32_t num_elems, uint8_t *dst, uint32_t base_offset = 0) {
    uint32_t stride = num_elems * elem_size;
    for (uint32_t lane = 0; lane < 64; ++lane) {
      if (!(lane_mask & (1ULL << lane)))
        continue;
      uint64_t base = static_cast<uint64_t>(static_cast<uint32_t>(addrs[lane])) + base_offset;
      for (uint32_t e = 0; e < num_elems; ++e) {
        uint64_t ea = base + static_cast<uint64_t>(e) * elem_size;
        if (ea + elem_size > capacity_bytes_) {
          std::memset(dst + lane * stride + e * elem_size, 0, elem_size);
          continue;
        }
        read_backing(static_cast<uint32_t>(ea), dst + lane * stride + e * elem_size, elem_size);
      }
    }
  }

  /// @brief Per-lane vector store to LDS. OOB writes are dropped.
  void vector_store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                    uint32_t num_elems, const uint8_t *src, uint32_t base_offset = 0) {
    uint32_t stride = num_elems * elem_size;
    for (uint32_t lane = 0; lane < 64; ++lane) {
      if (!(lane_mask & (1ULL << lane)))
        continue;
      uint64_t base = static_cast<uint64_t>(static_cast<uint32_t>(addrs[lane])) + base_offset;
      for (uint32_t e = 0; e < num_elems; ++e) {
        uint64_t ea = base + static_cast<uint64_t>(e) * elem_size;
        if (ea + elem_size > capacity_bytes_)
          continue;
        write_backing(static_cast<size_t>(ea), src + lane * stride + e * elem_size, elem_size);
      }
    }
  }

  /// @brief Zero all LDS contents.
  void clear() {
    if (!data_.empty())
      std::memset(data_.data(), 0, data_.size());
  }

  void zero_range(uint32_t offset, uint32_t len) {
    const size_t begin = offset;
    const size_t end = std::min(capacity_bytes_, begin + static_cast<size_t>(len));
    if (begin >= end)
      return;
    ensure_materialized(end);
    std::memset(&data_[begin], 0, end - begin);
  }

private:
  bool contains(uint32_t addr, uint32_t size) const {
    return static_cast<uint64_t>(addr) + size <= capacity_bytes_;
  }

  void read_backing(uint32_t addr, uint8_t *dst, uint32_t size) const {
    const size_t begin = addr;
    const size_t backed = begin < data_.size() ? std::min<size_t>(size, data_.size() - begin) : 0;
    if (backed != 0)
      std::memcpy(dst, &data_[begin], backed);
    if (backed != size)
      std::memset(dst + backed, 0, size - backed);
  }

  void write_backing(size_t offset, const uint8_t *src, size_t size) {
    assert(offset <= capacity_bytes_ && size <= capacity_bytes_ - offset);
    ensure_materialized(offset + size);
    // Keep the checked loop explicit: GCC 15 can retain the vector's old object
    // size across resize and report a false stringop-overflow for memcpy here.
    for (size_t i = 0; i < size; ++i)
      data_[offset + i] = src[i];
  }

  void ensure_materialized(size_t required) {
    if (required <= data_.size())
      return;
    assert(required <= capacity_bytes_);

    // Grow in fixed 4 KiB backing granules so storage tracks the allocated LDS
    // prefix without exposing vector growth heuristics as part of the model.
    constexpr size_t kBackingGranuleBytes = 4096;
    size_t rounded = required;
    const size_t remainder = rounded % kBackingGranuleBytes;
    if (remainder != 0) {
      const size_t increment = kBackingGranuleBytes - remainder;
      rounded = increment <= capacity_bytes_ - rounded ? rounded + increment : capacity_bytes_;
    }
    data_.resize(rounded, 0);
  }

  size_t capacity_bytes_ = 0;
  std::vector<uint8_t> data_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_H_
