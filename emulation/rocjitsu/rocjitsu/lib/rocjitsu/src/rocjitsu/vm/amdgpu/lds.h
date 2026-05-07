// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_LDS_H_
#define ROCJITSU_VM_AMDGPU_LDS_H_

#include "simdojo/components/memory_interface.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief Per-Compute Unit Local Data Share (LDS) memory.
///
/// @details LDS is a fast on-chip SRAM shared by all wavefronts within a single CU.
/// Size is configurable per CU. Addresses are byte-granularity, local to the CU
/// (not globally visible).
class Lds : public simdojo::MemoryInterface {
public:
  /// @brief Construct LDS with the given size in kilobytes.
  explicit Lds(uint32_t size_kb) : data_(static_cast<size_t>(size_kb) * 1024, 0) {}

  /// @brief Return the total size in bytes.
  size_t size_bytes() const { return data_.size(); }

  /// @brief Read a single byte from LDS. OOB returns 0.
  uint8_t read8(uint32_t addr) const {
    if (addr >= data_.size())
      return 0;
    return data_[addr];
  }

  /// @brief Write a single byte to LDS. OOB writes are dropped.
  void write8(uint32_t addr, uint8_t val) {
    if (addr >= data_.size())
      return;
    data_[addr] = val;
  }

  /// @brief Read 16 bits (little-endian) from LDS. OOB returns 0.
  uint16_t read16(uint32_t addr) const {
    if (addr + 1 >= data_.size())
      return 0;
    uint16_t val;
    std::memcpy(&val, &data_[addr], 2);
    return val;
  }

  /// @brief Write 16 bits (little-endian) to LDS. OOB writes are dropped.
  void write16(uint32_t addr, uint16_t val) {
    if (addr + 1 >= data_.size())
      return;
    std::memcpy(&data_[addr], &val, 2);
  }

  /// @brief Read 32 bits (little-endian) from LDS. OOB returns 0.
  uint32_t read32(uint32_t addr) const {
    if (addr + 3 >= data_.size())
      return 0;
    uint32_t val;
    std::memcpy(&val, &data_[addr], 4);
    return val;
  }

  /// @brief Write 32 bits (little-endian) to LDS. OOB writes are dropped.
  void write32(uint32_t addr, uint32_t val) {
    if (addr + 3 >= data_.size())
      return;
    std::memcpy(&data_[addr], &val, 4);
  }

  /// @brief Read 64 bits (little-endian) from LDS. OOB returns 0.
  uint64_t read64(uint32_t addr) const {
    if (addr + 7 >= data_.size())
      return 0;
    uint64_t val;
    std::memcpy(&val, &data_[addr], 8);
    return val;
  }

  /// @brief Write 64 bits (little-endian) to LDS. OOB writes are dropped.
  void write64(uint32_t addr, uint64_t val) {
    if (addr + 7 >= data_.size())
      return;
    std::memcpy(&data_[addr], &val, 8);
  }

  /// @brief Bulk read of arbitrary size from LDS. OOB returns 0.
  void read(uint32_t addr, uint8_t *dst, uint32_t size) const {
    if (addr + size > data_.size()) {
      std::memset(dst, 0, size);
      return;
    }
    std::memcpy(dst, &data_[addr], size);
  }

  /// @brief Bulk write of arbitrary size to LDS. OOB writes are dropped.
  void write(uint32_t addr, const uint8_t *src, uint32_t size) {
    if (addr + size > data_.size())
      return;
    std::memcpy(&data_[addr], src, size);
  }

  /// @brief MemoryInterface read (truncates addr to 32-bit local address).
  void read(uint64_t addr, uint8_t *dst, uint32_t size) override {
    auto a = static_cast<uint32_t>(addr);
    assert(a + size <= data_.size());
    std::memcpy(dst, &data_[a], size);
  }

  /// @brief MemoryInterface write (truncates addr to 32-bit local address).
  void write(uint64_t addr, const uint8_t *src, uint32_t size) override {
    auto a = static_cast<uint32_t>(addr);
    assert(a + size <= data_.size());
    std::memcpy(&data_[a], src, size);
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
        if (ea + elem_size > data_.size()) {
          std::memset(dst + lane * stride + e * elem_size, 0, elem_size);
          continue;
        }
        std::memcpy(dst + lane * stride + e * elem_size, &data_[ea], elem_size);
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
        if (ea + elem_size > data_.size())
          continue;
        std::memcpy(&data_[ea], src + lane * stride + e * elem_size, elem_size);
      }
    }
  }

  /// @brief Zero all LDS contents.
  void clear() { std::memset(data_.data(), 0, data_.size()); }

  void zero_range(uint32_t offset, uint32_t len) {
    uint32_t end = std::min(static_cast<uint32_t>(data_.size()), offset + len);
    if (offset < end)
      std::memset(&data_[offset], 0, end - offset);
  }

  /// @brief Direct pointer access (for DMA or debugging).
  const uint8_t *data() const { return data_.data(); }
  uint8_t *data() { return data_.data(); }

private:
  std::vector<uint8_t> data_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_H_
