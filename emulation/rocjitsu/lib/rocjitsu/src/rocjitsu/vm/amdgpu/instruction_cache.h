// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_INSTRUCTION_CACHE_H_
#define ROCJITSU_VM_AMDGPU_INSTRUCTION_CACHE_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"

#include <cstdint>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

/// @brief Per-compute-unit instruction cache (I$).
///
/// @details The issue path reads the 16 bytes at the PC for every instruction
/// it retires. Uncached, each of those reads walks the page table and takes a
/// reader lock on a GpuMemory page stripe. Every CU of a dispatch runs the same
/// kernel, so those reads all fall on the one stripe holding the code page and
/// contend on a single host cache line; that contention, not the emulation
/// work, dominates a multi-threaded run.
///
/// The cache is private to one CU and takes no lock. Lines are 64 bytes and
/// never straddle a page, so a fill costs one page-stripe lock instead of
/// sixteen word reads, and a kernel loop that fits in @ref kCacheBytes costs
/// none at all.
///
/// Coherence follows hardware: an AMDGPU I$ is not coherent with data writes,
/// and code that rewrites itself must issue s_icache_inv. The cache is
/// invalidated where the driver would issue one -- at wave launch, at CU cache
/// maintenance, and at the command processor's device-wide maintenance around
/// direct backing writes. It is bypassed entirely while a debugger is attached,
/// because breakpoint writes reach code memory without any of those.
class InstructionCache {
public:
  static constexpr uint32_t kLineSize = 64;
  static constexpr uint32_t kNumLines = 64;
  static constexpr uint32_t kCacheBytes = kLineSize * kNumLines;

  /// @brief Number of bytes the issue path reads at the PC.
  static constexpr uint32_t kFetchBytes = 16;

  static_assert(kLineSize >= kFetchBytes, "a fetch may straddle at most two lines");

  /// @brief Read @ref kFetchBytes at @p pc, filling from @p memory on a miss.
  /// @param memory Backing GPU memory.
  /// @param pc Program counter; four-byte aligned.
  /// @param vmid Owning process address space.
  /// @param[out] dst Buffer of at least @ref kFetchBytes bytes.
  void fetch(const GpuMemory &memory, uint64_t pc, uint32_t vmid, uint8_t *dst) {
    const uint32_t offset = static_cast<uint32_t>(pc) & (kLineSize - 1);
    const uint8_t *line = line_for(memory, pc, vmid);

    if (offset + kFetchBytes <= kLineSize) {
      std::memcpy(dst, line + offset, kFetchBytes);
      return;
    }

    // Straddles into the next line. Copy the head out first: the second lookup
    // may fill, and while adjacent lines never share a set today, relying on
    // that would make the geometry load-bearing.
    const uint32_t head = kLineSize - offset;
    std::memcpy(dst, line + offset, head);
    const uint64_t next = (pc & ~uint64_t{kLineSize - 1}) + kLineSize;
    std::memcpy(dst + head, line_for(memory, next, vmid), kFetchBytes - head);
  }

  /// @brief Discard every cached line (s_icache_inv).
  void invalidate_all() {
    for (Line &line : lines_)
      line.valid = false;
  }

private:
  struct Line {
    uint64_t addr = 0;
    uint32_t vmid = 0;
    bool valid = false;
    uint8_t data[kLineSize] = {};
  };

  const uint8_t *line_for(const GpuMemory &memory, uint64_t addr, uint32_t vmid) {
    const uint64_t line_addr = addr & ~uint64_t{kLineSize - 1};
    Line &line = lines_[(line_addr / kLineSize) & (kNumLines - 1)];
    if (line.valid && line.addr == line_addr && line.vmid == vmid)
      return line.data;

    memory.fetch_block(line_addr, line.data, kLineSize, vmid);
    line.addr = line_addr;
    line.vmid = vmid;
    line.valid = true;
    return line.data;
  }

  Line lines_[kNumLines];
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_INSTRUCTION_CACHE_H_
