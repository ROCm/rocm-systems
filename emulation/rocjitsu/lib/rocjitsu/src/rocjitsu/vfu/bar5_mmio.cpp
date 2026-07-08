// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/bar5_mmio.h"
#include "rocjitsu/vfu/mmio_registers.h"

#include <libvfio-user.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>

namespace rocjitsu::vfu {

MmioModel::MmioModel(int vram_fd, uint64_t vram_size)
    : vram_fd_(vram_fd), vram_size_(vram_size) {
  // Pre-populate registers that the amdgpu driver reads during hw_init.

  // GRBM_STATUS: report GPU idle (all busy bits clear).
  regs_[kRegGrbmStatus / 4]  = kGrbmStatusIdle;
  regs_[kRegGrbmStatus2 / 4] = kGrbmStatusIdle;

  // SDMA version registers (5 SDMA engines for MI350P).
  for (int i = 0; i < 5; ++i) {
    uint32_t off = (kRegSdma0Version + static_cast<uint32_t>(i) * kSdmaEngineStride) / 4;
    regs_[off] = kSdmaVersionStub;
  }

  // SMC response: pre-set "OK" so power management queries don't spin.
  regs_[kRegMpSmcResp / 4] = kSmcRespOk;

  // BIF device ID mirror (must match PCI config space).
  regs_[kRegBifBxDevZeroId / 4] = (0x75C8U << 16) | 0x1002U;

  // RCC_CONFIG_MEMSIZE: VRAM size in MB (used by amdgpu_discovery to locate
  // the IP discovery table at vram_size_bytes - DISCOVERY_TMR_OFFSET).
  regs_[kRegRccConfigMemsize / 4] = kVramSizeMb;
}

// ---------------------------------------------------------------------------
// Indirect register access (MM_INDEX / MM_DATA)
//
// amdgpu RREG32(dword_index) uses direct BAR5 for dword_index < 0x10000.
// For dword_index >= 0x10000, it writes (dword_index | 0x80000000) to
// MM_INDEX (byte 0x0) and then reads MM_DATA (byte 0x4).
// We respond to the key indirect registers that the IP discovery path needs.
// ---------------------------------------------------------------------------
uint32_t MmioModel::read_indirect(uint32_t dword_index) {
  // If MM_INDEX_HI is non-zero this is a 64-bit VRAM access via the MM path.
  // amdgpu_device_mm_access() writes MM_INDEX_HI then MM_INDEX (with bit 31
  // set for indirect), then reads MM_DATA one dword at a time to copy the
  // IP discovery binary out of VRAM.
  // All MM_DATA reads are VRAM accesses from amdgpu_device_mm_access().
  // C2PMSG_33 goes through the PCIE path, not here.
  // MM_INDEX_HI holds bits [63:32]; dword_index holds [31:0] (bit 31 already
  // stripped as the indirect flag by the caller).
  if (vram_fd_ >= 0) {
    uint64_t byte_addr = (static_cast<uint64_t>(mm_index_hi_) << 32) |
                         static_cast<uint64_t>(dword_index);
    uint32_t val = 0;
    if (byte_addr + 4 <= vram_size_) {
      void *p = ::mmap(nullptr, 4, PROT_READ, MAP_SHARED, vram_fd_,
                       static_cast<off_t>(byte_addr));
      if (p != MAP_FAILED) {
        std::memcpy(&val, p, 4);
        ::munmap(p, 4);
        std::fprintf(stderr,
                     "[vfu/bar5] MM VRAM read addr=0x%llx val=0x%08x\n",
                     static_cast<unsigned long long>(byte_addr), val);
      } else {
        std::perror("[vfu/bar5] mmap for MM VRAM read");
      }
    } else {
      std::fprintf(stderr,
                   "[vfu/bar5] MM VRAM read out of range: addr=0x%llx size=0x%llx\n",
                   static_cast<unsigned long long>(byte_addr),
                   static_cast<unsigned long long>(vram_size_));
    }
    return val;
  }

  std::fprintf(stderr, "[vfu/bar5] indirect read: no VRAM fd, dword_index=0x%x\n",
               dword_index);
  return 0;
}

uint32_t MmioModel::read_register(uint32_t byte_offset) {
  if (byte_offset + 4 > kBar5SizeBytes) {
    std::fprintf(stderr, "[vfu/bar5] read out-of-range: offset=0x%x\n", byte_offset);
    return 0;
  }

  // MM_DATA: serve an indirect register read if MM_INDEX was set.
  if (byte_offset == kRegMmData) {
    uint32_t dword_idx = mm_index_ & ~0x80000000U;
    if (mm_index_ & 0x80000000U)
      return read_indirect(dword_idx);
    // Without the high bit, it's a direct read via the index — treat as 0.
    return 0;
  }

  uint32_t idx = byte_offset / 4;

  // GPU timestamp: increment a counter so the driver's clock-delta check passes.
  if (byte_offset == kRegGoldenTscCountLower) {
    uint32_t lo = static_cast<uint32_t>(timestamp_counter_++);
    regs_[kRegGoldenTscCountLower / 4]  = lo;
    regs_[kRegGoldenTscCountUpper / 4] = static_cast<uint32_t>(timestamp_counter_ >> 32);
  }

  return regs_[idx];
}

void MmioModel::write_register(uint32_t byte_offset, uint32_t value) {
  if (byte_offset + 4 > kBar5SizeBytes) {
    std::fprintf(stderr, "[vfu/bar5] write out-of-range: offset=0x%x val=0x%x\n",
                 byte_offset, value);
    return;
  }

  // MM_INDEX: capture the indirect address for subsequent MM_DATA reads.
  if (byte_offset == kRegMmIndex) {
    mm_index_ = value;
    std::fprintf(stderr, "[vfu/bar5] MM_INDEX write: 0x%08x (hi=0x%08x)\n",
                 value, mm_index_hi_);
    return;
  }

  // MM_INDEX_HI: high 32 bits of a 64-bit indirect address.
  if (byte_offset == kRegMmIndexHi) {
    mm_index_hi_ = value;
    std::fprintf(stderr, "[vfu/bar5] MM_INDEX_HI write: 0x%08x\n", value);
    return;
  }

  // Shadow the write so a subsequent read returns the written value.
  regs_[byte_offset / 4] = value;

  // SMC message: when the driver writes the message register, immediately
  // set the response to OK so it doesn't busy-wait.
  if (byte_offset == kRegMpSmcMsg)
    regs_[kRegMpSmcResp / 4] = kSmcRespOk;
}

ssize_t MmioModel::read(char *buf, size_t count, long offset) {
  std::fprintf(stderr, "[vfu/bar5] read  offset=0x%lx count=%zu\n", offset, count);
  if (count != 4) {
    std::fprintf(stderr, "[vfu/bar5] non-dword read count=%zu at 0x%lx\n",
                 count, offset);
    errno = EINVAL;
    return -1;
  }
  uint32_t val = read_register(static_cast<uint32_t>(offset));
  std::memcpy(buf, &val, 4);
  return static_cast<ssize_t>(count);
}

ssize_t MmioModel::write(const char *buf, size_t count, long offset) {
  uint32_t val = 0;
  if (count == 4) std::memcpy(&val, buf, 4);
  std::fprintf(stderr, "[vfu/bar5] write offset=0x%lx count=%zu val=0x%08x\n",
               offset, count, val);
  if (count != 4) {
    std::fprintf(stderr, "[vfu/bar5] non-dword write count=%zu at 0x%lx\n",
                 count, offset);
    errno = EINVAL;
    return -1;
  }
  write_register(static_cast<uint32_t>(offset), val);
  return static_cast<ssize_t>(count);
}

ssize_t MmioModel::access_cb(vfu_ctx_t *ctx, char *buf, size_t count,
                              long offset, bool is_write) {
  auto *model = reinterpret_cast<MmioModel *>(vfu_get_private(ctx));
  if (!model)
    return -1;
  return is_write ? model->write(buf, count, offset)
                  : model->read(buf, count, offset);
}

} // namespace rocjitsu::vfu
