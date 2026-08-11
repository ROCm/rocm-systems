// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/bar5_mmio.h"
#include "rocjitsu/vfu/minimal_vbios.h"
#include "rocjitsu/vfu/mmio_registers.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace rocjitsu::vfu {

MmioModel::~MmioModel() {
  if (bar5_mem_ && bar5_mem_ != MAP_FAILED)
    ::munmap(const_cast<uint32_t *>(bar5_mem_), kBar5SizeBytes);
  if (bar5_fd_ >= 0)
    ::close(bar5_fd_);
}

MmioModel::MmioModel(int vram_fd, uint64_t vram_size)
    : vram_fd_(vram_fd), vram_size_(vram_size) {
  // Create a shared memfd for BAR5 so QEMU maps it directly into the guest.
  // CPU writes from amdgpu go into this buffer; we poll it in fence_service_loop
  // to auto-complete ring tests without needing the vfio-user callback path.
  bar5_fd_ = ::memfd_create("rocjitsu_bar5", MFD_ALLOW_SEALING | MFD_CLOEXEC);
  if (bar5_fd_ < 0) {
    std::perror("[vfu/bar5] memfd_create");
  } else if (::ftruncate(bar5_fd_, kBar5SizeBytes) != 0) {
    std::perror("[vfu/bar5] ftruncate bar5");
    ::close(bar5_fd_); bar5_fd_ = -1;
  } else {
    void *p = ::mmap(nullptr, kBar5SizeBytes, PROT_READ | PROT_WRITE,
                     MAP_SHARED, bar5_fd_, 0);
    if (p == MAP_FAILED) {
      std::perror("[vfu/bar5] mmap bar5");
      ::close(bar5_fd_); bar5_fd_ = -1;
    } else {
      bar5_mem_ = static_cast<volatile uint32_t *>(p);
    }
  }

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

  // Pre-populate GC and MMHUB VM invalidation engine ACK registers with
  // all bits set. This makes gmc_v9_0_flush_gpu_tlb exit immediately on
  // the first poll rather than spinning for 100ms. The write-triggered ACK
  // in write_register keeps them updated when REQ is written.
  for (uint32_t eng = 0; eng < 32; ++eng) {
    uint32_t gc_ack_off = kRegGcVmInvEng0Ack + eng * kGcVmInvEngDistance;
    if (gc_ack_off + 4 <= kBar5SizeBytes)
      regs_[gc_ack_off / 4] = 0xFFFFFFFFU;

    uint32_t mm_ack_off = kRegMmhubVmInvEng0Ack + eng * kMmhubVmInvEngDistance;
    if (mm_ack_off + 4 <= kBar5SizeBytes)
      regs_[mm_ack_off / 4] = 0xFFFFFFFFU;
  }

  // BIOS scratch register 7.
  // ATOM_S7_ASIC_INIT_COMPLETE_MASK (0x200) tells amdgpu_device_need_post() that
  // the GPU has already been posted, skipping the atom POST sequence.
  //
  // The driver reads RREG32(bios_scratch_reg_offset + 7). Possible offsets:
  //   - bios_scratch_reg_offset=0 (from firmware_info.bios_scratch_reg_startaddr=0):
  //     dword 7, byte 0x1C
  //   - bios_scratch_reg_offset=mmBIOS_SCRATCH_0=0x0038 (atombios path, BASE_IDX=1):
  //     with seg1=0xE00 → dword 0xE38+7=0xE3F, byte 0x38FC
  regs_[7]    = 0x00000200U;  // bios_scratch_reg_offset=0 path
  regs_[0x3F] = 0x00000200U;  // legacy atombios path (wrong base, but pre-fill anyway)
  regs_[0xE3F]= 0x00000200U;  // atombios path with seg1=0xE00

  // RCC_CONFIG_MEMSIZE: VRAM size in MB (used by amdgpu_discovery to locate
  // the IP discovery table at vram_size_bytes - DISCOVERY_TMR_OFFSET).
  regs_[kRegRccConfigMemsize / 4] = kVramSizeMb;

  // GFX/KIQ ring test: pre-initialize SCRATCH_REG0 to 0xDEADBEEF at BOTH
  // possible offsets so RREG32 polls immediately return the expected value.
  // The amdgpu driver's GFX ring test:
  //   WREG32(scratch_reg0, 0xCAFEDEAD) → goes to QEMU shadow (not our callback)
  //   RREG32(scratch_reg0) poll → comes through our callback → returns 0xDEADBEEF
  //
  // offset = SOC15_REG_OFFSET(GC, GET_INST(GC, xcc_id), regSCRATCH_REG0)
  //   xcc_id=0, GC instance 0 (base=0x2000): dword 0x4040, byte 0x10100
  //   xcc_id=0, GC instance 1 (base=0xa000): dword 0xC040, byte 0x30100
  regs_[0x10100 / 4] = kScratchReg0TestDone;
  regs_[0x30100 / 4] = kScratchReg0TestDone;

  // Sync pre-populated shadow values into the BAR5 memfd so QEMU's direct map
  // serves the correct defaults on first read (before any vfio-user callback).
  if (bar5_mem_)
    std::memcpy(const_cast<uint32_t *>(bar5_mem_), regs_, kBar5SizeBytes);
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
  // SMUIO ROM_DATA: return 4 bytes from kMinimalVbios at the current rom_index_.
  // amdgpu_soc15_read_bios_from_rom writes a byte address to ROM_INDEX (via
  // ROM_INDEX dword), then reads ROM_DATA dwords to copy the VBIOS.
  if (mm_index_hi_ == 0 && dword_index == kSmuioRomDataDword) {
    uint32_t byte_addr = rom_index_;
    uint32_t val = 0;
    if (byte_addr + 4 <= sizeof(kMinimalVbios))
      std::memcpy(&val, kMinimalVbios + byte_addr, 4);
    rom_index_ += 4;  // auto-advance (driver reads sequentially)
    std::fprintf(stderr, "[vfu/bar5] ROM_DATA read addr=0x%x val=0x%08x\n",
                 byte_addr, val);
    return val;
  }

  // mmMP0_SMN_C2PMSG_33: polled by amdgpu_discovery_read_binary_from_mem()
  // before IP block early_init has run (so pcie_rreg is still the invalid stub).
  // The driver writes (0x16061 | 0x80000000) to MM_INDEX and reads MM_DATA.
  // Return bit 31 set immediately so the driver doesn't spin for 2 seconds.
  if (mm_index_hi_ == 0 && dword_index == kMmIndirectMp0C2pmsg33) {
    std::fprintf(stderr, "[vfu/bar5] C2PMSG_33 indirect read → fw-ready\n");
    return kC2pmsg33FwReady;
  }

  // For indirect reads of high-address registers (GC/MMHUB register space,
  // dword index >= 0x10000 and mm_index_hi_ == 0):
  // Return the shadow value, but OR in all bits of the last value written
  // to any high-address indirect register. This models "hardware processed the
  // request immediately" for polling loops (TLB flush ACK, semaphore acquire,
  // fence completion, etc.) while avoiding returning 0xFFFFFFFF for version
  // registers or other non-polling reads that check specific bit patterns.
  //
  // The strategy: return (shadow | last_indirect_write_val) so:
  //   - ACK registers (shadow=0, last_write=inv_req) return inv_req which has
  //     the vmid bit set → polling loop exits.
  //   - Version/capability registers (never written → last_write=0) return shadow.
  if (mm_index_hi_ == 0 && dword_index >= 0x10000) {
    // Return the last value written to any high-address indirect register so
    // polling loops (TLB flush ACK, semaphore acquire, fence) see their request
    // reflected back and exit immediately.
    std::fprintf(stderr, "[vfu/bar5] indirect hi-reg read dword=0x%x → 0x%08x\n",
                 dword_index, last_indirect_write_);
    return last_indirect_write_;
  }

  // If MM_INDEX_HI is non-zero this is a 64-bit VRAM access via the MM path.
  // amdgpu_device_mm_access() writes MM_INDEX_HI then MM_INDEX (with bit 31
  // set for indirect), then reads MM_DATA one dword at a time to copy the
  // IP discovery binary out of VRAM.
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

  // RSMU_DATA: serve PCIE indirect reads (amdgpu_device_indirect_rreg path).
  // rsmu_index_ holds the BYTE address of the target register.
  if (byte_offset == kRegRsmuData) {
    // SMUIO ROM_DATA: serve VBIOS bytes.
    if (rsmu_index_ == kSmuioRomDataDword * 4) {
      uint32_t byte_addr = rom_index_;
      uint32_t val = 0;
      if (byte_addr + 4 <= sizeof(kMinimalVbios))
        std::memcpy(&val, kMinimalVbios + byte_addr, 4);
      rom_index_ += 4;
      std::fprintf(stderr, "[vfu/bar5] ROM_DATA (rsmu) read addr=0x%x val=0x%08x\n",
                   byte_addr, val);
      return val;
    }
    // MP0 register space: byte addresses 0x58000–0x5BFFF (seg0=0x16000, 256 regs).
    // Return targeted values for specific C2PMSG registers:
    //   C2PMSG_64 (0x58200): MBOX_TOS_READY_FLAG (0x80000000) — PSP ring creation ready
    //   All others: 0 — no sOS/tos loaded, no firmware version mismatch
    // Returning 0 for everything else prevents psp_v13_0_is_reload_needed from
    // triggering a mode1 reset when C2PMSG_58 (sOS version) returns 0xFFFFFFFF.
    // C2PMSG_35 (0x5808C): PSP bootloader ready — polled by psp_v13_0_wait_for_bootloader.
    // C2PMSG_64 (0x58200): PSP TOS ready — polled by psp_v13_0_ring_create.
    if (rsmu_index_ == 0x5808CU || rsmu_index_ == 0x58200U) {
      return 0x80000000U;
    }
    if (rsmu_index_ >= 0x58000U && rsmu_index_ < 0x5C000U) {
      return 0x00000000U;  // all other MP0 C2PMSG: 0
    }
    // Non-MP0 high-address RSMU reads.
    // Some MMHUB ACK registers appear to go through RSMU despite being within
    // rmmio_size; return 0x1 so bit 0 is set for vmid=0 TLB flush ACK check.
    // This is safe for FB_LOCATION registers (0 << 24 = 0) because those use
    // a different address range and our MMHUB ACK pre-population handles them.
    return 0x00000001U;
  }

  // MM_DATA: serve an indirect register read if MM_INDEX was set.
  if (byte_offset == kRegMmData) {
    uint32_t dword_idx = mm_index_ & ~0x80000000U;
    if (mm_index_ & 0x80000000U)
      return read_indirect(dword_idx);
    // PCIE indirect path: MM_INDEX written WITHOUT bit 31 for high-address regs.
    if (dword_idx >= 0x10000) {
      // SMUIO ROM_DATA: serve VBIOS bytes.
      if (dword_idx == kSmuioRomDataDword) {
        uint32_t byte_addr = rom_index_;
        uint32_t val = 0;
        if (byte_addr + 4 <= sizeof(kMinimalVbios))
          std::memcpy(&val, kMinimalVbios + byte_addr, 4);
        rom_index_ += 4;
        std::fprintf(stderr, "[vfu/bar5] ROM_DATA (pcie) read addr=0x%x val=0x%08x\n",
                     byte_addr, val);
        return val;
      }
      // Other high-address reads: return last write value so polling loops exit.
      return last_indirect_write_;
    }
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

  // RSMU_INDEX: capture the indirect register address for amdgpu_device_indirect_{r,w}reg.
  // The PCIE path writes reg_addr (dword offset) here, then reads/writes RSMU_DATA.
  if (byte_offset == kRegRsmuIndex) {
    rsmu_index_ = value;
    // C2PMSG_67 (byte 0x5810C) = PSP ring WPTR register. When the driver writes
    // the WPTR, a new ring frame is ready. Scan VRAM and write fence values so the
    // kernel fence poll exits immediately.
    if (value == 0x5810CU) {
      service_psp_ring_fence();
    }
    return;
  }

  // RSMU_DATA write: handle SMUIO ROM_INDEX writes (reg_addr stored via RSMU_INDEX).
  // rsmu_index_ holds the BYTE address of the target register (passed by pcie_wreg).
  if (byte_offset == kRegRsmuData && rsmu_index_ == kSmuioRomIndexDword * 4) {
    rom_index_ = value;
    std::fprintf(stderr, "[vfu/bar5] ROM_INDEX (rsmu) write addr=0x%x\n", value);
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

  // MM_DATA write: capture for indirect register access.
  if (byte_offset == kRegMmData) {
    uint32_t dword_idx = mm_index_ & ~0x80000000U;
    if (mm_index_hi_ == 0 && dword_idx >= 0x10000) {
      // SMUIO ROM_INDEX: driver writes the byte address of the VBIOS region to read.
      if (dword_idx == kSmuioRomIndexDword) {
        rom_index_ = value;
        std::fprintf(stderr, "[vfu/bar5] ROM_INDEX write addr=0x%x\n", value);
      } else {
        last_indirect_write_ = value;
        std::fprintf(stderr, "[vfu/bar5] indirect hi-reg write dword=0x%x val=0x%08x\n",
                     dword_idx, value);
      }
    }
    return;
  }

  // Shadow the write so a subsequent read returns the written value.
  regs_[byte_offset / 4] = value;
  // Mirror into BAR5 memfd so direct-mapped QEMU reads see the same value.
  if (bar5_mem_)
    bar5_mem_[byte_offset / 4] = value;

  // MMHUB VM invalidation engine REQ → auto-ACK (same pattern as GC).
  if (byte_offset >= kRegMmhubVmInvEng0Req &&
      byte_offset < kRegMmhubVmInvEng0Req + 32 * kMmhubVmInvEngDistance &&
      (byte_offset - kRegMmhubVmInvEng0Req) % kMmhubVmInvEngDistance == 0) {
    uint32_t eng = (byte_offset - kRegMmhubVmInvEng0Req) / kMmhubVmInvEngDistance;
    uint32_t ack_off = kRegMmhubVmInvEng0Ack + eng * kMmhubVmInvEngDistance;
    if (ack_off + 4 <= kBar5SizeBytes)
      regs_[ack_off / 4] = value;
  }

  // GC VM invalidation engine REQ → auto-ACK:
  // When the driver writes to any ENG[0..31]_REQ, copy the value to the
  // corresponding ENG[n]_ACK so the poll in gmc_v9_0_flush_gpu_tlb exits
  // immediately (bit vmid is set in the written inv_req).
  if (byte_offset >= kRegGcVmInvEng0Req &&
      byte_offset < kRegGcVmInvEng0Req + 32 * kGcVmInvEngDistance &&
      (byte_offset - kRegGcVmInvEng0Req) % kGcVmInvEngDistance == 0) {
    uint32_t eng = (byte_offset - kRegGcVmInvEng0Req) / kGcVmInvEngDistance;
    uint32_t ack_off = kRegGcVmInvEng0Ack + eng * kGcVmInvEngDistance;
    if (ack_off + 4 <= kBar5SizeBytes)
      regs_[ack_off / 4] = value;
  }

  // SMC message: when the driver writes the message register, immediately
  // set the response to OK so it doesn't busy-wait.
  if (byte_offset == kRegMpSmcMsg)
    regs_[kRegMpSmcResp / 4] = kSmcRespOk;

  // GFX/KIQ ring test auto-complete: when the ring test writes 0xCAFEDEAD to
  // SCRATCH_REG0 (init sentinel), immediately update to 0xDEADBEEF so the
  // RREG32 poll exits immediately without CP execution.
  // SCRATCH_REG0 is accessed at two possible offsets depending on which GC
  // instance the driver uses:
  //   GC instance 0 (base=0x2000): dword 0x4040, byte 0x10100
  //   GC instance 1 (base=0xa000): dword 0xC040, byte 0x30100
  if (value == kScratchReg0TestInit &&
      (byte_offset == kRegScratchReg0 || byte_offset == 0x30100)) {
    regs_[byte_offset / 4] = kScratchReg0TestDone;
  }
}

ssize_t MmioModel::read(char *buf, size_t count, long offset) {
  if (count != 1 && count != 2 && count != 4) {
    std::fprintf(stderr, "[vfu/bar5] unsupported read count=%zu at 0x%lx\n",
                 count, offset);
    errno = EINVAL;
    return -1;
  }
  // Read the aligned dword and extract the requested bytes.
  uint32_t aligned = static_cast<uint32_t>(offset) & ~3u;
  uint32_t byte_shift = static_cast<uint32_t>(offset) & 3u;
  uint32_t dword = read_register(aligned);
  std::memcpy(buf, reinterpret_cast<const char *>(&dword) + byte_shift, count);
  return static_cast<ssize_t>(count);
}

ssize_t MmioModel::write(const char *buf, size_t count, long offset) {
  if (count != 1 && count != 2 && count != 4) {
    std::fprintf(stderr, "[vfu/bar5] unsupported write count=%zu at 0x%lx\n",
                 count, offset);
    errno = EINVAL;
    return -1;
  }
  uint32_t val = 0;
  if (count == 4) {
    std::memcpy(&val, buf, 4);
  } else {
    // Sub-dword write: read-modify-write the shadow.
    uint32_t aligned = static_cast<uint32_t>(offset) & ~3u;
    uint32_t byte_shift = static_cast<uint32_t>(offset) & 3u;
    if (aligned + 4 <= kBar5SizeBytes)
      val = regs_[aligned / 4];
    std::memcpy(reinterpret_cast<char *>(&val) + byte_shift, buf, count);
    offset = aligned;
  }
  write_register(static_cast<uint32_t>(offset), val);
  return static_cast<ssize_t>(count);
}

void MmioModel::service_psp_ring_fence() {
  if (vram_fd_ < 0 || vram_size_ == 0)
    return;

  // Scan first 4 MB of VRAM for PSP ring frames (64 bytes each, aligned to 64).
  // A valid frame has fence_addr in VRAM range and fence_value > 0.
  constexpr uint64_t scan_size = 4ULL * 1024 * 1024;
  constexpr size_t frame_size = 64;

  void *p = ::mmap(nullptr, scan_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   vram_fd_, static_cast<off_t>(kFbStart));
  if (p == MAP_FAILED)
    return;

  auto *mem = static_cast<uint8_t *>(p);
  for (size_t off = 0; off + frame_size <= scan_size; off += frame_size) {
    const auto *frame = reinterpret_cast<const uint32_t *>(mem + off);
    uint32_t fence_lo  = frame[3];  // fence_addr_lo
    uint32_t fence_hi  = frame[4];  // fence_addr_hi
    uint32_t fence_val = frame[5];  // fence_value

    if (fence_val == 0 || fence_hi != 0)
      continue;  // skip empty frames or 64-bit fence addresses

    // fence_addr is a GPU MC address; convert to memfd offset
    uint64_t fence_gpu = static_cast<uint64_t>(fence_lo);
    if (fence_gpu < kFbStart || fence_gpu >= kFbStart + vram_size_)
      continue;

    // fence_offset = bytes from kFbStart; mem points to vram[kFbStart]
    uint64_t fence_off_in_scan = fence_gpu - kFbStart;
    if (fence_off_in_scan >= scan_size)
      continue;  // outside scanned region
    auto *fence_ptr = reinterpret_cast<uint32_t *>(mem + fence_off_in_scan);
    if (*fence_ptr != fence_val) {
      *fence_ptr = fence_val;
    }
  }

  ::munmap(p, scan_size);
}

} // namespace rocjitsu::vfu
