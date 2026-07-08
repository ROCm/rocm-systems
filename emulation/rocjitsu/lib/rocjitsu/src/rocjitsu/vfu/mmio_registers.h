// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mmio_registers.h
/// @brief BAR5 MMIO register offset constants for GFX950 (MI350P / CDNA4).
///
/// All offsets are byte addresses within BAR5 (256 KB, non-prefetchable).
/// The register word index used by the amdgpu driver is the dword offset;
/// multiply by 4 to get the byte offset stored here.
///
/// Sources: drivers/gpu/drm/amd/include/  (IP register header files),
///          amdgpu_device.c RREG32/WREG32 usage for aqua_vanjaram / gc_9_4_4.

#ifndef ROCJITSU_VFU_MMIO_REGISTERS_H_
#define ROCJITSU_VFU_MMIO_REGISTERS_H_

#include <cstdint>

namespace rocjitsu::vfu {

// ---------------------------------------------------------------------------
// Register byte-offset constants within BAR5 (256 KB direct window).
// All offsets are byte addresses: dword_index * 4.
// Registers at dword index >= 0x10000 are accessed via MM_INDEX/MM_DATA.
// ---------------------------------------------------------------------------

// MM indirect access registers (always at the start of BAR5)
// amdgpu RREG32() writes (dword_index | 0x80000000) to MM_INDEX, then reads MM_DATA.
inline constexpr uint32_t kRegMmIndex              = 0x0000 * 4; ///< MM indirect index.
inline constexpr uint32_t kRegMmData               = 0x0001 * 4; ///< MM indirect data.
inline constexpr uint32_t kRegMmIndexHi            = 0x0006 * 4; ///< MM indirect index high (for >32-bit).

// GFX engine status
inline constexpr uint32_t kRegGrbmStatus           = 0x8010 * 4; ///< GFX/CP busy bits.
inline constexpr uint32_t kRegGrbmStatus2          = 0x8012 * 4; ///< Additional busy bits.
inline constexpr uint32_t kRegGrbmSoftReset        = 0x8020 * 4; ///< GFX soft reset control.

// CP firmware / handshake
inline constexpr uint32_t kRegCpMecFwShadescrSize  = 0xC812 * 4; ///< MEC FW descriptor size.
inline constexpr uint32_t kRegRlcGpuClock          = 0xD800 * 4; ///< RLC GPU clock counter low.
inline constexpr uint32_t kRegRlcGpuClockHi        = 0xD801 * 4; ///< RLC GPU clock counter high.
inline constexpr uint32_t kRegRlcSpm               = 0xD840 * 4; ///< RLC SPM control.
inline constexpr uint32_t kRegRlcSafeMode          = 0xD80E * 4; ///< RLC safe-mode status.
inline constexpr uint32_t kRegRlcPgfsm             = 0xD84F * 4; ///< RLC PG state machine.
inline constexpr uint32_t kRegRlcCntlToken         = 0xD8C0 * 4; ///< RLC control token.

// SDMA version / status (engines 0–4 for MI350P; stride 0x200 per engine)
inline constexpr uint32_t kRegSdma0Version         = 0xD200 * 4; ///< SDMA0 version register.
inline constexpr uint32_t kSdmaEngineStride        = 0x0200 * 4; ///< Byte stride between SDMAs.

// GPU timestamp (read by ROCr on startup)
inline constexpr uint32_t kRegGoldenTscCountUpper  = 0xD82E * 4; ///< Timestamp upper 32 bits.
inline constexpr uint32_t kRegGoldenTscCountLower  = 0xD82F * 4; ///< Timestamp lower 32 bits.

// Interrupt handler ring buffer
inline constexpr uint32_t kRegIhRbBase             = 0x3E10 * 4; ///< IH ring base low.
inline constexpr uint32_t kRegIhRbBaseHi           = 0x3E11 * 4; ///< IH ring base high.
inline constexpr uint32_t kRegIhRbRptr             = 0x3E12 * 4; ///< IH ring read pointer.
inline constexpr uint32_t kRegIhRbWptr             = 0x3E13 * 4; ///< IH ring write pointer.
inline constexpr uint32_t kRegIhRbCntl             = 0x3E14 * 4; ///< IH ring control.
inline constexpr uint32_t kRegIhCntl               = 0x3E80 * 4; ///< IH global control.

// GPU memory controller (GMC / VMC)
inline constexpr uint32_t kRegGmcVmFbOffset        = 0x5480 * 4; ///< Framebuffer base (BAR0 offset).
inline constexpr uint32_t kRegGmcVmSysApertureHi   = 0x548A * 4; ///< AGP aperture high address.
inline constexpr uint32_t kRegGmcVmSysApertureLo   = 0x548B * 4; ///< AGP aperture low address.

// PCIe BIF
inline constexpr uint32_t kRegBifBxDevZeroId       = 0x1480 * 4; ///< PCIe device ID (BIF).

// HDP flush (KFD MMIO remap)
inline constexpr uint32_t kRegHdpMemFlushCntl      = 0x00B8 * 4; ///< HDP memory flush trigger.
inline constexpr uint32_t kRegHdpRegFlushCntl      = 0x00BA * 4; ///< HDP register flush trigger.

// Doorbell routing (aqua_vanjaram / GFX950)
// Doorbell index must not exceed 0x1FF per BIF_DOORBELLx_RANGE_OFFSET_ENTRY.
inline constexpr uint32_t kRegDoorbellRangeOffset  = 0x1580 * 4; ///< Doorbell range offset entry.

// SMU / power management (stubbed — amdgpu polls but accepts 0)
inline constexpr uint32_t kRegMpSmcMsg             = 0x3B00 * 4; ///< SMC message register.
inline constexpr uint32_t kRegMpSmcMsgArg          = 0x3B01 * 4; ///< SMC message argument.
inline constexpr uint32_t kRegMpSmcResp            = 0x3B02 * 4; ///< SMC response (1 = OK).

// IP discovery: registers used by amdgpu_discovery_read_binary_from_mem().
//
// mmRCC_CONFIG_MEMSIZE (dword 0xde3 = byte 0x378C): returns VRAM size in MB.
//   amdgpu computes: vram_size = (RREG32(mmRCC_CONFIG_MEMSIZE) << 20)
//   then places the discovery table at: vram_size - DISCOVERY_TMR_OFFSET (64 KB)
//   Value for 256 MB BAR: 256.
inline constexpr uint32_t kRegRccConfigMemsize     = 0x0de3 * 4; ///< VRAM size in MB.

// mmMP0_SMN_C2PMSG_33 (dword 0x16061): accessed via MM_INDEX/MM_DATA indirect.
//   amdgpu polls this until bit 31 is set (firmware initialisation complete).
//   We must return 0x80000000 immediately so the driver doesn't spin.
inline constexpr uint32_t kMmIndirectMp0C2pmsg33  = 0x16061; ///< Dword index (not byte) for indirect.

// ---------------------------------------------------------------------------
// Key reset / init values returned by the register model.
// ---------------------------------------------------------------------------

/// GRBM_STATUS idle value: all busy bits clear (GPU ready for commands).
inline constexpr uint32_t kGrbmStatusIdle = 0x00000000;

/// SDMA version stub for SDMA 4.4.5 (MI350P firmware version).
inline constexpr uint32_t kSdmaVersionStub = 0x00040405;

/// SMC response "OK" (driver polls until non-zero).
inline constexpr uint32_t kSmcRespOk = 0x00000001;

/// RCC_CONFIG_MEMSIZE: 256 MB default VRAM BAR reported as 256 (in MB).
inline constexpr uint32_t kVramSizeMb = 256;

/// C2PMSG_33 firmware-ready bit (bit 31 set = PSP/firmware init complete).
inline constexpr uint32_t kC2pmsg33FwReady = 0x80000000;

/// BAR5 window size in bytes.
inline constexpr uint32_t kBar5SizeBytes = 256 * 1024;

} // namespace rocjitsu::vfu

#endif // ROCJITSU_VFU_MMIO_REGISTERS_H_
