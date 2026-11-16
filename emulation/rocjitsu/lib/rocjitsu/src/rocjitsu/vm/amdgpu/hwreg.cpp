// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hwreg.cpp
/// @brief Shader-visible AMDGPU hardware register access helpers.

#include "rocjitsu/vm/amdgpu/hwreg.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {
namespace {

enum class HwregState : uint8_t {
  Unsupported,
  Mode,
  Status,
  Gfx11Status,
  PcLo,
  PcHi,
  ApertureBases,
  ShaderCycles,
  FlushIb,
  IbStsGfx11,
  IbSts2Gfx11,
  IbDbgGfx11,
  HwId1Gfx11,
  GprAllocGfx11,
  LdsAllocGfx11,
  StatePrivGfx12,
  ExcpFlagPrivGfx12,
  ExcpFlagUserGfx12,
  TrapCtrlGfx12,
  XnackStatePrivGfx1250,
  XnackMaskGfx1250,
  Trapsts,
  SimdIdCdna2,
  LocationCdna3_4,
  XccIdCdna3_4,
  StatusGfx10,
  LdsAllocGfx10,
  IbStsGfx10,
  ShMemBasesGfx10,
  ShaderTbaLoGfx10,
  ShaderTbaHiGfx10,
  ShaderTmaLoGfx10,
  ShaderTmaHiGfx10,
  ShaderCyclesGfx1030,
  GprAllocGfx9_10,
  GprAllocCdna3_4,
  LdsAllocCdna,
  IbStsCdna,
  WaveSchedMode,
  FlatScratchLo,
  FlatScratchHi,
  PopsPackerGfx10,
  TbaLo,
  TbaHi,
  TmaLo,
  TmaHi,
  IbStsGfx1250,
  IbSts2Gfx1250,
  WgpIdGfx1250,
};

enum class HwregWritePolicy : uint8_t {
  // ReadOnly means this unprivileged shader HWREG path cannot write it. Some
  // hardware registers are trap-handler writable, which rocjitsu does not
  // model for this path yet.
  ReadOnly,
  UserWritable,
  Privileged,
};

struct HwregDescriptor {
  uint32_t id;
  const char *name;
  HwregState state;
  HwregWritePolicy write_policy;
};

struct HwregTable {
  const HwregDescriptor *entries;
  size_t size;
};

struct DecodedHwreg {
  uint32_t id;
  uint32_t offset;
  uint32_t size;
  uint32_t mask;
};

uint32_t mask_to_bits(uint32_t value, uint32_t bits) {
  if (bits >= 32)
    return value;
  return value & ((1u << bits) - 1u);
}

uint32_t field_value(uint32_t value, uint32_t shift, uint32_t bits) {
  return mask_to_bits(value, bits) << shift;
}

uint32_t blocks_minus_one(uint32_t count, uint32_t block_size) {
  if (count == 0)
    return 0;
  return ((count + block_size - 1u) / block_size) - 1u;
}

uint32_t cdna2_4_simd_id_raw(const Wavefront &wf) {
  // CDNA2/3/4 define SIMD_ID at HW_ID[5:4]. Assign permanent CU wave slots
  // cyclically to four logical SIMDs. Slot identity survives reuse and
  // checkpoint restoration; no timing model is implied.
  return field_value(wf.wf_id() % 4u, 4, 2);
}

uint32_t gfx9_10_gpr_alloc_raw(const Wavefront &wf) {
  const RegAllocation &vgpr = wf.vgpr_alloc();
  const RegAllocation &sgpr = wf.sgpr_alloc();
  return field_value(vgpr.base >> 2, 0, 6) | field_value(blocks_minus_one(vgpr.count, 4), 8, 6) |
         field_value(sgpr.base >> 3, 16, 6) | field_value(blocks_minus_one(sgpr.count, 8), 24, 4);
}

uint32_t cdna3_4_gpr_alloc_raw(const Wavefront &wf) {
  const RegAllocation &vgpr = wf.vgpr_alloc();
  const RegAllocation &sgpr = wf.sgpr_alloc();
  // ACCV_OFF is architected in bits 17:12. read_hwreg_field rejects reads that
  // intersect it until rocjitsu backs the per-wave AccVGPR allocation base.
  return field_value(vgpr.base >> 2, 0, 6) | field_value(blocks_minus_one(vgpr.count, 4), 6, 6) |
         field_value(0, 12, 6) | field_value(sgpr.base >> 3, 18, 6) |
         field_value(blocks_minus_one(sgpr.count, 16), 24, 4);
}

uint32_t gfx1250_ib_sts2_raw(const Wavefront &wf) {
  const WaitCounters &counters = wf.wait_counters();
  return field_value(counters.kmcnt, 0, 5) | field_value(wf.cluster_size() > 1 ? 1u : 0u, 6, 4) |
         field_value(counters.asynccnt, 12, 6) |
         field_value(std::min<uint32_t>(counters.tensorcnt, 3u), 18, 2) |
         field_value(wf.cluster_rank(), 21, 4) | field_value(3u, 28, 2);
}

uint32_t gfx1250_ib_sts_raw(const Wavefront &wf) {
  const WaitCounters &counters = wf.wait_counters();
  return field_value(counters.dscnt, 3, 6) | field_value(counters.vmcnt, 9, 6) |
         field_value(counters.vscnt, 24, 6);
}

// RocJITsu keeps the scalar condition, priorities, and halt state in its
// common internal STATUS word. GFX12 exposes those fields through
// STATE_PRIV instead, so shader HWREG accesses need an explicit translation.
// SCRATCH_EN is derived from the live scratch allocation just as the hardware
// initializes it at wave creation.
void set_gfx12_state_priv_raw(Wavefront &wf, uint32_t value) {
  wf.set_status_raw(update_status_from_gfx12_state_priv(wf.status_raw(), value));
  set_gfx12_state_priv_workgroup_flags(wf, value);
  // A privileged STATE_PRIV write, rather than S_SENDMSGHALT, owns HALT.
  wf.set_self_halted(false);
}

// GFX12 split the legacy TRAPSTS/MODE state into EXCP_FLAG_PRIV,
// EXCP_FLAG_USER and TRAP_CTRL. RocJITsu keeps the execution-facing causes in
// its common representation, so shader HWREG accesses translate the complete
// architected field inventory in both directions. EXCP_FLAG_USER's two texture
// flags have no legacy slot and are held separately on the wave.
uint32_t gfx12_excp_flag_priv_raw(const Wavefront &wf) {
  return gfx12_excp_flag_priv_from_trapsts(wf.trapsts());
}

void set_gfx12_excp_flag_priv_raw(Wavefront &wf, uint32_t value) {
  wf.set_trapsts(update_trapsts_from_gfx12_excp_flag_priv(wf.trapsts(), value));
}

uint32_t gfx12_excp_flag_user_raw(const Wavefront &wf) {
  return (wf.trapsts() & 0x7Fu) | wf.gfx12_excp_flag_user_extra_raw();
}

void set_gfx12_excp_flag_user_raw(Wavefront &wf, uint32_t value) {
  wf.set_trapsts(update_trapsts_from_gfx12_excp_flag_user(wf.trapsts(), value));
  wf.set_gfx12_excp_flag_user_extra_raw(value & (0x3u << 30));
}

uint32_t gfx12_trap_ctrl_raw(const Wavefront &wf) { return wf.gfx12_trap_ctrl_raw(); }

void set_gfx12_trap_ctrl_raw(Wavefront &wf, uint32_t value) { wf.set_gfx12_trap_ctrl_raw(value); }

// UserWritable records ISA privilege, not simulator completeness. If a
// user-writable register has no HwregState backing yet, writes return
// Unsupported rather than fabricating hidden state or side effects.

// CDNA2 adds backed wave state to the shared CDNA1/2 register numbering.
template <bool Cdna2>
constexpr HwregDescriptor CDNA1_2_HWREGS[] = {
    // CDNA-family ISA XML uses MODE=1 and STATUS=2. Keep XML-only and
    // unmodeled state explicit but unsupported until the corresponding VM
    // state, field layouts, and privilege side effects are represented.
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::Status, HwregWritePolicy::Privileged},
    {3, "TRAPSTS", HwregState::Trapsts, HwregWritePolicy::UserWritable},
    {4, "HW_ID", Cdna2 ? HwregState::SimdIdCdna2 : HwregState::Unsupported,
     HwregWritePolicy::ReadOnly},
    {5, "GPR_ALLOC", HwregState::GprAllocGfx9_10, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::LdsAllocCdna, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::IbStsCdna, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "INST_DW0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "INST_DW1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "IB_DBG0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {16, "SQ_SHADER_TBA_LO", HwregState::TbaLo, HwregWritePolicy::ReadOnly},
    {17, "SQ_SHADER_TBA_HI", HwregState::TbaHi, HwregWritePolicy::ReadOnly},
    {18, "SQ_SHADER_TMA_LO", HwregState::TmaLo, HwregWritePolicy::ReadOnly},
    {19, "SQ_SHADER_TMA_HI", HwregState::TmaHi, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor CDNA3_4_HWREGS[] = {
    // MODE/STATUS, TRAPSTS, GPR_ALLOC, HW_ID location fields, and XCC_ID
    // have modeled sources in wave state and CU/CP topology.
    // The other XML HWREG IDs stay named but unsupported until their VM state,
    // field layouts and privilege side effects are represented.
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::Status, HwregWritePolicy::Privileged},
    {3, "TRAPSTS", HwregState::Trapsts, HwregWritePolicy::UserWritable},
    {4, "HW_ID", HwregState::LocationCdna3_4, HwregWritePolicy::ReadOnly},
    {5, "GPR_ALLOC", HwregState::GprAllocCdna3_4, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::LdsAllocCdna, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::IbStsCdna, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "INST_DW0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "INST_DW1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "IB_DBG0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {16, "SQ_SHADER_TBA_LO", HwregState::TbaLo, HwregWritePolicy::ReadOnly},
    {17, "SQ_SHADER_TBA_HI", HwregState::TbaHi, HwregWritePolicy::ReadOnly},
    {18, "SQ_SHADER_TMA_LO", HwregState::TmaLo, HwregWritePolicy::ReadOnly},
    {19, "SQ_SHADER_TMA_HI", HwregState::TmaHi, HwregWritePolicy::ReadOnly},
    {20, "XCC_ID", HwregState::XccIdCdna3_4, HwregWritePolicy::ReadOnly},
    {21, "SQ_PERF_SNAPSHOT_DATA", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {22, "SQ_PERF_SNAPSHOT_DATA1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {23, "SQ_PERF_SNAPSHOT_PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {24, "SQ_PERF_SNAPSHOT_PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor RDNA1_HWREGS[] = {
    // RDNA1 XML and ISA prose use MODE=1, STATUS=2. Remaining XML HWREG IDs
    // are named for diagnostics but stay unsupported without modeled VM state.
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::StatusGfx10, HwregWritePolicy::Privileged},
    {3, "TRAPSTS", HwregState::Trapsts, HwregWritePolicy::UserWritable},
    {4, "HW_ID_LEGACY", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {5, "GPR_ALLOC", HwregState::GprAllocGfx9_10, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::LdsAllocGfx10, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::IbStsGfx10, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "INST_DW0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "INST_DW1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "IB_DBG0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::ShMemBasesGfx10, HwregWritePolicy::UserWritable},
    {16, "SHADER_TBA_LO", HwregState::ShaderTbaLoGfx10, HwregWritePolicy::Privileged},
    {17, "SHADER_TBA_HI", HwregState::ShaderTbaHiGfx10, HwregWritePolicy::Privileged},
    {18, "SHADER_TMA_LO", HwregState::ShaderTmaLoGfx10, HwregWritePolicy::Privileged},
    {19, "SHADER_TMA_HI", HwregState::ShaderTmaHiGfx10, HwregWritePolicy::Privileged},
    {20, "SHADER_FLAT_SCRATCH_LO", HwregState::FlatScratchLo, HwregWritePolicy::UserWritable},
    {21, "SHADER_FLAT_SCRATCH_HI", HwregState::FlatScratchHi, HwregWritePolicy::UserWritable},
    {22, "SHADER_XNACK_MASK", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {23, "HW_ID1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {24, "HW_ID2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {25, "POPS_PACKER", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {26, "SCHED_MODE", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {27, "VGPR_OFFSET", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {28, "IB_STS2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor RDNA2_HWREGS[] = {
    // RDNA2 XML and ISA prose use MODE=1, STATUS=2. Remaining XML HWREG IDs
    // are named for diagnostics but stay unsupported without modeled VM state.
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::StatusGfx10, HwregWritePolicy::Privileged},
    {3, "TRAPSTS", HwregState::Trapsts, HwregWritePolicy::UserWritable},
    {5, "GPR_ALLOC", HwregState::GprAllocGfx9_10, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::LdsAllocGfx10, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::IbStsGfx10, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "INST_DW0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::ShMemBasesGfx10, HwregWritePolicy::UserWritable},
    {16, "SHADER_TBA_LO", HwregState::ShaderTbaLoGfx10, HwregWritePolicy::Privileged},
    {17, "SHADER_TBA_HI", HwregState::ShaderTbaHiGfx10, HwregWritePolicy::Privileged},
    {18, "SHADER_TMA_LO", HwregState::ShaderTmaLoGfx10, HwregWritePolicy::Privileged},
    {19, "SHADER_TMA_HI", HwregState::ShaderTmaHiGfx10, HwregWritePolicy::Privileged},
    {20, "SHADER_FLAT_SCRATCH_LO", HwregState::FlatScratchLo, HwregWritePolicy::UserWritable},
    {21, "SHADER_FLAT_SCRATCH_HI", HwregState::FlatScratchHi, HwregWritePolicy::UserWritable},
    {23, "HW_ID1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {24, "HW_ID2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {25, "POPS_PACKER", HwregState::PopsPackerGfx10, HwregWritePolicy::ReadOnly},
    {26, "SCHED_MODE", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {27, "VGPR_OFFSET", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {28, "IB_STS2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {29, "SHADER_CYCLES", HwregState::ShaderCyclesGfx1030, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor RDNA3_HWREGS[] = {
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::Gfx11Status, HwregWritePolicy::Privileged},
    {3, "TRAPSTS", HwregState::Trapsts, HwregWritePolicy::UserWritable},
    {5, "GPR_ALLOC", HwregState::GprAllocGfx11, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::LdsAllocGfx11, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::IbStsGfx11, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::PcLo, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::PcHi, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::IbDbgGfx11, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::FlushIb, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::ApertureBases, HwregWritePolicy::ReadOnly},
    {16, "SHADER_TBA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {17, "SHADER_TBA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {18, "PERF_SNAPSHOT_PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {19, "PERF_SNAPSHOT_PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {20, "SHADER_FLAT_SCRATCH_LO", HwregState::FlatScratchLo, HwregWritePolicy::Privileged},
    {21, "SHADER_FLAT_SCRATCH_HI", HwregState::FlatScratchHi, HwregWritePolicy::Privileged},
    {23, "HW_ID1", HwregState::HwId1Gfx11, HwregWritePolicy::ReadOnly},
    {24, "HW_ID2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {25, "POPS_PACKER", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {26, "SCHED_MODE", HwregState::WaveSchedMode, HwregWritePolicy::ReadOnly},
    {27, "PERF_SNAPSHOT_DATA", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {28, "IB_STS2", HwregState::IbSts2Gfx11, HwregWritePolicy::ReadOnly},
    {29, "SHADER_CYCLES", HwregState::ShaderCycles, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor RDNA4_HWREGS[] = {
    // RDNA4 ISA manual section 3.4 and OPR_HWREG XML. ID 28 (IB_STS2)
    // appears in prose only, so keep it as a named unsupported read-only entry.
    {1, "WAVE_MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "WAVE_STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {4, "WAVE_STATE_PRIV", HwregState::StatePrivGfx12, HwregWritePolicy::Privileged},
    {5, "WAVE_GPR_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {6, "WAVE_LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "PERF_SNAPSHOT_DATA", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "PERF_SNAPSHOT_PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "PERF_SNAPSHOT_PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {15, "PERF_SNAPSHOT_DATA1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {16, "PERF_SNAPSHOT_DATA2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {17, "WAVE_EXCP_FLAG_PRIV", HwregState::ExcpFlagPrivGfx12, HwregWritePolicy::Privileged},
    {18, "WAVE_EXCP_FLAG_USER", HwregState::ExcpFlagUserGfx12, HwregWritePolicy::UserWritable},
    {19, "WAVE_TRAP_CTRL", HwregState::TrapCtrlGfx12, HwregWritePolicy::Privileged},
    {20, "WAVE_SCRATCH_BASE_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {21, "WAVE_SCRATCH_BASE_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {23, "WAVE_HW_ID1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {24, "WAVE_HW_ID2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {28, "IB_STS2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {29, "SHADER_CYCLES_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {30, "SHADER_CYCLES_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor GFX1250_HWREGS[] = {
    // gfx1250 XML follows the RDNA4/GFX12 naming pattern. Where XML omits
    // privilege policy, inherit RDNA4 prose unless contradicted by gfx1250
    // data; this is intentionally decoupled from the MODE[27] profile
    // departure recorded in cdna5::Isa.
    {1, "WAVE_MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "WAVE_STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {4, "WAVE_STATE_PRIV", HwregState::StatePrivGfx12, HwregWritePolicy::Privileged},
    {5, "WAVE_GPR_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {6, "WAVE_LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::IbStsGfx1250, HwregWritePolicy::ReadOnly},
    {10, "PERF_SNAPSHOT_DATA", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "PERF_SNAPSHOT_PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "PERF_SNAPSHOT_PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "PERF_SNAPSHOT_DATA1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {16, "PERF_SNAPSHOT_DATA2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {17, "WAVE_EXCP_FLAG_PRIV", HwregState::ExcpFlagPrivGfx12, HwregWritePolicy::Privileged},
    {18, "WAVE_EXCP_FLAG_USER", HwregState::ExcpFlagUserGfx12, HwregWritePolicy::UserWritable},
    {19, "WAVE_TRAP_CTRL", HwregState::TrapCtrlGfx12, HwregWritePolicy::Privileged},
    {20, "WAVE_SCRATCH_BASE_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {21, "WAVE_SCRATCH_BASE_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {23, "WAVE_HW_ID1", HwregState::WgpIdGfx1250, HwregWritePolicy::ReadOnly},
    {24, "WAVE_HW_ID2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {26, "WAVE_SCHED_MODE", HwregState::WaveSchedMode, HwregWritePolicy::UserWritable},
    {28, "IB_STS2", HwregState::IbSts2Gfx1250, HwregWritePolicy::ReadOnly},
    {33, "WAVE_XNACK_STATE_PRIV", HwregState::XnackStatePrivGfx1250, HwregWritePolicy::Privileged},
    {34, "WAVE_XNACK_MASK", HwregState::XnackMaskGfx1250, HwregWritePolicy::Privileged},
};

template <size_t N> constexpr HwregTable make_table(const HwregDescriptor (&entries)[N]) {
  return {entries, N};
}

HwregTable table_for_arch(rj_code_arch_t arch) {
  /*
   * \NPI new ISA architecture: define its own HwregDescriptor table with the \
   * shader HWREG inventory, modeled state, and privilege policy, then add an \
   * explicit case here.
   */
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
    return make_table(CDNA1_2_HWREGS<false>);
  case ROCJITSU_CODE_ARCH_CDNA2:
    return make_table(CDNA1_2_HWREGS<true>);
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return make_table(CDNA3_4_HWREGS);
  case ROCJITSU_CODE_ARCH_RDNA1:
    return make_table(RDNA1_HWREGS);
  case ROCJITSU_CODE_ARCH_RDNA2:
    return make_table(RDNA2_HWREGS);
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return make_table(RDNA3_HWREGS);
  case ROCJITSU_CODE_ARCH_RDNA4:
    return make_table(RDNA4_HWREGS);
  case ROCJITSU_CODE_ARCH_CDNA5:
    return make_table(GFX1250_HWREGS);
  default:
    return {nullptr, 0};
  }
}

const HwregDescriptor *find_descriptor(rj_code_arch_t arch, uint32_t id) {
  HwregTable table = table_for_arch(arch);
  for (size_t i = 0; i < table.size; ++i) {
    if (table.entries[i].id == id)
      return &table.entries[i];
  }
  return nullptr;
}

DecodedHwreg decode_hwreg(uint16_t hwreg) {
  DecodedHwreg decoded{};
  decoded.id = hwreg_id(hwreg);
  decoded.offset = (hwreg >> 6) & 0x1Fu;
  decoded.size = ((hwreg >> 11) & 0x1Fu) + 1u;
  if (decoded.offset + decoded.size > 32u)
    decoded.size = 32u - decoded.offset;
  decoded.mask = (decoded.size == 32u) ? 0xFFFFFFFFu : ((1u << decoded.size) - 1u);
  return decoded;
}

bool field_intersects(const DecodedHwreg &decoded, uint32_t offset, uint32_t size) {
  return decoded.offset < offset + size && offset < decoded.offset + decoded.size;
}

// Observe only the requested field, never the other bits read to preserve a
// partial HWREG write. GFX12 moved SCC from STATUS[0] to STATE_PRIV[9].
bool field_aliases_scc(const Wavefront &wf, HwregState state, const DecodedHwreg &decoded) {
  if (state == HwregState::StatePrivGfx12)
    return field_intersects(decoded, 9, 1);
  const auto arch = wf.cu().arch();
  return (state == HwregState::Status || state == HwregState::StatusGfx10 ||
          state == HwregState::Gfx11Status) &&
         arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA5 &&
         field_intersects(decoded, 0, 1);
}

HwregAccessResult read_raw_hwreg(Wavefront &wf, HwregState state, uint32_t &raw_value) {
  switch (state) {
  case HwregState::Mode:
    raw_value = wf.mode_raw();
    return HwregAccessResult::Success;
  case HwregState::Gfx11Status:
    raw_value = (wf.status_raw() & ~((7u << 9) | (1u << 12) | (1u << 16) | (1u << 28))) |
                field_value(wf.exec() == 0, 9, 1) | field_value(wf.vcc_mask() == 0, 10, 1) |
                field_value(wf.cu().in_multiwave_workgroup(), 11, 1) |
                field_value(wf.state() == WfState::BARRIER, 12, 1) |
                field_value(!wf.is_halted(), 16, 1) |
                field_value(wf.wait_counters().empty(), 28, 1);
    return HwregAccessResult::Success;
  case HwregState::GprAllocGfx11: {
    // The GFX11 CWSR handler interprets VGPR_SIZE+1 in units of four
    // registers per lane for both wave sizes, independently of the descriptor.
    raw_value = field_value(wf.vgpr_alloc().base / 4, 0, 9) |
                field_value(blocks_minus_one(wf.num_vgprs(), 4), 12, 8);
    return HwregAccessResult::Success;
  }
  case HwregState::LdsAllocGfx11:
    raw_value =
        field_value(wf.lds_base() / 256, 0, 9) | field_value((wf.lds_size() + 255) / 256, 12, 9);
    return HwregAccessResult::Success;
  case HwregState::IbSts2Gfx11:
    raw_value = wf.ib_sts2_gfx11();
    return HwregAccessResult::Success;
  case HwregState::IbDbgGfx11:
    raw_value = field_value(wf.wait_counters().empty(), 24, 1);
    return HwregAccessResult::Success;
  case HwregState::HwId1Gfx11: {
    const auto &cu = wf.cu();
    const uint32_t index = cu.shader_array_cu_id();
    raw_value = field_value(wf.wf_id() / 2, 0, 5) |
                field_value(2 * (index & 1) + wf.wf_id() % 2, 8, 2) |
                field_value(index / 2, 10, 4) | field_value(cu.shader_array_id(), 16, 1) |
                field_value(cu.shader_engine_id(), 18, 3);
    return HwregAccessResult::Success;
  }
  case HwregState::IbStsGfx11: {
    const auto &counters = wf.wait_counters();
    raw_value = field_value(counters.expcnt, 0, 3) | field_value(counters.lgkmcnt, 4, 6) |
                field_value(counters.vmcnt, 10, 6) | field_value(counters.vscnt, 26, 6);
    return HwregAccessResult::Success;
  }
  case HwregState::FlushIb:
    return HwregAccessResult::Unsupported; // Write-only.
  case HwregState::PcLo:
    raw_value = static_cast<uint32_t>(wf.pc);
    return HwregAccessResult::Success;
  case HwregState::PcHi:
    raw_value = static_cast<uint32_t>(wf.pc >> 32);
    return HwregAccessResult::Success;
  case HwregState::ApertureBases:
    raw_value = field_value(wf.private_aperture_base() >> 32, 0, 16) |
                field_value(wf.shared_aperture_base() >> 32, 16, 16);
    return HwregAccessResult::Success;
  case HwregState::ShaderCycles:
    // Full-width HWREG 29 reads on W7900 wrap at bit 20.
    raw_value = wf.cu().shader_cycles() & 0xfffffu;
    return HwregAccessResult::Success;
  case HwregState::Status:
    raw_value = wf.status_raw();
    if ((wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 ||
         wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5)) {
      // SCC, user priority and HALT are in STATE_PRIV on GFX12. Derive the
      // live STATUS fields instead of exposing their legacy backing bits.
      constexpr uint32_t external_bits =
          (1u << 8) | (1u << 14) | (1u << 18) | (1u << 23) | (0xfu << 24) | (1u << 30);
      raw_value =
          (raw_value & external_bits) | field_value(wf.in_trap_handler(), 5, 1) |
          field_value(wf.cu().trap_message_address(0x85) != 0, 6, 1) |
          field_value(wf.exec() == 0, 9, 1) | field_value(wf.vcc_mask() == 0, 10, 1) |
          field_value(wf.cu().in_multiwave_workgroup(), 11, 1) |
          field_value(wf.trap_barrier_complete(), 15, 1) | field_value(!wf.is_halted(), 16, 1) |
          field_value(wf.wait_counters().empty(), 28, 1) | field_value(wf.wf_size() == 64, 29, 1);
    } else {
      constexpr uint32_t dynamic_bits = (1u << 6) | (7u << 9) | (1u << 12) | (1u << 16);
      raw_value = (raw_value & ~dynamic_bits) | ((wf.cu().trap_addresses().first != 0) << 6) |
                  ((!wf.exec()) << 9) | ((!wf.vcc_mask()) << 10) |
                  (wf.cu().in_multiwave_workgroup() << 11) |
                  ((wf.state() == WfState::BARRIER) << 12) | ((!wf.is_halted()) << 16);
    }
    return HwregAccessResult::Success;
  case HwregState::StatusGfx10: {
    constexpr uint32_t live_fields = (1u << 6) | (1u << 8) | (15u << 9) | (1u << 16);
    raw_value =
        (wf.status_raw() & ~live_fields) | (uint32_t{wf.cu().trap_addresses().first != 0} << 6) |
        (uint32_t{wf.exec() == 0} << 9) | (uint32_t{wf.vcc_mask() == 0} << 10) |
        (uint32_t{wf.cu().in_multiwave_workgroup()} << 11) |
        (uint32_t{wf.state() == WfState::BARRIER} << 12) | (uint32_t{!wf.is_halted()} << 16);
    return HwregAccessResult::Success;
  }
  case HwregState::LdsAllocGfx10:
    raw_value = field_value(wf.lds_base() / 256u, 0, 9) | field_value(wf.lds_size() / 256u, 12, 9);
    return HwregAccessResult::Success;
  case HwregState::IbStsGfx10: {
    const auto &counters = wf.wait_counters();
    raw_value = field_value(counters.vmcnt, 0, 4) | field_value(counters.expcnt, 4, 3) |
                field_value(counters.lgkmcnt >> 4, 7, 1) | field_value(counters.lgkmcnt, 8, 4) |
                field_value(counters.vmcnt >> 4, 22, 2) |
                field_value(counters.lgkmcnt >> 5, 24, 1) | field_value(counters.vscnt, 26, 6);
    return HwregAccessResult::Success;
  }
  case HwregState::ShMemBasesGfx10: {
    const unsigned shift = wf.flat_pointer32() ? 16 : 48;
    raw_value = field_value(wf.private_aperture_base() >> shift, 0, 16) |
                field_value(wf.shared_aperture_base() >> shift, 16, 16);
    return HwregAccessResult::Success;
  }
  case HwregState::ShaderTbaLoGfx10:
  case HwregState::ShaderTbaHiGfx10:
  case HwregState::ShaderTmaLoGfx10:
  case HwregState::ShaderTmaHiGfx10: {
    const bool tma = state == HwregState::ShaderTmaLoGfx10 || state == HwregState::ShaderTmaHiGfx10;
    const bool high =
        state == HwregState::ShaderTbaHiGfx10 || state == HwregState::ShaderTmaHiGfx10;
    const auto addresses = wf.cu().trap_addresses();
    const uint64_t address = (tma ? addresses.second : addresses.first) >> 8;
    raw_value =
        high ? static_cast<uint32_t>(address >> 32) & 0xffffu : static_cast<uint32_t>(address);
    return HwregAccessResult::Success;
  }
  case HwregState::ShaderCyclesGfx1030:
    raw_value = wf.shader_cycles() & 0xfffffu;
    return HwregAccessResult::Success;
  case HwregState::StatePrivGfx12:
    raw_value = gfx12_state_priv_from_status(wf, wf.status_raw(), wf.scratch_base() != 0);
    return HwregAccessResult::Success;
  case HwregState::ExcpFlagPrivGfx12:
    raw_value = gfx12_excp_flag_priv_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::ExcpFlagUserGfx12:
    raw_value = gfx12_excp_flag_user_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::TrapCtrlGfx12:
    raw_value = gfx12_trap_ctrl_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::XnackStatePrivGfx1250:
    raw_value = wf.gfx1250_xnack_state_priv_raw();
    return HwregAccessResult::Success;
  case HwregState::XnackMaskGfx1250:
    raw_value = wf.gfx1250_xnack_mask_raw();
    return HwregAccessResult::Success;
  case HwregState::FlatScratchLo:
    raw_value = static_cast<uint32_t>(wf.scratch_base());
    return HwregAccessResult::Success;
  case HwregState::FlatScratchHi:
    raw_value = static_cast<uint32_t>(wf.scratch_base() >> 32);
    return HwregAccessResult::Success;
  case HwregState::Trapsts:
    raw_value = wf.trapsts();
    return HwregAccessResult::Success;
  case HwregState::SimdIdCdna2:
    raw_value = cdna2_4_simd_id_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::LocationCdna3_4:
    raw_value = cdna2_4_simd_id_raw(wf) | field_value(wf.wf_id() / 4, 0, 4) |
                field_value(wf.cu().shader_array_cu_id(), 8, 4) |
                field_value(wf.cu().shader_engine_id(), 13, 3);
    return HwregAccessResult::Success;
  case HwregState::XccIdCdna3_4: {
    const std::optional<uint32_t> xcc = wf.cu().xcc_id();
    if (!xcc || *xcc > 15)
      return HwregAccessResult::Unsupported;
    raw_value = *xcc;
    return HwregAccessResult::Success;
  }
  case HwregState::GprAllocGfx9_10:
    raw_value = gfx9_10_gpr_alloc_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::LdsAllocCdna:
    raw_value = ((wf.lds_base() >> 8) & 0x1ffu) | (((wf.lds_size() >> 8) & 0x3ffu) << 12);
    return HwregAccessResult::Success;
  case HwregState::IbStsCdna: {
    const auto &counters = wf.wait_counters();
    raw_value = (counters.vmcnt & 0xfu) | ((counters.expcnt & 7u) << 4) |
                ((counters.lgkmcnt & 0xfu) << 8) | ((counters.vmcnt & 0x30u) << 18);
    return HwregAccessResult::Success;
  }
  case HwregState::GprAllocCdna3_4:
    raw_value = cdna3_4_gpr_alloc_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::WaveSchedMode:
    raw_value = wf.wave_sched_mode_raw();
    return HwregAccessResult::Success;
  case HwregState::PopsPackerGfx10:
    raw_value = 0; // Compute waves have no ordered pixel packer.
    return HwregAccessResult::Success;
  case HwregState::TbaLo:
  case HwregState::TbaHi:
  case HwregState::TmaLo:
  case HwregState::TmaHi: {
    const auto [tba, tma] = wf.cu().trap_addresses();
    const bool high = state == HwregState::TbaHi || state == HwregState::TmaHi;
    const uint64_t address =
        (state == HwregState::TbaHi || state == HwregState::TbaLo ? tba : tma) >> 8;
    raw_value = high ? field_value(address >> 32, 0, 16) : static_cast<uint32_t>(address);
    return HwregAccessResult::Success;
  }
  case HwregState::IbStsGfx1250:
    raw_value = gfx1250_ib_sts_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::IbSts2Gfx1250:
    raw_value = gfx1250_ib_sts2_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::WgpIdGfx1250:
    if (!wf.cu().cus_per_shader_array() || wf.cu().cus_per_shader_array() > 16)
      return HwregAccessResult::Unsupported;
    // Each gfx1250 ComputeUnit models one WGP, including its shared LDS.
    raw_value = field_value(wf.cu().shader_array_cu_id(), 10, 4);
    return HwregAccessResult::Success;
  case HwregState::Unsupported:
    return HwregAccessResult::Unsupported;
  }
  return HwregAccessResult::Unsupported;
}

HwregAccessResult write_raw_hwreg(Wavefront &wf, HwregState state, uint32_t raw_value) {
  switch (state) {
  case HwregState::Mode:
    wf.set_mode_raw(raw_value);
    return HwregAccessResult::Success;
  case HwregState::WaveSchedMode:
    wf.set_wave_sched_mode_raw(raw_value);
    return HwregAccessResult::Success;
  case HwregState::FlatScratchLo:
    wf.set_scratch_base((wf.scratch_base() & 0xffffffff00000000ULL) | raw_value);
    return HwregAccessResult::Success;
  case HwregState::FlatScratchHi:
    wf.set_scratch_base((wf.scratch_base() & 0xffffffffULL) | (uint64_t{raw_value} << 32));
    return HwregAccessResult::Success;
  case HwregState::Trapsts:
    // The trap handler clears the excp bit it just serviced before s_rfe, so
    // this has to write through to wave state rather than report Unsupported.
    wf.set_trapsts(raw_value);
    return HwregAccessResult::Success;
  case HwregState::FlushIb:
    if (raw_value & 1u)
      wf.cu().instruction_cache().invalidate_all();
    return HwregAccessResult::Success;
  case HwregState::GprAllocGfx11:
  case HwregState::LdsAllocGfx11:
  case HwregState::IbSts2Gfx11:
  case HwregState::IbDbgGfx11:
  case HwregState::HwId1Gfx11:
  case HwregState::IbStsGfx11:
  case HwregState::PcLo:
  case HwregState::PcHi:
  case HwregState::ApertureBases:
  case HwregState::ShaderCycles:
    return HwregAccessResult::ReadOnly;
  case HwregState::Gfx11Status: {
    constexpr uint32_t writable = 0x1fu | (1u << 8) | (1u << 13) | (1u << 18) | (3u << 20) |
                                  (1u << 25) | (1u << 27) | (1u << 29);
    raw_value = (raw_value & writable) | (wf.status_raw() & ~writable);
    [[fallthrough]];
  }
  case HwregState::Status:
    // Only reachable from a privileged (trap-handler) write; the policy check
    // in write_hwreg_field() refuses user-mode writes before we get here. The
    // gfx9 trap handler sets STATUS.HALT this way before returning.
    //
    // That makes this the other origin of STATUS.HALT, so it owns the bit now:
    // drop any s_sendmsghalt provenance, whichever way the handler wrote it.
    wf.set_status_raw(raw_value);
    wf.set_self_halted(false);
    return HwregAccessResult::Success;
  case HwregState::StatePrivGfx12:
    set_gfx12_state_priv_raw(wf, raw_value);
    return HwregAccessResult::Success;
  case HwregState::ExcpFlagPrivGfx12:
    set_gfx12_excp_flag_priv_raw(wf, raw_value);
    return HwregAccessResult::Success;
  case HwregState::ExcpFlagUserGfx12:
    set_gfx12_excp_flag_user_raw(wf, raw_value);
    return HwregAccessResult::Success;
  case HwregState::TrapCtrlGfx12:
    set_gfx12_trap_ctrl_raw(wf, raw_value);
    return HwregAccessResult::Success;
  case HwregState::XnackStatePrivGfx1250:
    wf.set_gfx1250_xnack_state_priv_raw(raw_value);
    return HwregAccessResult::Success;
  case HwregState::XnackMaskGfx1250:
    wf.set_gfx1250_xnack_mask_raw(raw_value);
    return HwregAccessResult::Success;
  case HwregState::ShMemBasesGfx10: {
    const unsigned shift = wf.flat_pointer32() ? 16 : 48;
    const uint64_t low_mask = (uint64_t{1} << shift) - 1;
    const uint64_t shared = uint64_t{raw_value >> 16} << shift;
    const uint64_t priv = uint64_t{raw_value & 0xffffu} << shift;
    wf.set_apertures(shared | (wf.shared_aperture_base() & low_mask),
                     shared | (wf.shared_aperture_limit() & low_mask),
                     priv | (wf.private_aperture_base() & low_mask),
                     priv | (wf.private_aperture_limit() & low_mask), /*override_launch=*/true);
    return HwregAccessResult::Success;
  }
  case HwregState::StatusGfx10: {
    // ROCr restores STATUS (including SCC and HALT) before RFE. PRIV remains
    // asserted until RFE; bits derived from live hardware state cannot be set.
    constexpr uint32_t writable = 1u | (3u << 3) | (1u << 13);
    wf.set_status_raw((wf.status_raw() & ~writable) | (raw_value & writable));
    wf.set_self_halted(false);
    return HwregAccessResult::Success;
  }
  case HwregState::LdsAllocGfx10:
  case HwregState::IbStsGfx10:
  case HwregState::ShaderTbaLoGfx10:
  case HwregState::ShaderTbaHiGfx10:
  case HwregState::ShaderTmaLoGfx10:
  case HwregState::ShaderTmaHiGfx10:
  case HwregState::ShaderCyclesGfx1030:
  case HwregState::SimdIdCdna2:
  case HwregState::LocationCdna3_4:
  case HwregState::XccIdCdna3_4:
  case HwregState::GprAllocGfx9_10:
  case HwregState::LdsAllocCdna:
  case HwregState::IbStsCdna:
  case HwregState::GprAllocCdna3_4:
  case HwregState::PopsPackerGfx10:
  case HwregState::TbaLo:
  case HwregState::TbaHi:
  case HwregState::TmaLo:
  case HwregState::TmaHi:
  case HwregState::IbStsGfx1250:
  case HwregState::IbSts2Gfx1250:
  case HwregState::WgpIdGfx1250:
  case HwregState::Unsupported:
    return HwregAccessResult::Unsupported;
  }
  return HwregAccessResult::Unsupported;
}

uint32_t insert_hwreg_field(uint32_t raw_value, uint32_t src, const DecodedHwreg &decoded) {
  uint32_t shifted_mask = decoded.mask << decoded.offset;
  return (raw_value & ~shifted_mask) | ((src & decoded.mask) << decoded.offset);
}

} // namespace

uint32_t gfx12_state_priv_from_status(const Wavefront &wf, uint32_t status, bool scratch_enabled) {
  uint32_t value = ((status & 1u) << 9) | (((status >> 1) & 0xFu) << 10);
  value |= ((status >> 13) & 1u) << 14;
  value |= (scratch_enabled ? 1u : 0u) << 18;
  value |= ((status >> 19) & 1u) << 19;
  value |= ((status >> 7) & 1u) << 20;
  value |= (uint32_t{wf.sleep_wakeup()} << 1) | (uint32_t{wf.workgroup_barrier_complete()} << 2);
  return value;
}

void set_gfx12_state_priv_workgroup_flags(Wavefront &wf, uint32_t value) {
  wf.set_sleep_wakeup((value & (1u << 1)) != 0);
  wf.set_workgroup_barrier_complete((value & (1u << 2)) != 0);
}

uint32_t update_status_from_gfx12_state_priv(uint32_t status, uint32_t state_priv) {
  constexpr uint32_t kMappedStatus =
      (1u << 0) | (0xFu << 1) | (1u << 7) | Wavefront::kStatusHaltMask | (1u << 19);
  status &= ~kMappedStatus;
  status |= (state_priv >> 9) & 1u;
  status |= ((state_priv >> 10) & 0xFu) << 1;
  status |= ((state_priv >> 14) & 1u) << 13;
  status |= ((state_priv >> 19) & 1u) << 19;
  status |= ((state_priv >> 20) & 1u) << 7;
  return status;
}

uint32_t gfx12_excp_flag_priv_from_trapsts(uint32_t trapsts) {
  uint32_t value = ((trapsts >> 7) & 1u) << 0;
  value |= ((trapsts >> 12) & 0x7u) << 1;
  value |= ((trapsts >> 8) & 1u) << 4;
  value |= ((trapsts >> 10) & 1u) << 5;
  value |= ((trapsts >> 11) & 1u) << 6;
  value |= ((trapsts >> 22) & 1u) << 7;
  value |= ((trapsts >> 23) & 1u) << 8;
  value |= ((trapsts >> 24) & 1u) << 9;
  value |= ((trapsts >> 26) & 1u) << 10;
  value |= ((trapsts >> 25) & 1u) << 11;
  value |= ((trapsts >> 28) & 1u) << 12;
  value |= trapsts & (0x3u << 30);
  return value;
}

uint32_t update_trapsts_from_gfx12_excp_flag_priv(uint32_t trapsts, uint32_t excp_flag_priv) {
  constexpr uint32_t kMappedTrapsts = (1u << 7) | (1u << 8) | (1u << 10) | (1u << 11) |
                                      (0x7u << 12) | (0x1Fu << 22) | (1u << 28) | (0x3u << 30);
  trapsts &= ~kMappedTrapsts;
  trapsts |= ((excp_flag_priv >> 0) & 1u) << 7;
  trapsts |= ((excp_flag_priv >> 1) & 0x7u) << 12;
  trapsts |= ((excp_flag_priv >> 4) & 1u) << 8;
  trapsts |= ((excp_flag_priv >> 5) & 1u) << 10;
  trapsts |= ((excp_flag_priv >> 6) & 1u) << 11;
  trapsts |= ((excp_flag_priv >> 7) & 1u) << 22;
  trapsts |= ((excp_flag_priv >> 8) & 1u) << 23;
  trapsts |= ((excp_flag_priv >> 9) & 1u) << 24;
  trapsts |= ((excp_flag_priv >> 10) & 1u) << 26;
  trapsts |= ((excp_flag_priv >> 11) & 1u) << 25;
  trapsts |= ((excp_flag_priv >> 12) & 1u) << 28;
  trapsts |= excp_flag_priv & (0x3u << 30);
  return trapsts;
}

uint32_t update_trapsts_from_gfx12_excp_flag_user(uint32_t trapsts, uint32_t excp_flag_user) {
  return (trapsts & ~0x7Fu) | (excp_flag_user & 0x7Fu);
}

uint32_t hwreg_id(uint16_t hwreg) { return hwreg & 0x3Fu; }

const char *hwreg_name(const Wavefront &wf, uint16_t hwreg) {
  const HwregDescriptor *desc = find_descriptor(wf.cu().arch(), hwreg_id(hwreg));
  return desc ? desc->name : "unknown";
}

const char *hwreg_access_result_name(HwregAccessResult result) {
  switch (result) {
  case HwregAccessResult::Success:
    return "success";
  case HwregAccessResult::Unsupported:
    return "unsupported";
  case HwregAccessResult::ReadOnly:
    return "read-only";
  case HwregAccessResult::Privileged:
    return "privileged";
  }
  return "unknown";
}

HwregAccessResult read_hwreg_field(Wavefront &wf, uint16_t hwreg, uint32_t &value) {
  DecodedHwreg decoded = decode_hwreg(hwreg);
  const HwregDescriptor *desc = find_descriptor(wf.cu().arch(), decoded.id);
  if (!desc) {
    value = 0;
    return HwregAccessResult::Unsupported;
  }

  // CDNA4 exposes the configured 48-bit TBA and TMA as low/high dword pairs.
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA4 && decoded.id >= 16 && decoded.id <= 19) {
    const auto [tba, tma] = wf.cu().trap_addresses();
    const uint64_t address = ((decoded.id < 18 ? tba : tma) & 0x0000ffffffffffffULL) >> 8;
    const uint32_t word = static_cast<uint32_t>(address >> ((decoded.id & 1u) * 32));
    value = (word >> decoded.offset) & decoded.mask;
    return HwregAccessResult::Success;
  }

  // Only SIMD_ID[5:4] has backing state; do not fabricate the other HW_ID fields.
  if (desc->state == HwregState::SimdIdCdna2 &&
      (decoded.offset < 4 || decoded.offset + decoded.size > 6)) {
    value = 0;
    return HwregAccessResult::Unsupported;
  }

  if (desc->state == HwregState::LocationCdna3_4) {
    // Only expose fields backed by physical wave slots and CU topology.
    // PIPE/TG/VM/QUEUE/ME/STATE identities need their own allocation model.
    // WAVE_ID[3:0], SIMD_ID[5:4], CU_ID[11:8], SH_ID[12], SE_ID[15:13].
    // CDNA3/4 specify SH_ID as zero, independently of the configured geometry.
    constexpr uint32_t kSupportedHwIdBits = 0x0000ff3f;
    if (((decoded.mask << decoded.offset) & ~kSupportedHwIdBits) ||
        (field_intersects(decoded, 0, 4) && wf.wf_id() >= 64) ||
        (field_intersects(decoded, 8, 4) &&
         (!wf.cu().cus_per_shader_array() || wf.cu().cus_per_shader_array() > 16 ||
          wf.cu().shader_engine_cu_count() != wf.cu().cus_per_shader_array())) ||
        (field_intersects(decoded, 13, 3) && wf.cu().shader_engine_id() > 7)) {
      value = 0;
      return HwregAccessResult::Unsupported;
    }
  }

  if (desc->state == HwregState::GprAllocCdna3_4 && field_intersects(decoded, 12, 6)) {
    value = 0;
    return HwregAccessResult::Unsupported;
  }

  // DP_RATE is a physical implementation parameter, independent of CU placement.
  if (desc->state == HwregState::HwId1Gfx11 &&
      (field_intersects(decoded, 29, 3) ||
       (field_intersects(decoded, 8, 13) && !wf.cu().cus_per_shader_array()))) {
    value = 0;
    return HwregAccessResult::Unsupported;
  }

  // Only WGP_ID[13:10] has backing state; do not fabricate the other HW_ID1 fields.
  if (desc->state == HwregState::WgpIdGfx1250 &&
      (decoded.offset < 10 || decoded.offset + decoded.size > 14)) {
    value = 0;
    return HwregAccessResult::Unsupported;
  }

  uint32_t raw_value = 0;
  HwregAccessResult result = read_raw_hwreg(wf, desc->state, raw_value);
  if (result != HwregAccessResult::Success) {
    value = 0;
    return result;
  }

  if (desc->state == HwregState::Gfx11Status && field_intersects(decoded, 6, 1) &&
      wf.cu().trap_addresses().first != 0)
    raw_value |= 1u << 6;
  value = (raw_value >> decoded.offset) & decoded.mask;
  return HwregAccessResult::Success;
}

bool hwreg_accesses_scc(const Wavefront &wf, uint16_t hwreg, bool write) {
  const auto decoded = decode_hwreg(hwreg);
  const auto *desc = find_descriptor(wf.cu().arch(), decoded.id);
  if (!desc || !field_aliases_scc(wf, desc->state, decoded))
    return false;
  return !write || (desc->write_policy != HwregWritePolicy::ReadOnly &&
                    (desc->write_policy != HwregWritePolicy::Privileged || wf.in_trap_handler()));
}

HwregAccessResult write_hwreg_field(Wavefront &wf, uint16_t hwreg, uint32_t src,
                                    HwregWriteKind kind) {
  DecodedHwreg decoded = decode_hwreg(hwreg);
  const HwregDescriptor *desc = find_descriptor(wf.cu().arch(), decoded.id);
  if (!desc)
    return HwregAccessResult::Unsupported;

  switch (desc->write_policy) {
  case HwregWritePolicy::ReadOnly:
    return HwregAccessResult::ReadOnly;
  case HwregWritePolicy::Privileged:
    // A wave running the trap handler is in privileged mode (hardware sets
    // STATUS.PRIV on trap entry), which is exactly the context these registers
    // are writable from. Everything else is user mode and is refused.
    if (!wf.in_trap_handler())
      return HwregAccessResult::Privileged;
    break;
  case HwregWritePolicy::UserWritable:
    break;
  }

  // Update shared trap addresses under the frontend lock. Reading here and
  // writing back a full word could discard a concurrent shader or ioctl write.
  switch (desc->state) {
  case HwregState::ShaderTbaLoGfx10:
  case HwregState::ShaderTbaHiGfx10:
  case HwregState::ShaderTmaLoGfx10:
  case HwregState::ShaderTmaHiGfx10: {
    const bool tma =
        desc->state == HwregState::ShaderTmaLoGfx10 || desc->state == HwregState::ShaderTmaHiGfx10;
    const bool high =
        desc->state == HwregState::ShaderTbaHiGfx10 || desc->state == HwregState::ShaderTmaHiGfx10;
    const uint32_t shift = 8 + decoded.offset + (high ? 32 : 0);
    if (shift >= 48)
      return HwregAccessResult::Success;
    const uint64_t mask = (uint64_t{decoded.mask} << shift) & 0x0000ffffffffffffull;
    return wf.cu().update_trap_address(tma, mask, uint64_t{src} << shift)
               ? HwregAccessResult::Success
               : HwregAccessResult::Unsupported;
  }
  default:
    break;
  }

  if (desc->state == HwregState::FlushIb)
    return write_raw_hwreg(wf, desc->state, insert_hwreg_field(0, src, decoded));
  uint32_t raw_value = 0;
  HwregAccessResult result = read_raw_hwreg(wf, desc->state, raw_value);
  if (result != HwregAccessResult::Success)
    return result;

  uint32_t updated = insert_hwreg_field(raw_value, src, decoded);
  if (desc->state == HwregState::Mode && kind != HwregWriteKind::Generic &&
      wf.cu().setreg_vgpr_msb_fixup()) {
    // Public LLVM's SetregVGPRMSBFixup contract records that gfx1250 MODE
    // writes take VGPR-MSB from the unshifted source bits[12:19], regardless of
    // the requested HWREG slice. gfx1251 lacks this target capability and uses
    // the ordinary field insertion above.
    updated = (updated & ~VGPR_MSB_MODE_MASK) | (src & VGPR_MSB_MODE_MASK);
    if (kind == HwregWriteKind::SetregImm32)
      wf.arm_setreg_vgpr_msb_hazard();
  }

  return write_raw_hwreg(wf, desc->state, updated);
}

} // namespace amdgpu
} // namespace rocjitsu
