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
  GprAllocGfx9_10,
  GprAllocCdna3_4,
  WaveSchedMode,
  IbStsGfx1250,
  IbSts2Gfx1250,
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

// UserWritable records ISA privilege, not simulator completeness. If a
// user-writable register has no HwregState backing yet, writes return
// Unsupported rather than fabricating hidden state or side effects.
constexpr HwregDescriptor CDNA1_2_HWREGS[] = {
    // CDNA-family ISA XML uses MODE=1 and STATUS=2. Keep XML-only and
    // unmodeled state explicit but unsupported until the corresponding VM
    // state, field layouts, and privilege side effects are represented.
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {3, "TRAPSTS", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {4, "HW_ID", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {5, "GPR_ALLOC", HwregState::GprAllocGfx9_10, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "INST_DW0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "INST_DW1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "IB_DBG0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {16, "SQ_SHADER_TBA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {17, "SQ_SHADER_TBA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {18, "SQ_SHADER_TMA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {19, "SQ_SHADER_TMA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
};

constexpr HwregDescriptor CDNA3_4_HWREGS[] = {
    // CDNA3/CDNA4 ISA manuals define the GPR_ALLOC field layout; MODE/STATUS
    // and GPR_ALLOC have backing wave state, while the other XML HWREG IDs stay
    // named but unsupported until their VM state and privilege side effects are
    // represented.
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {3, "TRAPSTS", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {4, "HW_ID", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {5, "GPR_ALLOC", HwregState::GprAllocCdna3_4, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "INST_DW0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "INST_DW1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "IB_DBG0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {16, "SQ_SHADER_TBA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {17, "SQ_SHADER_TBA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {18, "SQ_SHADER_TMA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {19, "SQ_SHADER_TMA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {20, "XCC_ID", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {21, "SQ_PERF_SNAPSHOT_DATA", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {22, "SQ_PERF_SNAPSHOT_DATA1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {23, "SQ_PERF_SNAPSHOT_PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {24, "SQ_PERF_SNAPSHOT_PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor RDNA1_HWREGS[] = {
    // RDNA1 XML and ISA prose use MODE=1, STATUS=2. Remaining XML HWREG IDs
    // are named for diagnostics but stay unsupported without modeled VM state.
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {3, "TRAPSTS", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {4, "HW_ID_LEGACY", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {5, "GPR_ALLOC", HwregState::GprAllocGfx9_10, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "INST_DW0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "INST_DW1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "IB_DBG0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {16, "SHADER_TBA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {17, "SHADER_TBA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {18, "SHADER_TMA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {19, "SHADER_TMA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {20, "SHADER_FLAT_SCRATCH_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {21, "SHADER_FLAT_SCRATCH_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
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
    {2, "STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {3, "TRAPSTS", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {5, "GPR_ALLOC", HwregState::GprAllocGfx9_10, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "INST_DW0", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {16, "SHADER_TBA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {17, "SHADER_TBA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {18, "SHADER_TMA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {19, "SHADER_TMA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {20, "SHADER_FLAT_SCRATCH_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {21, "SHADER_FLAT_SCRATCH_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {23, "HW_ID1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {24, "HW_ID2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {25, "POPS_PACKER", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {26, "SCHED_MODE", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {27, "VGPR_OFFSET", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {28, "IB_STS2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {29, "SHADER_CYCLES", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor RDNA3_HWREGS[] = {
    // RDNA3/RDNA3.5 XML exposes the full GFX11 HWREG ID inventory. MODE/STATUS
    // have backing wave state; remaining registers are named for diagnostics
    // but left unsupported until VM state and privilege side effects are added.
    {1, "MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {3, "TRAPSTS", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {5, "GPR_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {6, "LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {8, "PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {9, "PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {13, "IB_DBG1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "SH_MEM_BASES", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {16, "SHADER_TBA_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {17, "SHADER_TBA_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {18, "PERF_SNAPSHOT_PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {19, "PERF_SNAPSHOT_PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {20, "SHADER_FLAT_SCRATCH_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {21, "SHADER_FLAT_SCRATCH_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {23, "HW_ID1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {24, "HW_ID2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {25, "POPS_PACKER", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {26, "SCHED_MODE", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {27, "PERF_SNAPSHOT_DATA", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {28, "IB_STS2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {29, "SHADER_CYCLES", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
};

constexpr HwregDescriptor RDNA4_HWREGS[] = {
    // RDNA4 ISA manual section 3.4 and OPR_HWREG XML. ID 28 (IB_STS2)
    // appears in prose only, so keep it as a named unsupported read-only entry.
    {1, "WAVE_MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "WAVE_STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {4, "WAVE_STATE_PRIV", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {5, "WAVE_GPR_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {6, "WAVE_LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {10, "PERF_SNAPSHOT_DATA", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "PERF_SNAPSHOT_PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "PERF_SNAPSHOT_PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {15, "PERF_SNAPSHOT_DATA1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {16, "PERF_SNAPSHOT_DATA2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {17, "WAVE_EXCP_FLAG_PRIV", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {18, "WAVE_EXCP_FLAG_USER", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {19, "WAVE_TRAP_CTRL", HwregState::Unsupported, HwregWritePolicy::Privileged},
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
    // departure recorded in gfx1250::Isa.
    {1, "WAVE_MODE", HwregState::Mode, HwregWritePolicy::UserWritable},
    {2, "WAVE_STATUS", HwregState::Status, HwregWritePolicy::ReadOnly},
    {4, "WAVE_STATE_PRIV", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {5, "WAVE_GPR_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {6, "WAVE_LDS_ALLOC", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {7, "IB_STS", HwregState::IbStsGfx1250, HwregWritePolicy::ReadOnly},
    {10, "PERF_SNAPSHOT_DATA", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {11, "PERF_SNAPSHOT_PC_LO", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {12, "PERF_SNAPSHOT_PC_HI", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {14, "FLUSH_IB", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {15, "PERF_SNAPSHOT_DATA1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {16, "PERF_SNAPSHOT_DATA2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {17, "WAVE_EXCP_FLAG_PRIV", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {18, "WAVE_EXCP_FLAG_USER", HwregState::Unsupported, HwregWritePolicy::UserWritable},
    {19, "WAVE_TRAP_CTRL", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {20, "WAVE_SCRATCH_BASE_LO", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {21, "WAVE_SCRATCH_BASE_HI", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {23, "WAVE_HW_ID1", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {24, "WAVE_HW_ID2", HwregState::Unsupported, HwregWritePolicy::ReadOnly},
    {26, "WAVE_SCHED_MODE", HwregState::WaveSchedMode, HwregWritePolicy::UserWritable},
    {28, "IB_STS2", HwregState::IbSts2Gfx1250, HwregWritePolicy::ReadOnly},
    {33, "WAVE_XNACK_STATE_PRIV", HwregState::Unsupported, HwregWritePolicy::Privileged},
    {34, "WAVE_XNACK_MASK", HwregState::Unsupported, HwregWritePolicy::Privileged},
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
  case ROCJITSU_CODE_ARCH_CDNA2:
    return make_table(CDNA1_2_HWREGS);
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
  case ROCJITSU_CODE_ARCH_GFX1250:
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

HwregAccessResult read_raw_hwreg(Wavefront &wf, HwregState state, uint32_t &raw_value) {
  switch (state) {
  case HwregState::Mode:
    raw_value = wf.mode_raw();
    return HwregAccessResult::Success;
  case HwregState::Status:
    raw_value = wf.status_raw();
    return HwregAccessResult::Success;
  case HwregState::GprAllocGfx9_10:
    raw_value = gfx9_10_gpr_alloc_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::GprAllocCdna3_4:
    raw_value = cdna3_4_gpr_alloc_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::WaveSchedMode:
    raw_value = wf.wave_sched_mode_raw();
    return HwregAccessResult::Success;
  case HwregState::IbStsGfx1250:
    raw_value = gfx1250_ib_sts_raw(wf);
    return HwregAccessResult::Success;
  case HwregState::IbSts2Gfx1250:
    raw_value = gfx1250_ib_sts2_raw(wf);
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
  case HwregState::Status:
  case HwregState::GprAllocGfx9_10:
  case HwregState::GprAllocCdna3_4:
  case HwregState::IbStsGfx1250:
  case HwregState::IbSts2Gfx1250:
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

  if (desc->state == HwregState::GprAllocCdna3_4 && field_intersects(decoded, 12, 6)) {
    value = 0;
    return HwregAccessResult::Unsupported;
  }

  uint32_t raw_value = 0;
  HwregAccessResult result = read_raw_hwreg(wf, desc->state, raw_value);
  if (result != HwregAccessResult::Success) {
    value = 0;
    return result;
  }

  value = (raw_value >> decoded.offset) & decoded.mask;
  return HwregAccessResult::Success;
}

HwregAccessResult write_hwreg_field(Wavefront &wf, uint16_t hwreg, uint32_t src) {
  DecodedHwreg decoded = decode_hwreg(hwreg);
  const HwregDescriptor *desc = find_descriptor(wf.cu().arch(), decoded.id);
  if (!desc)
    return HwregAccessResult::Unsupported;

  switch (desc->write_policy) {
  case HwregWritePolicy::ReadOnly:
    return HwregAccessResult::ReadOnly;
  case HwregWritePolicy::Privileged:
    return HwregAccessResult::Privileged;
  case HwregWritePolicy::UserWritable:
    break;
  }

  uint32_t raw_value = 0;
  HwregAccessResult result = read_raw_hwreg(wf, desc->state, raw_value);
  if (result != HwregAccessResult::Success)
    return result;

  return write_raw_hwreg(wf, desc->state, insert_hwreg_field(raw_value, src, decoded));
}

} // namespace amdgpu
} // namespace rocjitsu
