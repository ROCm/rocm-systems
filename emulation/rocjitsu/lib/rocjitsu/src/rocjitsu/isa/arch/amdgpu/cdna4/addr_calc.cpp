// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/cdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_flat.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include <cassert>
#include <cstdint>
#include <format>

namespace rocjitsu {
namespace cdna4 {

namespace {
bool is_scratch_op(uint32_t op) { return (op >= 5 && op <= 7) || (op >= 21 && op <= 23); }
} // namespace

uint64_t smem_calculate_address(const SmemMachineInst &inst, amdgpu::Wavefront &wf) {
  if (!is_scratch_op(inst.op))
    return amdgpu::addr_calc::smem_calculate_address(inst, wf);

  // S_SCRATCH_LOAD / S_SCRATCH_STORE per CDNA4 ISA spec (section 8.2.1.1):
  // ADDR = SGPR[base] + inst_offset + { M0 or SGPR[soffset] or zero } * 64
  auto &cu = wf.cu();
  uint32_t sbase = wf.sgpr_alloc().base + inst.sbase * 2;
  uint64_t base = (static_cast<uint64_t>(cu.read_sgpr(sbase + 1)) << 32) | cu.read_sgpr(sbase);
  int64_t inst_offset = 0;
  if (inst.imm)
    inst_offset = static_cast<int64_t>(static_cast<int32_t>(inst.offset << 11) >> 11);
  uint64_t sgpr_off = 0;
  if (inst.soffset_en)
    sgpr_off = cu.read_sgpr(wf.sgpr_alloc().base + inst.soffset);
  uint64_t addr = base + inst_offset + sgpr_off * 64;
  util::Logger::cp([&](auto &os) {
    static thread_local uint64_t scratch_smem_count = 0;
    if (++scratch_smem_count <= 32)
      os << std::format("S_SCRATCH_SMEM wf{} op={} sbase_idx={} base={:#x} inst_off={} "
                        "sgpr_off={} addr={:#x} pc={:#x}",
                        wf.wf_id(), inst.op, inst.sbase, base, inst_offset, sgpr_off, addr, wf.pc);
  });
  return addr;
}

void flat_calculate_addresses(const FlatMachineInst &inst, amdgpu::Wavefront &wf,
                              amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::flat_calculate_addresses(inst, wf, d);
}

void mubuf_calculate_addresses(const MubufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mubuf_calculate_addresses(inst, wf, d);
}

void mtbuf_calculate_addresses(const MtbufMachineInst &inst, amdgpu::Wavefront &wf,
                               amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::mtbuf_calculate_addresses(inst, wf, d);
}

void ds_calculate_addresses(const DsMachineInst &inst, amdgpu::Wavefront &wf,
                            amdgpu::VectorMemState &d) {
  amdgpu::addr_calc::ds_calculate_addresses(inst, wf, d);
}

} // namespace cdna4
} // namespace rocjitsu
