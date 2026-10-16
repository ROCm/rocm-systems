// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>

namespace {

using namespace rocjitsu;

TEST(RdnaAddrCalcTest, BufferDescriptorCanSpanSgprsAndVccWithoutReadingAdjacentWave) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    amdgpu::GpuMemory mem("descriptor_vcc_mem");
    amdgpu::L2Cache l2("descriptor_vcc_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 2;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("descriptor_vcc", cfg, &mem, &l2);
    auto *wf = cu->dispatch_wf(0, 0, 106, 16);
    auto *next = cu->dispatch_wf(1, 0, 106, 16);
    ASSERT_NE(wf, nullptr);
    ASSERT_NE(next, nullptr);
    wf->set_exec(1);
    const uint32_t sb = wf->sgpr_alloc().base;
    cu->write_sgpr(sb + 104, 0x1000);
    cu->write_sgpr(sb + 105, arch == ROCJITSU_CODE_ARCH_CDNA5 ? 16u << 25 : 0);
    // Adjacent storage has zero NUM_RECORDS; the real bound lives in VCC_LO.
    cu->write_sgpr(next->sgpr_alloc().base, 0);
    cu->write_sgpr(next->sgpr_alloc().base + 1, 0);
    wf->set_vcc_raw(arch == ROCJITSU_CODE_ARCH_CDNA5 ? 0 : (uint64_t{3u << 28} << 32) | 16);
    amdgpu::VectorMemState state(amdgpu::GLOBAL_MEM);
    state.elem_size = 4;
    state.num_elems = 1;
    if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
      cdna5::VbufferMachineInst inst{};
      inst.rsrc = 104;
      inst.soffset = cdna5::OPR_SREG_M0_NULL;
      cdna5::mubuf_calculate_addresses(inst, *wf, state);
    } else if (arch == ROCJITSU_CODE_ARCH_RDNA1) {
      rdna1::MubufMachineInst inst{};
      inst.srsrc = 26;
      inst.soffset = 128;
      rdna1::mubuf_calculate_addresses(inst, *wf, state);
    } else if (arch == ROCJITSU_CODE_ARCH_RDNA2) {
      rdna2::MubufMachineInst inst{};
      inst.srsrc = 26;
      inst.soffset = 128;
      rdna2::mubuf_calculate_addresses(inst, *wf, state);
    } else if (arch == ROCJITSU_CODE_ARCH_RDNA3_5) {
      rdna3_5::MubufMachineInst inst{};
      inst.srsrc = 26;
      inst.soffset = 128;
      rdna3_5::mubuf_calculate_addresses(inst, *wf, state);
    } else if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
      rdna3::MubufMachineInst inst{};
      inst.srsrc = 26;
      inst.soffset = 128;
      rdna3::mubuf_calculate_addresses(inst, *wf, state);
    } else {
      rdna4::VbufferMachineInst inst{};
      inst.rsrc = 104;
      inst.soffset = rdna4::OPR_SREG_M0_NULL;
      rdna4::mubuf_calculate_addresses(inst, *wf, state);
    }
    EXPECT_EQ(state.lane_mask, 1);
    EXPECT_EQ(state.per_lane_addr[0], 0x1000);
    next->halt();
    wf->halt();
  }
}

} // namespace
