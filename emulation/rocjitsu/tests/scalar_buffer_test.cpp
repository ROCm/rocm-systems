// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "counting_gpu_memory.h"
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
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
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

class ScalarBufferInstructionTest : public ::testing::TestWithParam<rj_code_arch_t> {
protected:
  bool gfx9() const {
    const auto arch = GetParam();
    return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
  }
};

INSTANTIATE_TEST_SUITE_P(AllArchitectures, ScalarBufferInstructionTest,
                         ::testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                           ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                           ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA1,
                                           ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                                           ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5));

TEST(ScalarBufferTest, TranslatedNarrowLoadsSuppressMaskedReadsAndZeroDestinations) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    amdgpu::GpuMemory memory("narrow_scalar_memory");
    amdgpu::L2Cache l2("narrow_scalar_l2");
    l2.set_backing_memory(&memory);
    amdgpu::L1ScalarCache l1(&l2);
    amdgpu::ScalarMemPipeline pipeline(&l1);
    amdgpu::GpuVm vm;
    auto physical = std::make_shared<test::CountingGpuMemory>(memory);
    const auto address_space = vm.register_address_space(
        0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(), physical);
    ASSERT_TRUE(address_space);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("narrow_scalar", cfg, &memory, &l2);
    cu->set_gpu_vm(&vm);
    auto *wf = cu->dispatch_wf(0, 0, 106, 16);
    ASSERT_NE(wf, nullptr);
    wf->set_address_space(address_space);
    auto decoder = Decoder::create(arch);
    constexpr uint32_t base = 0x1000;
    memory.write32(base, 0xfedcba98);
    const auto sb = wf->sgpr_alloc().base;
    for (bool faulting : {false, true}) {
      physical->fault_reads = faulting;
      for (bool masked : {true, false}) {
        const uint32_t records = masked ? 0 : 4;
        cu->write_sgpr(sb, base);
        cu->write_sgpr(sb + 1, arch == ROCJITSU_CODE_ARCH_CDNA5 ? records << 25 : 0);
        cu->write_sgpr(sb + 2, arch == ROCJITSU_CODE_ARCH_CDNA5 ? 0 : records);
        cu->write_sgpr(sb + 3, 0);
        for (uint32_t op : {24u, 25u, 26u, 27u}) {
          SCOPED_TRACE(testing::Message() << "arch=" << arch << " op=" << op << " masked=" << masked
                                          << " faulting=" << faulting);
          const std::array<uint32_t, 2> words{0xf4000200u | (op << 13), 0xf8000000u};
          std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
          ASSERT_NE(inst, nullptr);
          cu->write_sgpr(sb + 8, 0xdeadbeef);
          ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
          ASSERT_EQ(inst->data_as<amdgpu::ScalarMemState>()->load_dword_mask, masked ? 0 : 1);
          physical->reads = 0;
          const auto result = pipeline.issue(inst.release(), *wf);
          EXPECT_EQ(physical->reads, masked ? 0 : 1);
          EXPECT_EQ(result, faulting && !masked ? amdgpu::VmAccessOutcome::Faulted
                                                : amdgpu::VmAccessOutcome::Complete);
          const uint32_t expected = op == 24   ? 0xffffff98u
                                    : op == 25 ? 0x98u
                                    : op == 26 ? 0xffffba98u
                                               : 0xba98u;
          EXPECT_EQ(cu->read_sgpr(sb + 8), masked ? 0 : faulting ? 0xdeadbeef : expected);
        }
      }
    }
    wf->halt();
  }
}

TEST_P(ScalarBufferInstructionTest, ScalarBufferLoadsZeroOnlyOutOfRangeDwords) {
  for (bool translated : {false, true}) {
    SCOPED_TRACE(translated);
    const auto arch = GetParam();
    const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
    amdgpu::GpuMemory mem("scalar_buffer_mem");
    amdgpu::L2Cache l2("scalar_buffer_l2");
    l2.set_backing_memory(&mem);
    amdgpu::L1ScalarCache l1(&l2);
    amdgpu::ScalarMemPipeline pipeline(&l1);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = gfx9() ? 102 : 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("scalar_buffer", cfg, &mem, &l2);
    auto decoder = Decoder::create(arch);
    auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, 16);
    ASSERT_NE(wf, nullptr);
    amdgpu::GpuVm vm;
    if (translated) {
      const auto address_space =
          vm.register_address_space(0, std::make_shared<amdgpu::IdentityAddressSpaceTranslator>(),
                                    std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(mem));
      ASSERT_TRUE(address_space);
      cu->set_gpu_vm(&vm);
      wf->set_address_space(address_space);
    }
    const auto sb = wf->sgpr_alloc().base;
    constexpr uint64_t base = 0x1000;
    for (uint32_t i = 0; i < 32; ++i)
      mem.write32(base + 4 * i, 100 + i);
    for (uint32_t stride : {0u, 4u}) {
      const uint32_t records = stride ? 3 : 12;
      cu->write_sgpr(sb, base);
      if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
        cu->write_sgpr(sb + 1, records << 25);
        cu->write_sgpr(sb + 2, 0);
        // STRIDE_SCALE must be ignored by scalar loads.
        cu->write_sgpr(sb + 3, (stride << 12) | (3u << 26));
      } else {
        cu->write_sgpr(sb + 1, stride << 16);
        cu->write_sgpr(sb + 2, records);
        cu->write_sgpr(sb + 3, 0);
      }
      for (uint32_t offset : {0u, 4u, 12u, 16u}) {
        for (uint32_t i = 0; i < 4; ++i)
          cu->write_sgpr(sb + 8 + i, 0xdeadbeef);
        // s_buffer_load_{dwordx4,b128} s[8:11], s[0:3], offset.
        const uint32_t null_offset =
            (arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2) ? 0xfa000000u
                                                                                   : 0xf8000000u;
        const std::array<uint32_t, 2> words{gfx9()  ? 0xc02a0200u
                                            : gfx12 ? 0xf4024200u
                                                    : 0xf4280200u,
                                            offset | (gfx9() ? 0u : null_offset)};
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
        ASSERT_NE(inst, nullptr);
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        ASSERT_NE(inst->data(), nullptr);
        pipeline.issue(inst.release(), *wf);
        for (uint32_t i = 0; i < 4; ++i)
          EXPECT_EQ(cu->read_sgpr(sb + 8 + i), offset + i * 4 < 12 ? 100 + offset / 4 + i : 0)
              << "offset=" << offset << " stride=" << stride << " dword=" << i;
      }
    }
    // Fully out-of-bounds loads must not touch even an unmapped base address.
    cu->write_sgpr(sb, 0);
    cu->write_sgpr(sb + 1, 0);
    cu->write_sgpr(sb + 2, 0);
    cu->write_sgpr(sb + 3, 0);
    const uint32_t null_offset =
        (arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2) ? 0xfa000000u
                                                                               : 0xf8000000u;
    const std::array<uint32_t, 2> words{gfx9()  ? 0xc02a0200u
                                        : gfx12 ? 0xf4024200u
                                                : 0xf4280200u,
                                        gfx9() ? 0u : null_offset};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
    ASSERT_NE(inst->data(), nullptr);
    pipeline.issue(inst.release(), *wf);
    for (uint32_t i = 0; i < 4; ++i)
      EXPECT_EQ(cu->read_sgpr(sb + 8 + i), 0u);
    wf->halt();
  }
}

TEST(RdnaAddrCalcTest, ScalarAndVectorBuffersUseCanonicalHighAddresses) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(static_cast<int>(arch));
    amdgpu::GpuMemory mem("high_buffer_mem");
    amdgpu::L2Cache l2("high_buffer_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 128;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("high_buffer_cu", cfg, &mem, &l2);
    auto *wf = cu->dispatch_wf(0, 0, 128, 16);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);
    // RADV places buffers in the high half of the 48-bit GPU VA space.
    constexpr uint64_t base = 0xffff800108200000ull;
    const uint32_t s = wf->sgpr_alloc().base;
    cu->write_sgpr(s + 4, static_cast<uint32_t>(base));
    cu->write_sgpr(s + 5, (base >> 32) & 0xffff);
    cu->write_sgpr(s + 6, 1024);
    cu->write_sgpr(s + 7, 0x30016fac);
    cu->write_vgpr(wf->vgpr_alloc().base, 0, 12);
    amdgpu::VectorMemState vector(amdgpu::GLOBAL_MEM);
    std::optional<uint64_t> scalar;
    if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
      rdna3::MubufMachineInst inst{};
      inst.srsrc = 1;
      inst.soffset = 0x80;
      inst.offen = 1;
      rdna3::mubuf_calculate_addresses(inst, *wf, vector);
      rdna3::MtbufMachineInst typed{};
      typed.srsrc = 1;
      typed.soffset = 0x80;
      typed.offen = 1;
      amdgpu::VectorMemState typed_vector(amdgpu::GLOBAL_MEM);
      rdna3::mtbuf_calculate_addresses(typed, *wf, typed_vector);
      EXPECT_EQ(typed_vector.lane_mask, 1);
      EXPECT_EQ(typed_vector.per_lane_addr[0], base + 12);
      rdna3::SmemMachineInst smem{};
      smem.op = 8; // s_buffer_load_b32
      smem.sbase = 2;
      smem.soffset = rdna3::OPR_SMEM_OFFSET_NULL;
      smem.offset = 12;
      scalar = rdna3::smem_calculate_address(smem, *wf);
    } else {
      rdna4::VbufferMachineInst inst{};
      inst.rsrc = 4;
      inst.soffset = rdna4::OPR_SREG_M0_NULL;
      inst.offen = 1;
      rdna4::mubuf_calculate_addresses(inst, *wf, vector);
      rdna4::SmemMachineInst smem{};
      smem.op = 16; // s_buffer_load_b32
      smem.sbase = 2;
      smem.soffset = rdna4::OPR_SMEM_OFFSET_NULL;
      smem.ioffset = 12;
      for (uint32_t opcode : {16u, 17u, 18u, 19u, 20u, 21u, 24u, 25u, 26u, 27u}) {
        smem.op = opcode;
        scalar = rdna4::smem_calculate_address(smem, *wf);
        ASSERT_TRUE(scalar);
        EXPECT_EQ(*scalar, base + 12) << "opcode " << opcode;
      }
    }
    ASSERT_TRUE(scalar);
    EXPECT_EQ(*scalar, base + 12);
    EXPECT_EQ(vector.lane_mask, 1);
    EXPECT_EQ(vector.per_lane_addr[0], base + 12);
  }
}

TEST(RdnaAddrCalcTest, ScalarBufferLoadsMaskEachDwordAndHonorStride) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4}) {
    amdgpu::GpuMemory mem("scalar_bounds_mem");
    amdgpu::L2Cache l2("scalar_bounds_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("scalar_bounds", cfg, &mem, &l2);
    auto *wf = cu->dispatch_wf(0, 0, 106, 16);
    ASSERT_NE(wf, nullptr);
    const uint32_t sb = wf->sgpr_alloc().base;
    constexpr uint64_t base = 0xffff800000001000ull;
    for (uint32_t stride : {0u, 4u}) {
      cu->write_sgpr(sb, uint32_t(base));
      cu->write_sgpr(sb + 1, (uint32_t(base >> 32) & 0xffff) | (stride << 16));
      // Both descriptors expose 12 bytes. A 16-byte load at offset 4
      // must return two dwords from memory and two zero dwords.
      cu->write_sgpr(sb + 2, stride ? 3 : 12);
      for (uint32_t offset : {0u, 4u, 12u, 16u}) {
        amdgpu::ScalarMemState state;
        state.num_dwords = 4;
        std::optional<uint64_t> address;
        if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
          rdna3::SmemMachineInst inst{};
          inst.op = 10;
          inst.offset = offset;
          inst.soffset = rdna3::OPR_SMEM_OFFSET_NULL;
          address = rdna3::smem_calculate_address(inst, *wf, &state);
        } else {
          rdna4::SmemMachineInst inst{};
          inst.op = 18;
          inst.ioffset = offset;
          inst.soffset = rdna4::OPR_SMEM_OFFSET_NULL;
          address = rdna4::smem_calculate_address(inst, *wf, &state);
        }
        ASSERT_TRUE(address);
        EXPECT_EQ(*address, base + offset);
        EXPECT_EQ(state.load_dword_mask, offset == 0 ? 7u : offset == 4 ? 3u : 0u);
      }
    }
    if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
      cu->write_sgpr(sb + 1, uint32_t(base >> 32) & 0xffff);
      cu->write_sgpr(sb + 2, 12);
      for (uint32_t size : {1u, 2u}) {
        for (uint32_t offset : {1u, 2u, 3u, 11u, 12u}) {
          rdna4::SmemMachineInst inst{};
          inst.op = size == 1 ? 24 : 26;
          inst.ioffset = offset;
          inst.soffset = rdna4::OPR_SMEM_OFFSET_NULL;
          amdgpu::ScalarMemState state;
          state.num_dwords = 1;
          state.elem_size = size;
          const auto address = rdna4::smem_calculate_address(inst, *wf, &state);
          ASSERT_TRUE(address);
          EXPECT_EQ(*address, (base + offset) & ~uint64_t(size - 1));
          EXPECT_EQ(state.load_dword_mask, offset < 12 ? 1u : 0u);
        }
      }
    }
    wf->halt();
  }
}

TEST(Gfx1250AddrCalcTest, ScalarBufferUsesWideDescriptorAndUnscaledStride) {
  amdgpu::GpuMemory mem("wide_scalar_buffer_mem");
  amdgpu::L2Cache l2("wide_scalar_buffer_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 16;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("wide_scalar_buffer", cfg, &mem, &l2);
  auto *wf = cu->dispatch_wf(0, 0, 106, 16);
  ASSERT_NE(wf, nullptr);
  const auto sb = wf->sgpr_alloc().base;
  constexpr uint64_t base = 0x123456789abc000ull;
  const auto descriptor = [&](uint64_t records, uint32_t stride) {
    cu->write_sgpr(sb, uint32_t(base));
    cu->write_sgpr(sb + 1, uint32_t(base >> 32) | (uint32_t(records & 127) << 25));
    cu->write_sgpr(sb + 2, uint32_t(records >> 7));
    cu->write_sgpr(sb + 3, uint32_t(records >> 39) | (stride << 12) | (3u << 26));
  };
  cdna5::SmemMachineInst inst{};
  inst.op = 18;
  inst.soffset = cdna5::OPR_SREG_M0_NULL;
  inst.ioffset = 4;
  amdgpu::ScalarMemState state;
  state.num_dwords = 4;
  state.elem_size = 4;
  descriptor(3, 4);
  EXPECT_EQ(cdna5::smem_calculate_address(inst, *wf, 16, &state), base + 4);
  EXPECT_EQ(state.load_dword_mask, 3u);
  descriptor((uint64_t{1} << 40) | 3, 4);
  EXPECT_EQ(cdna5::smem_calculate_address(inst, *wf, 16, &state), base + 4);
  EXPECT_EQ(state.load_dword_mask, 15u);
  descriptor(3, 4);
  inst.soffset = 4;
  cu->write_sgpr(sb + 4, 5);
  inst.scale_offset = 1; // This bit has no effect on buffer loads.
  inst.ioffset = 7;
  EXPECT_EQ(cdna5::smem_calculate_address(inst, *wf, 16, &state), base + 8);
  EXPECT_EQ(state.load_dword_mask, 1u);
  for (uint32_t size : {1u, 2u}) {
    state.num_dwords = 1;
    state.elem_size = size;
    inst.op = size == 1 ? 25 : 27;
    inst.ioffset = 3;
    inst.soffset = cdna5::OPR_SREG_M0_NULL;
    descriptor(4, 0);
    EXPECT_EQ(cdna5::smem_calculate_address(inst, *wf, size, &state), base + (size == 1 ? 3 : 2));
    EXPECT_EQ(state.load_dword_mask, 1u);
    inst.ioffset = 4;
    EXPECT_EQ(cdna5::smem_calculate_address(inst, *wf, size, &state), base + 4);
    EXPECT_EQ(state.load_dword_mask, 0u);
  }
  wf->halt();
}

TEST(Gfx12AddrCalcTest, ScalarLoadsAlignEachComponentAndBufferSize) {
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    amdgpu::GpuMemory mem("scalar_component_mem");
    amdgpu::L2Cache l2("scalar_component_l2");
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 16;
    cfg.lds_size_kb = 64;
    auto cu = amdgpu::ComputeUnitCore::create("scalar_component", cfg, &mem, &l2);
    auto *wf = cu->dispatch_wf(0, 0, 106, 16);
    ASSERT_NE(wf, nullptr);
    const auto sb = wf->sgpr_alloc().base;
    for (uint32_t size : {1u, 2u, 4u}) {
      for (bool buffer : {false, true}) {
        const auto address = [&](uint32_t base, uint32_t immediate, uint32_t reg, uint32_t records,
                                 amdgpu::ScalarMemState &state) {
          cu->write_sgpr(sb, base);
          cu->write_sgpr(sb + 1, buffer && arch == ROCJITSU_CODE_ARCH_CDNA5 ? records << 25 : 0);
          cu->write_sgpr(sb + 2, arch == ROCJITSU_CODE_ARCH_CDNA5 ? 0 : records);
          cu->write_sgpr(sb + 3, 0);
          cu->write_sgpr(sb + 4, reg);
          const uint32_t opcode = (size == 1 ? 9 : size == 2 ? 11 : 0) + (buffer ? 16 : 0);
          if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
            cdna5::SmemMachineInst inst{};
            inst.op = opcode;
            inst.ioffset = immediate;
            inst.soffset = 4;
            return cdna5::smem_calculate_address(inst, *wf, size, &state);
          }
          rdna4::SmemMachineInst inst{};
          inst.op = opcode;
          inst.ioffset = immediate;
          inst.soffset = 4;
          return rdna4::smem_calculate_address(inst, *wf, &state);
        };
        amdgpu::ScalarMemState state;
        state.num_dwords = 1;
        state.elem_size = size;
        EXPECT_EQ(address(0x1001, 3, 5, 64, state), size == 1   ? 0x1009u
                                                    : size == 2 ? 0x1006u
                                                                : 0x1004u);
        if (buffer) {
          EXPECT_EQ(state.load_dword_mask, 1u);
          EXPECT_EQ(address(0x1000, size, 0, size + 1, state), 0x1000u + size);
          EXPECT_EQ(state.load_dword_mask, size == 1 ? 1u : 0u);
          EXPECT_EQ(address(0x1000, 0, 0, size - 1, state), 0x1000u);
          EXPECT_EQ(state.load_dword_mask, 0u);
        }
      }
    }
    wf->halt();
  }
}

} // namespace
