// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <format>
#include <memory>

namespace {

using namespace rocjitsu;

struct ForceScalarGuard {
  explicit ForceScalarGuard(bool force) : previous(util::force_scalar()) {
    util::set_force_scalar_for_testing(force);
  }
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(previous); }
  bool previous;
};

struct DotCase {
  uint32_t opcode;
  const char *mnemonic;
  uint32_t one0, two0, one1;
};

constexpr DotCase kDots[] = {
    {rdna4::kVDot2F32F16Vop3p, "v_dot2_f32_f16", 0x3c00, 0x4000, 0x3c00},
    {rdna4::kVDot2F32Bf16Vop3p, "v_dot2_f32_bf16", 0x3f80, 0x4000, 0x3f80},
    {rdna4::kVDot4F32Fp8Bf8Vop3p, "v_dot4_f32_fp8_bf8", 0x38, 0x40, 0x3c},
    {rdna4::kVDot4F32Bf8Fp8Vop3p, "v_dot4_f32_bf8_fp8", 0x3c, 0x40, 0x38},
    {rdna4::kVDot4F32Fp8Fp8Vop3p, "v_dot4_f32_fp8_fp8", 0x38, 0x40, 0x38},
    {rdna4::kVDot4F32Bf8Bf8Vop3p, "v_dot4_f32_bf8_bf8", 0x3c, 0x40, 0x3c},
};

TEST(Rdna4DotDppTest, DecodesHardwareQualifiedForms) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  for (const auto &dot : kDots) {
    // LLVM gfx1201 encodings, also executed on physical gfx1201:
    // v_dot* v16, v8, v9, v10 row_shr:1 row_mask:3 bank_mask:13 bound_ctrl:1
    // v_dot* v16, v8, v9, v10 dpp8:[7,6,5,4,3,2,1,0] fi:1
    for (const bool dpp8 : {false, true}) {
      SCOPED_TRACE(dot.mnemonic);
      const uint32_t words[] = {0xcc004010u | (dot.opcode << 16), dpp8 ? 0x1c2a12eau : 0x1c2a12fau,
                                dpp8 ? 0x05397708u : 0x3d091108u};
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(inst->mnemonic(), dot.mnemonic);
      EXPECT_EQ(inst->size(), 12u);
      EXPECT_EQ(inst->src_operand(0)->name(), "v8");
      EXPECT_NE(inst->disassemble().find(dpp8 ? "dpp8:" : "row_shr:1"), std::string::npos);
    }
  }
}

TEST(Rdna4DotDppTest, PermutesSourcesAndPreservesMaskedDestinations) {
  constexpr uint32_t sentinel = 0xdeadbeefu;
  for (bool force_scalar : {false, true}) {
    ForceScalarGuard guard(force_scalar);
    for (uint32_t wave_size : {32u, 64u}) {
      amdgpu::GpuMemory memory("dot_dpp_memory");
      amdgpu::L2Cache l2("dot_dpp_l2");
      amdgpu::ComputeUnitCore::Config cfg{};
      cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
      cfg.num_wf_slots = 1;
      cfg.sgprs_per_wf = 106;
      cfg.vgprs_per_wf = 32;
      cfg.lds_size_kb = 64;
      auto cu = amdgpu::ComputeUnitCore::create("dot_dpp", cfg, &memory, &l2);
      auto decoder = Decoder::create(cfg.arch);
      auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf, wave_size);
      ASSERT_NE(wf, nullptr);
      const auto vb = wf->vgpr_alloc().base;
      for (const auto &dot : kDots) {
        for (bool dpp8 : {false, true}) {
          for (bool fi : {false, true}) {
            for (uint64_t exec : {~0ull, 0xa55aa55a5aa55aa5ull}) {
              SCOPED_TRACE(std::format("{} wave{} scalar={} dpp8={} fi={} exec={:x}", dot.mnemonic,
                                       wave_size, force_scalar, dpp8, fi, exec));
              wf->set_exec(exec);
              const uint32_t words[] = {0xcc004010u | (dot.opcode << 16),
                                        0x1c2a1200u | (dpp8 ? (fi ? 234u : 233u) : 250u),
                                        dpp8 ? 0x05397708u : (0x3d091108u | (uint32_t{fi} << 18))};
              std::unique_ptr<Instruction> permuted(decode_valid(*decoder, words));
              ASSERT_NE(permuted, nullptr);
              const uint32_t plain_words[] = {0xcc004011u | (dot.opcode << 16), 0x1c2a1308u};
              std::unique_ptr<Instruction> plain(decode_valid(*decoder, plain_words));
              ASSERT_NE(plain, nullptr);
              for (uint32_t lane = 0; lane < wave_size; ++lane) {
                cu->write_vgpr(vb + 8, lane, lane % 2 ? dot.two0 : dot.one0);
                cu->write_vgpr(vb + 9, lane, dot.one1);
                cu->write_vgpr(vb + 10, lane, std::bit_cast<uint32_t>(float(lane) + 0.5f));
                cu->write_vgpr(vb + 16, lane, sentinel);
              }
              ASSERT_TRUE(cu->execute_instruction(permuted.get(), *wf).succeeded());
              // Independently construct the equivalent unpermuted source vector.
              for (uint32_t lane = 0; lane < wave_size; ++lane) {
                const uint32_t source = dpp8 ? (lane ^ 7u) : (lane ? lane - 1 : 0);
                const bool in_range = dpp8 || lane % 16 != 0;
                const bool read_source = in_range && (fi || (exec & (1ull << source)));
                cu->write_vgpr(vb + 8, lane, read_source ? (source % 2 ? dot.two0 : dot.one0) : 0u);
              }
              ASSERT_TRUE(cu->execute_instruction(plain.get(), *wf).succeeded());
              for (uint32_t lane = 0; lane < wave_size; ++lane) {
                const bool row_bank =
                    dpp8 || ((3u & (1u << (lane / 16))) && (13u & (1u << ((lane % 16) / 4))));
                const uint32_t expected =
                    (exec & (1ull << lane)) && row_bank ? cu->read_vgpr(vb + 17, lane) : sentinel;
                EXPECT_EQ(cu->read_vgpr(vb + 16, lane), expected) << "lane=" << lane;
              }
            }
          }
        }
      }
      wf->halt();
    }
  }
}

TEST(Rdna4DotDppTest, FloatingDot4UsesEachOperandFormat) {
  amdgpu::GpuMemory memory("dot4_format_memory");
  amdgpu::L2Cache l2("dot4_format_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 32;
  auto cu = amdgpu::ComputeUnitCore::create("dot4_format", cfg, &memory, &l2);
  auto decoder = Decoder::create(cfg.arch);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xffffffffu);
  const auto vb = wf->vgpr_alloc().base;
  for (bool force_scalar : {false, true}) {
    ForceScalarGuard guard(force_scalar);
    for (const auto &dot : kDots) {
      if (dot.opcode == rdna4::kVDot2F32F16Vop3p || dot.opcode == rdna4::kVDot2F32Bf16Vop3p)
        continue;
      for (unsigned form = 0; form != 3; ++form) {
        SCOPED_TRACE(std::format("{} form={} scalar={}", dot.mnemonic, form, force_scalar));
        const uint32_t words[] = {0xcc004010u | (dot.opcode << 16),
                                  form == 0 ? 0x1c2a1308u : (form == 1 ? 0x1c2a12fau : 0x1c2a12eau),
                                  form == 1 ? 0xff091108u : 0x05397708u};
        std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
        ASSERT_NE(inst, nullptr);
        for (uint32_t lane = 0; lane != 32; ++lane) {
          // E4M3 0x38 and E5M2 0x3c both encode one. Four products plus 0.5 are 4.5.
          cu->write_vgpr(vb + 8, lane, dot.one0 * 0x01010101u);
          cu->write_vgpr(vb + 9, lane, dot.one1 * 0x01010101u);
          cu->write_vgpr(vb + 10, lane, std::bit_cast<uint32_t>(0.5f));
        }
        ASSERT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
        for (uint32_t lane = 0; lane != 32; ++lane) {
          const float expected = form == 1 && lane % 16 == 0 ? 0.5f : 4.5f;
          EXPECT_EQ(cu->read_vgpr(vb + 16, lane), std::bit_cast<uint32_t>(expected))
              << "lane=" << lane;
        }
      }
    }
  }
  wf->halt();
}

} // namespace
