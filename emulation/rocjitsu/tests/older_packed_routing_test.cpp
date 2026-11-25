// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older packed-register routing: selected words/halves, sign bits, aliases and EXEC.
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <random>

namespace rocjitsu::amdgpu {
namespace {
// One register file permits overlapping operands; references always use its initial snapshot.
struct PackedRoutingFixture {
  ScopedIsaExecutionBackend backend;
  GpuMemory memory{"packed_routing_memory"};
  L2Cache l2{"packed_routing_l2"};
  std::unique_ptr<ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  Wavefront *wave = nullptr;
  uint32_t input[6][64];

  // Create a wave with default-seeded, implementation-independent high-bit test inputs.
  PackedRoutingFixture(rocjitsu::IsaFamily arch, unsigned lanes, ArithmeticSemantics semantics)
      : backend(arch == rocjitsu::IsaFamily::Cdna2 ? &cdna2::execution_backend()
                                                   : &rdna3_5::execution_backend()) {
    ComputeUnitCore::Config config{};
    config.arch = arch;
    config.arithmetic = {semantics};
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu = ComputeUnitCore::create("packed_routing", config, &memory, &l2);
    wave = cu->dispatch_wf(0, 0, 104, 16, lanes);
    decoder = Decoder::create(arch);
    std::minstd_rand random;
    for (auto &reg : input) {
      for (auto &word : reg) {
        const auto first_draw = random();
        word = (uint32_t(first_draw >> 15) << 16) | uint32_t(random() >> 15);
      }
    }
    input[0][0] = 0x7fc12345U;
    input[1][1] = 0x80000000U;
  }

  // Reset independent operands and status so each case exposes its own writes.
  void reset(uint64_t exec_mask) {
    wave->set_exec(exec_mask);
    wave->set_vgpr_write_mask(~4ULL);
    wave->set_vcc_raw(0x123456789abcdef0ULL);
    wave->write_scc(true);
    for (unsigned reg = 0; reg < 6; ++reg) {
      for (unsigned lane = 0; lane < wave->wf_size(); ++lane) {
        cu->write_vgpr(wave->vgpr_alloc().base + reg, lane, input[reg][lane]);
      }
    }
  }

  // Execute a legal encoding, requiring the diagnostic count to stay clear.
  template <std::size_t Words> void execute(const std::array<uint32_t, Words> &words) {
    auto decoded = decoder->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    const auto diagnostics = cu->isa_diagnostic_count();
    ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
    EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
    EXPECT_EQ(wave->vcc(), 0x123456789abcdef0ULL);
    EXPECT_TRUE(wave->read_scc());
  }
};

// Each source is a register pair; OP_SEL picks one word from each for the output pair.
TEST(OlderPackedRouting, Gfx90aPackedWords) {
  for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
    PackedRoutingFixture f(rocjitsu::IsaFamily::Cdna2, 64, semantics);
    ASSERT_NE(f.wave, nullptr);
    for (uint8_t opsel : {0, 1, 2, 3}) {
      for (uint8_t destination : {0, 1, 4}) {
        for (uint64_t exec_mask : {~0ULL, 0x8000000380000005ULL, 0ULL}) {
          f.reset(exec_mask);
          const auto words = cdna2::build_vop3p(cdna2::kVPkMovB32Vop3p, {.vdst = destination,
                                                                         .neg_hi = 3,
                                                                         .op_sel = opsel,
                                                                         .clamp = 1,
                                                                         .src0 = 256,
                                                                         .src1 = 258,
                                                                         .op_sel_hi = 3,
                                                                         .neg = 3});
          ASSERT_NO_FATAL_FAILURE(f.execute(words));
          for (unsigned reg = 0; reg < 6; ++reg) {
            for (unsigned lane = 0; lane < 64; ++lane) {
              uint32_t expected = f.input[reg][lane];
              if ((((exec_mask & ~4ULL) >> lane) & 1) &&
                  (reg == destination || reg == unsigned(destination) + 1)) {
                expected = f.input[reg == destination ? (opsel & 1) : 2 + ((opsel >> 1) & 1)][lane];
              }
              EXPECT_EQ(f.cu->read_vgpr(f.wave->vgpr_alloc().base + reg, lane), expected)
                  << unsigned(opsel) << ',' << unsigned(destination) << ',' << reg << ',' << lane;
            }
          }
          EXPECT_EQ(f.wave->exec(), exec_mask);
        }
      }
    }
  }
}

// The explicit scalar mask selects a half before destination merging; VCC is unchanged.
TEST(OlderPackedRouting, Gfx1150ConditionalHalf) {
  constexpr uint64_t condition = 0xcccc33335555aaaaULL;
  for (unsigned lanes : {32U, 64U}) {
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
      PackedRoutingFixture f(rocjitsu::IsaFamily::Rdna3_5, lanes, semantics);
      ASSERT_NE(f.wave, nullptr);
      for (uint8_t opsel : {0, 3, 9, 10}) {
        for (uint8_t destination : {0, 2, 4}) {
          for (uint64_t exec_mask : {~0ULL, 0x8000000380000005ULL, 0ULL}) {
            exec_mask &= lanes == 32 ? 0xffffffffULL : ~0ULL;
            f.reset(exec_mask);
            const auto sb = f.wave->sgpr_alloc().base;
            f.cu->write_sgpr(sb + 4, uint32_t(condition));
            f.cu->write_sgpr(sb + 5, uint32_t(condition >> 32));
            const auto words = rdna3_5::build_vop3(rdna3_5::kVCndmaskB16Vop3, {.vdst = destination,
                                                                               .abs = 1,
                                                                               .op_sel = opsel,
                                                                               .src0 = 256,
                                                                               .src1 = 258,
                                                                               .src2 = 4,
                                                                               .neg = 2});
            ASSERT_NO_FATAL_FAILURE(f.execute(words));
            for (unsigned reg = 0; reg < 6; ++reg) {
              for (unsigned lane = 0; lane < lanes; ++lane) {
                uint32_t expected = f.input[reg][lane];
                if (reg == destination && (((exec_mask & ~4ULL) >> lane) & 1)) {
                  const bool b = (condition >> lane) & 1;
                  const unsigned shift = (opsel & (b ? 2 : 1)) ? 16 : 0;
                  uint32_t half = (f.input[b ? 2 : 0][lane] >> shift) & 0xffffU;
                  half = b ? half ^ 0x8000U : half & 0x7fffU;
                  const unsigned out_shift = opsel & 8 ? 16 : 0;
                  expected = (expected & ~(0xffffU << out_shift)) | (half << out_shift);
                }
                EXPECT_EQ(f.cu->read_vgpr(f.wave->vgpr_alloc().base + reg, lane), expected)
                    << lanes << ',' << unsigned(opsel) << ',' << reg << ',' << lane;
              }
            }
            EXPECT_EQ(f.wave->exec(), exec_mask);
          }
        }
      }
    }
  }
}
// DPP stages selected high halves before aliased writes, including inactive source lanes.
TEST(OlderPackedRouting, Gfx1150ConditionalHalfDpp) {
  for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
    PackedRoutingFixture f(rocjitsu::IsaFamily::Rdna3_5, 32, semantics);
    ASSERT_NE(f.wave, nullptr);
    constexpr uint32_t exec_mask = 0x8001000fU, condition = 5;
    f.reset(exec_mask);
    f.cu->write_sgpr(f.wave->sgpr_alloc().base + 4, condition);
    rdna3_5::Vop3VopDpp16MachineInst raw{};
    raw.encoding = rdna3_5::build_vop3(rdna3_5::kVCndmaskB16Vop3)[0] >> 26;
    raw.op = rdna3_5::kVCndmaskB16Vop3;
    raw.vdst = 0;
    raw.abs = 1;
    raw.neg = 2;
    raw.op_sel = 9; // High A to high D; B uses its low half.
    raw.src0 = SRC_DPP;
    raw.src1 = 258;
    raw.src2 = 4;
    raw.vsrc0 = 0;
    raw.dpp_ctrl = dpp::ROW_SHR1;
    raw.fi = 1;
    raw.bound_ctrl = 1;
    raw.row_mask = 0xf;
    raw.bank_mask = 0xf;
    auto words = std::bit_cast<std::array<uint32_t, sizeof(raw) / sizeof(uint32_t)>>(raw);
    ASSERT_NO_FATAL_FAILURE(f.execute(words));
    for (unsigned lane = 0; lane < 32; ++lane) {
      uint32_t expected = f.input[0][lane];
      if (((exec_mask & ~4U) >> lane) & 1) {
        const uint32_t half = (condition >> lane) & 1 ? (f.input[2][lane] ^ 0x8000U) & 0xffffU
                              : lane % 16             ? (f.input[0][lane - 1] >> 16) & 0x7fffU
                                                      : 0;
        expected = (expected & 0xffffU) | (half << 16);
      }
      EXPECT_EQ(f.cu->read_vgpr(f.wave->vgpr_alloc().base, lane), expected) << lane;
    }
    EXPECT_EQ(f.wave->exec(), exec_mask);
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
