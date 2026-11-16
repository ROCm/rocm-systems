// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna3_predicate_mask_test.cpp
/// @brief CDNA3 predicate-mask semantics from hip-fpsan algebraic kernels.

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 64;
constexpr uint32_t DST_SENTINEL = 0x13579BDFu;

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  explicit Fixture(unsigned sgprs = SGPRS_PER_WF, unsigned wave_slots = 1)
      : gpu_mem("cdna3_predicate_mask_mem"), l2("cdna3_predicate_mask_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA3;
    cfg.num_wf_slots = wave_slots;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    rocjitsu::test::initialize_compute_unit(cu, "cu_cdna3_predicate_mask", cfg, &gpu_mem, &l2);
    if (!cu)
      return;
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
    wf = cu->dispatch_wf(0, 0, sgprs, VGPRS_PER_WF);
  }

  void write_vgpr(uint32_t reg, uint32_t lane, uint32_t val) {
    cu->write_vgpr(wf->vgpr_alloc().base + reg, lane, val);
  }

  uint32_t read_vgpr(uint32_t reg, uint32_t lane) const {
    return cu->read_vgpr(wf->vgpr_alloc().base + reg, lane);
  }

  uint64_t read_vgpr64(uint32_t reg, uint32_t lane) const {
    const uint32_t lo = read_vgpr(reg, lane);
    const uint32_t hi = read_vgpr(reg + 1, lane);
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }

  void write_vgpr64(uint32_t reg, uint32_t lane, uint64_t val) {
    write_vgpr(reg, lane, static_cast<uint32_t>(val));
    write_vgpr(reg + 1, lane, static_cast<uint32_t>(val >> 32));
  }

  void write_sgpr_pair(uint32_t reg, uint64_t val) {
    const uint32_t sb = wf->sgpr_alloc().base;
    cu->write_sgpr(sb + reg, static_cast<uint32_t>(val));
    cu->write_sgpr(sb + reg + 1, static_cast<uint32_t>(val >> 32));
  }

  uint64_t read_sgpr_pair(uint32_t reg) const {
    const uint32_t sb = wf->sgpr_alloc().base;
    const uint32_t lo = cu->read_sgpr(sb + reg);
    const uint32_t hi = cu->read_sgpr(sb + reg + 1);
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }

  void execute(const std::array<uint32_t, 2> &words) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(cu->execute_instruction(inst.get(), *wf).succeeded());
  }
};

TEST(Cdna3VectorStateTest, ConditionalSdwaPreservesUnselectedBytesAndLanes) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->write_exec(1);
  fx.write_vgpr(0, 0, 0x11223344);
  fx.write_vgpr(1, 0, 0xaabbcc55);
  fx.write_vgpr(2, 1, 0xdeadbeef);
  for (uint64_t condition : {0ULL, 1ULL}) {
    fx.wf->set_vcc(condition);
    fx.write_vgpr(2, 0, 0xdeadbeef);
    // v_cndmask_b32 v2, v0, v1; BYTE_1, UNUSED_PRESERVE, DWORD sources.
    fx.execute({0x000402f9, 0x06061100});
    EXPECT_EQ(fx.read_vgpr(2, 0), condition ? 0xdead55efU : 0xdead44efU);
    EXPECT_EQ(fx.read_vgpr(2, 1), 0xdeadbeefU);
  }
}

TEST(Cdna3VectorStateTest, AccumulatorMovesKeepTheSourceBankAcrossModifiers) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->write_exec(1);
  fx.write_vgpr(2, 0, 0x11223344);
  fx.write_vgpr(5, 0, 0xaabbccdd);
  fx.execute({0xd3d94002u, 0x18000105u}); // v_accvgpr_write_b32 a2, v5
  for (const auto words : {std::array{0x7e02a502u, 0u},
                           std::array{0x7e02a4fau, 0xff08e402u},    // DPP, identity quad permute
                           std::array{0x7e02a4f9u, 0x00060602u}}) { // SDWA, DWORD
    fx.execute({0xd3d94001u, 0x18000102u}); // Seed a1 with v2 before each move.
    fx.execute(words);
    fx.execute({0xd3d84009u, 0x18000101u}); // v_accvgpr_read_b32 v9, a1
    EXPECT_EQ(fx.read_vgpr(9, 0), 0xaabbccddu);
  }
}

TEST(Cdna3ScalarControlTest, PredicateSelectorsAndStatus) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  for (unsigned masks = 0; masks < 4; ++masks) {
    fx.wf->write_exec(masks & 1 ? 0 : 1ull << 32);
    fx.wf->set_vcc_raw(masks & 2 ? 0 : 1ull << 63);
    fx.execute({0xbe8201fc, 0}); // s_mov_b64 s[2:3],src_execz
    EXPECT_EQ(fx.read_sgpr_pair(2), masks & 1u);
    fx.execute({0xbe8201fb, 0}); // s_mov_b64 s[2:3],src_vccz
    EXPECT_EQ(fx.read_sgpr_pair(2), masks >> 1);
    fx.execute({0xb8820a42, 0}); // s_getreg_b32 s2,hwreg(STATUS,9,2)
    EXPECT_EQ(fx.read_sgpr_pair(2), masks);
  }
}

TEST(Cdna3ScalarControlTest, UnallocatedScalarRegistersAndTrapTemporaries) {
  Fixture fx(16);
  ASSERT_NE(fx.wf, nullptr);
  fx.write_sgpr_pair(0, 0xabcdef1234ull);
  fx.write_sgpr_pair(32, 0x1badf00dull);
  fx.execute({0xbe820120, 0}); // s_mov_b64 s[2:3],s[32:33]
  EXPECT_EQ(fx.read_sgpr_pair(2), fx.read_sgpr_pair(0));
  fx.execute({0xbea00100, 0}); // s_mov_b64 s[32:33],s[0:1]
  EXPECT_EQ(fx.read_sgpr_pair(32), 0x1badf00dull);
  fx.wf->set_ttmp(0, 0x73);
  fx.execute({0xbe82016c, 0}); // s_mov_b64 s[2:3],ttmp[0:1]
  EXPECT_EQ(fx.read_sgpr_pair(2), 0u);
  fx.execute({0xbeec0081, 0}); // s_mov_b32 ttmp0,1
  EXPECT_EQ(fx.wf->ttmp(0), 0x73u);
  fx.wf->set_in_trap_handler(true);
  fx.execute({0xbe82016c, 0});
  EXPECT_EQ(fx.read_sgpr_pair(2), 0x73u);
  fx.execute({0xbeec0081, 0});
  EXPECT_EQ(fx.wf->ttmp(0), 1u);
}

TEST(Cdna3ScalarControlTest, DescriptorSgprGranuleRoundsUpToPhysicalStorage) {
  Fixture fx(8);
  ASSERT_NE(fx.wf, nullptr);
  fx.write_sgpr_pair(0, 0x9876);
  fx.write_sgpr_pair(8, 0xabcdef1234ull);
  fx.execute({0xbe820108, 0}); // s_mov_b64 s[2:3],s[8:9]
  EXPECT_EQ(fx.read_sgpr_pair(2), 0xabcdef1234ull);
  fx.execute({0xbe880100, 0}); // s_mov_b64 s[8:9],s[0:1]
  EXPECT_EQ(fx.read_sgpr_pair(8), 0x9876u);
}

TEST(Cdna3ScalarControlTest, GetregReadsLdsCountersAndTrapAddresses) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_lds_base(512);
  fx.wf->set_lds_size(1024);
  fx.wf->wait_counters().vmcnt = 37;
  fx.wf->wait_counters().expcnt = 3;
  fx.wf->wait_counters().lgkmcnt = 9;
  fx.cu->set_trap_handler_resolver([](const amdgpu::Wavefront &) {
    return amdgpu::ComputeUnitCore::TrapHandlerConfig{0x123456788000, 0x432112340000, false};
  });
  const std::array<std::pair<unsigned, uint32_t>, 6> fields{{{6, 0x4002},
                                                             {7, (2u << 22) | 0x935},
                                                             {16, 0x34567880},
                                                             {17, 0x12},
                                                             {18, 0x21123400},
                                                             {19, 0x43}}};
  for (const auto &[id, expected] : fields) {
    fx.execute({0xb882f800u + id, 0}); // s_getreg_b32 s2,hwreg(id)
    EXPECT_EQ(fx.read_sgpr_pair(2), expected);
  }
  fx.wf->wait_counters() = {};
}

struct ForceScalarGuard {
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }

  bool orig;
};

uint64_t expected_ge_mask(uint64_t exec, const Fixture &fx, uint32_t lhs_reg, uint32_t rhs_reg) {
  uint64_t mask = 0;
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (((exec >> lane) & 1ULL) == 0)
      continue;
    if (fx.read_vgpr(lhs_reg, lane) >= fx.read_vgpr(rhs_reg, lane))
      mask |= 1ULL << lane;
  }
  return mask;
}

TEST(Cdna3PredicateMaskTest, ClassCmpxUpdatesExplicitScalarPairAndExec) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  constexpr uint64_t exec = 0xf07e153489abcdefULL;
  constexpr uint64_t vcc = 0x894af0e73825196bULL;
  // F16, F32, F64. Request positive zero (class mask bit 6) on even lanes.
  for (uint32_t opcode : {0x15u, 0x11u, 0x13u}) {
    fx.wf->set_exec(exec);
    fx.wf->set_vcc_raw(vcc);
    fx.write_sgpr_pair(6, UINT64_MAX);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      fx.write_vgpr64(2, lane, 0);
      fx.write_vgpr(4, lane, (lane & 1) ? 0 : (1u << 6));
    }
    fx.execute({0xd0000006u | (opcode << 16), 0x00020902u});
    const uint64_t expected = exec & 0x5555555555555555ULL;
    EXPECT_EQ(fx.wf->exec(), expected);
    EXPECT_EQ(fx.read_sgpr_pair(6), expected);
    EXPECT_EQ(fx.wf->vcc(), vcc);
  }
}

TEST(Cdna3VectorStateTest, F8WideningSdwaUsesOnlySourceSelection) {
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);
  fx.wf->set_exec(1);
  // Byte 0..3 encode different powers of two in the CDNA3 FNUZ formats.
  constexpr uint32_t fp8[] = {0x3d800000, 0xbf800000, 0x3f800000, 0x40800000};
  constexpr uint32_t bf8[] = {0x3b800000, 0xbf800000, 0x3f800000, 0x41800000};
  constexpr uint32_t packed_source = 0x5040c020;
  constexpr uint32_t sentinel = 0xfedcba98;
  fx.write_vgpr(2, 0, packed_source);
  fx.write_sgpr_pair(6, packed_source);
  for (uint32_t opcode : {84u, 85u, 86u, 87u}) {
    const bool packed = opcode >= 86;
    const auto &expected = (opcode & 1) ? bf8 : fp8;
    for (uint32_t selection = packed ? 4 : 0; selection < (packed ? 6u : 4u); ++selection) {
      const unsigned first = packed ? 2 * (selection - 4) : selection;
      for (bool scalar : {false, true}) {
        for (bool ignored_fields : {false, true}) {
          // Include every source sign modifier, destination WORD_1/PRESERVE,
          // CLAMP and div:2. Only the selected source byte or word affects D.
          const uint32_t fields = ignored_fields ? 0x0038f500u : 0x00000600u;
          const uint32_t src = scalar ? (1u << 23) | 6u : 2u;
          const std::array<uint32_t, 2> words = {0x7e1000f9u | (opcode << 9),
                                                 fields | (selection << 16) | src};
          std::unique_ptr<Instruction> inst(decode_valid(*fx.decoder, words.data()));
          ASSERT_NE(inst, nullptr);
          EXPECT_EQ(inst->raw_encoding()[1], words[1]);
          if (ignored_fields) {
            const auto assembly = inst->disassemble();
            EXPECT_NE(assembly.find("dst_sel:WORD_1"), std::string::npos);
            EXPECT_NE(assembly.find("dst_unused:UNUSED_PRESERVE"), std::string::npos);
          }
          amdgpu::RegisterModifiers modifiers;
          inst->amdgpu_register_modifiers(modifiers);
          EXPECT_TRUE(modifiers.sdwa);
          EXPECT_EQ(modifiers.src0_selection, selection);
          EXPECT_EQ(modifiers.dst_selection, 6u);
          RegisterSet implicit;
          inst->implicit_uses(implicit);
          EXPECT_FALSE(implicit.intersects({RegClass::VGPR, 8, 1}));
          for (unsigned lane : {0u, 1u}) {
            fx.write_vgpr(8, lane, sentinel);
            fx.write_vgpr(9, lane, sentinel);
          }
          ASSERT_TRUE(fx.cu->execute_instruction(inst.get(), *fx.wf).succeeded());
          EXPECT_EQ(fx.read_vgpr(8, 0), expected[first]);
          EXPECT_EQ(fx.read_vgpr(9, 0), packed ? expected[first + 1] : sentinel);
          EXPECT_EQ(fx.read_vgpr(8, 1), sentinel);
          EXPECT_EQ(fx.read_vgpr(9, 1), sentinel);
        }
      }
    }
  }
}

TEST(Cdna3VectorStateTest, HalfArithmeticAndSignalingCompareReportOnlyActiveCauses) {
  constexpr uint32_t invalid = 1u, overflow = 1u << 3, inexact = 1u << 5;
  ForceScalarGuard guard;
  for (bool force_scalar : {false, true}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    fx.wf->set_mode_raw(0x2f0);
    fx.write_vgpr(2, 0, 0x7bffu);
    fx.write_vgpr(4, 0, 0x7bffu);
    fx.write_vgpr(2, 63, 0x3c00u);
    fx.write_vgpr(4, 63, 0x3c00u);
    // Overflow is inactive initially. Later issues must report fresh causes
    // even after the same bits have been set in the sticky register.
    fx.wf->set_exec(1ULL << 63);
    fx.wf->set_trapsts(1u << 7);
    fx.execute({0x3e100902u, 0});
    EXPECT_EQ(fx.wf->trapsts(), 1u << 7);
    fx.wf->set_exec(UINT64_MAX);
    for (const std::array<uint32_t, 2> word :
         {std::array<uint32_t, 2>{0x3e100902u, 0}, {0xd11f0008u, 0x00020902u}}) {
      fx.wf->clear_pending_alu_causes();
      fx.execute(word);
      EXPECT_EQ(fx.read_vgpr(8, 0), 0x7c00u);
      EXPECT_EQ(fx.wf->trapsts(), (1u << 7) | overflow | inexact);
      EXPECT_EQ(fx.wf->pending_alu_causes(), overflow | inexact);
    }
    fx.wf->set_trapsts(0);
    fx.wf->clear_pending_alu_causes();
    fx.write_vgpr(2, 0, 0x7fc00000u);
    fx.write_vgpr(4, 0, 0x3f800000u);
    fx.write_vgpr(2, 63, 0);
    fx.write_vgpr(4, 63, 0x3f800000u);
    fx.wf->set_exec((1ULL << 63) | 1);
    fx.wf->set_vcc_raw(0x12345678);
    fx.execute({0xd0410006u, 0x00020902u});
    EXPECT_EQ(fx.wf->trapsts(), 0u);
    fx.execute({0xd0518006u, 0x00020902u}); // CMPX LT, CLAMP, explicit s[6:7].
    EXPECT_EQ(fx.wf->trapsts(), invalid);
    EXPECT_EQ(fx.wf->pending_alu_causes(), invalid);
    EXPECT_EQ(fx.wf->exec(), 1ULL << 63);
    EXPECT_EQ(fx.read_sgpr_pair(6), 1ULL << 63);
    EXPECT_EQ(fx.wf->vcc(), 0x12345678u);
    fx.wf->set_trapsts(0);
    fx.wf->clear_pending_alu_causes();
    fx.execute({0xd0418006u, 0x00020902u});
    EXPECT_EQ(fx.wf->pending_alu_causes(), 0u);
    EXPECT_EQ(fx.wf->trapsts(), 0u);
  }
}

TEST(Cdna3VectorStateTest, HalfExceptionsFollowDppBankAndSdwaSourceSelection) {
  ForceScalarGuard guard;
  for (bool force_scalar : {false, true}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    fx.wf->set_mode_raw(0x2f0);
    fx.wf->set_exec(0x38);
    constexpr uint32_t sentinel = 0xa5a5beef;
    for (uint32_t snan_lane : {4u, 5u}) {
      for (uint32_t lane = 2; lane <= 6; ++lane) {
        fx.write_vgpr(2, lane, lane == snan_lane ? 0x7c01u : 0x3c00u);
        fx.write_vgpr(4, lane, 0x3c00u);
        fx.write_vgpr(8, lane, sentinel);
      }
      fx.wf->set_trapsts(0x80);
      fx.wf->clear_pending_alu_causes();
      // row_shl:1 routes v2[4] to lane 3 and v2[5] to lane 4.
      // bank_mask:1 enables lane 3 but suppresses lane 4.
      fx.execute({0x3e1008fau, 0xf1010102u});
      const uint32_t invalid = snan_lane == 4;
      EXPECT_EQ(fx.wf->trapsts(), 0x80u | invalid);
      EXPECT_EQ(fx.wf->pending_alu_causes(), invalid);
      EXPECT_EQ(fx.read_vgpr(8, 2), sentinel);
      EXPECT_EQ(fx.read_vgpr(8, 4), sentinel);
      EXPECT_EQ(fx.read_vgpr(8, 5), sentinel);
      if (!invalid) {
        EXPECT_EQ(fx.read_vgpr(8, 3), 0x4000u);
      }
    }
    fx.wf->set_exec(1);
    fx.write_vgpr(4, 0, 0x3c00);
    fx.write_vgpr(2, 1, 0x7c010000); // Inactive lane with a selected sNaN.
    fx.write_vgpr(8, 1, sentinel);
    for (bool high_snan : {false, true}) {
      fx.write_vgpr(2, 0, high_snan ? 0x7c013c00u : 0x3c007c01u);
      fx.write_vgpr(8, 0, sentinel);
      fx.wf->set_trapsts(0x80);
      fx.wf->clear_pending_alu_causes();
      // src0_sel:WORD_1, src1_sel:WORD_0, dst_sel:WORD_0, dst_unused:PRESERVE.
      fx.execute({0x3e1008f9u, 0x04051402u});
      EXPECT_EQ(fx.wf->trapsts(), 0x80u | uint32_t(high_snan));
      EXPECT_EQ(fx.wf->pending_alu_causes(), uint32_t(high_snan));
      EXPECT_EQ(fx.read_vgpr(8, 1), sentinel);
      EXPECT_EQ(fx.read_vgpr(8, 0) >> 16, sentinel >> 16);
      if (!high_snan) {
        EXPECT_EQ(fx.read_vgpr(8, 0), 0xa5a54000u);
      }
    }
  }
}

TEST(Cdna3VectorStateTest, SaturatingUnsignedHalfArithmetic) {
  ForceScalarGuard guard;
  for (bool force_scalar : {false, true}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    fx.wf->set_exec(1);
    fx.write_vgpr(2, 0, 0x0002ffffu);
    fx.write_vgpr(4, 0, 0x00030020u);
    // Two positive half-products, only the low half overflows.
    fx.execute({0xd381c008u, 0x18020902u});
    EXPECT_EQ(fx.read_vgpr(8, 0), 0x0006ffffu);
    fx.execute({0xd3814008u, 0x18020902u});
    EXPECT_EQ(fx.read_vgpr(8, 0), 0x0006ffe0u);
    fx.execute({0x521008f9u, 0x04042402u}); // SDWA, CLAMP and WORD_0.
    EXPECT_EQ(fx.read_vgpr(8, 0), 0xffffu);
    fx.execute({0xd1298008u, 0x00020902u}); // VOP3, CLAMP.
    EXPECT_EQ(fx.read_vgpr(8, 0), 0xffffu);
    fx.write_vgpr(2, 0, 0x7fffffffu);
    fx.write_vgpr(4, 0, 0x08070605u);
    fx.execute({0xd38bc008u, 0x18020902u});
    EXPECT_EQ(fx.read_vgpr(8, 0), 0x77f8f9fau);
  }
}

TEST(Cdna3VectorStateTest, Xf32MfmaQuantizesMultiplicandsBeforeAccumulation) {
  // First XF32 tuples in Ray's MI300X capture corpus, ROCm/rocm-systems PR 14809.
  struct Sample {
    uint32_t size, opcode, result;
    std::array<uint32_t, 8> a, b;
  };
  constexpr Sample samples[] = {{32,
                                 0x3f,
                                 0x3f844a48,
                                 {0x3e9c41a2, 0x3ed322db, 0x3f4597fe, 0x3f1a6aa0},
                                 {0xbc7654b7, 0xbeaa396c, 0x3ee7d997, 0xbe10b675}},
                                {16,
                                 0x3e,
                                 0x3f8b9e2b,
                                 {0x3e9c41a2, 0x3ed322db, 0x3ed7cc17, 0x3e36c081, 0xbf157fd5,
                                  0xbf5e5f72, 0xbe407a6f, 0x3f2ac957},
                                 {0xbc7654b7, 0xbeaa396c, 0x3f60c4a5, 0x3eede339, 0x3d028c38,
                                  0x3e0567fe, 0x3ee5e9cc, 0x3e01bf8a}}};
  ForceScalarGuard guard;
  for (bool force_scalar : {false, true}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);
    for (const auto &sample : samples) {
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        for (uint32_t item = 0; item < 2; ++item) {
          uint32_t k = (lane / sample.size) * 2 + item;
          fx.write_vgpr(item, lane, sample.a[k]);
          fx.write_vgpr(2 + item, lane, sample.b[k]);
        }
        for (uint32_t reg = 0; reg < sample.size * sample.size / WF_SIZE; ++reg)
          fx.write_vgpr(32 + reg, lane, 0x3f693ce1u);
      }
      fx.execute({0xd3800010u | (sample.opcode << 16), 0x04820500u});
      EXPECT_EQ(fx.read_vgpr(16, 0), sample.result);
    }
  }
}

TEST(Cdna3PredicateMaskTest, Vop3CompareWritesArbitrarySgprPair) {
  // v_cmp_ge_u32_e64 s[2:3], v5, v23
  constexpr std::array<uint32_t, 2> kCmpGeU32S2 = {0xD0CE0002u, 0x00022F05u};
  constexpr uint64_t kExec = 0xF0F0'0F0F'A5A5'8001ULL;
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  const std::array<uint32_t, 8> lhs = {{
      0x00000000u,
      0x00000001u,
      0x7fffffffu,
      0x80000000u,
      0xff68043au,
      0xff68043bu,
      0xff68043cu,
      0xffffffffu,
  }};
  const std::array<uint32_t, 8> rhs = {{
      0x00000001u,
      0x00000001u,
      0x80000000u,
      0x7fffffffu,
      0xff68043bu,
      0xff68043bu,
      0xff68043bu,
      0xffffffffu,
  }};

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    fx.write_vgpr(5, lane, lhs[lane % lhs.size()]);
    fx.write_vgpr(23, lane, rhs[lane % rhs.size()]);
  }
  fx.wf->set_exec(kExec);
  fx.write_sgpr_pair(2, 0xDEAD'BEEF'DEAD'BEEFULL);

  const uint64_t expected = expected_ge_mask(kExec, fx, 5, 23);
  fx.execute(kCmpGeU32S2);

  EXPECT_EQ(fx.read_sgpr_pair(2), expected);
}

TEST(Cdna3PredicateMaskTest, Vop3CndmaskReadsArbitrarySgprPair) {
  // v_cndmask_b32_e64 v33, 0, -1, s[2:3]
  constexpr std::array<uint32_t, 2> kCndmaskS2 = {0xD1000021u, 0x00098280u};
  constexpr uint64_t kExec = 0xA5A5'F0F0'1234'8001ULL;
  constexpr uint64_t kSel = 0x0123'4567'89AB'CDEFULL;
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
    fx.write_vgpr(33, lane, DST_SENTINEL);
  fx.wf->set_exec(kExec);
  fx.write_sgpr_pair(2, kSel);

  fx.execute(kCndmaskS2);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = ((kExec >> lane) & 1ULL) != 0;
    const bool selected = ((kSel >> lane) & 1ULL) != 0;
    const uint32_t expected = active ? (selected ? 0xFFFFFFFFu : 0u) : DST_SENTINEL;
    EXPECT_EQ(fx.read_vgpr(33, lane), expected) << "lane " << lane;
  }
}

TEST(Cdna3PredicateMaskTest, Vop3CompareThenCndmaskMatchesMask) {
  // Sequence from the hip-fpsan parity probe:
  //   v_cmp_ge_u32_e64 s[2:3], v5, v23
  //   v_cndmask_b32_e64 v33, 0, -1, s[2:3]
  constexpr std::array<uint32_t, 2> kCmpGeU32S2 = {0xD0CE0002u, 0x00022F05u};
  constexpr std::array<uint32_t, 2> kCndmaskS2 = {0xD1000021u, 0x00098280u};
  constexpr uint64_t kExec = 0xFFFF'FFFF'FFFF'FFFFULL;
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const uint32_t lhs = (lane % 4 == 0)   ? 0xff68043au
                         : (lane % 4 == 1) ? 0xff68043bu
                         : (lane % 4 == 2) ? 0xff68043cu
                                           : 0x00000000u;
    const uint32_t rhs = (lane % 4 == 3) ? 0xffffffffu : 0xff68043bu;
    fx.write_vgpr(5, lane, lhs);
    fx.write_vgpr(23, lane, rhs);
    fx.write_vgpr(33, lane, DST_SENTINEL);
  }
  fx.wf->set_exec(kExec);
  fx.write_sgpr_pair(2, 0);

  const uint64_t expected_mask = expected_ge_mask(kExec, fx, 5, 23);
  fx.execute(kCmpGeU32S2);
  EXPECT_EQ(fx.read_sgpr_pair(2), expected_mask);

  fx.execute(kCndmaskS2);
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const uint32_t expected = ((expected_mask >> lane) & 1ULL) ? 0xFFFFFFFFu : 0u;
    EXPECT_EQ(fx.read_vgpr(33, lane), expected) << "lane " << lane;
  }
}

TEST(Cdna3PredicateMaskTest, Vop3CompareU64WithScalarSource) {
  // v_cmp_eq_u64_e64 s[0:1], s[2:3], v[6:7]
  constexpr std::array<uint32_t, 2> kCmpEqU64S0 = {0xD0EA0000u, 0x00020C02u};
  constexpr uint64_t kExec = 0x0F0F'F0F0'AAAA'5555ULL;
  constexpr uint64_t kModulus = 0x00000000FF68043BULL;
  Fixture fx;
  ASSERT_NE(fx.wf, nullptr);

  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const uint64_t val = (lane % 4 == 0)   ? kModulus
                         : (lane % 4 == 1) ? kModulus - 1
                         : (lane % 4 == 2) ? kModulus + 1
                                           : 0;
    fx.write_vgpr64(6, lane, val);
  }
  fx.wf->set_exec(kExec);
  fx.write_sgpr_pair(0, 0xDEAD'BEEF'DEAD'BEEFULL);
  fx.write_sgpr_pair(2, kModulus);

  uint64_t expected = 0;
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    if (((kExec >> lane) & 1ULL) && (lane % 4 == 0))
      expected |= 1ULL << lane;
  }

  fx.execute(kCmpEqU64S0);

  EXPECT_EQ(fx.read_sgpr_pair(0), expected);
  EXPECT_EQ(fx.read_sgpr_pair(2), kModulus);
}

TEST(Cdna3PredicateMaskTest, VopcGtU64E32ScalarSourceWritesExpectedVcc) {
  // v_cmp_gt_u64_e32 vcc, s[2:3], v[8:9]
  constexpr std::array<uint32_t, 2> kCmpGtU64Vcc = {0x7DD81002u, 0u};
  constexpr uint64_t kExec = 0xF0F0'0F0F'A5A5'8001ULL;
  constexpr uint64_t kModulus = 0x00000000FF65CF7BULL;
  ForceScalarGuard guard;

  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const uint64_t val = (lane % 8 == 0)   ? kModulus - 1
                           : (lane % 8 == 1) ? kModulus
                           : (lane % 8 == 2) ? kModulus + 1
                           : (lane % 8 == 3) ? 0
                           : (lane % 8 == 4) ? 1
                           : (lane % 8 == 5) ? 0xFFFF'FFFFULL
                           : (lane % 8 == 6) ? 0x1'0000'0000ULL
                                             : 0xFFFF'FFFF'FFFF'FFFFULL;
      fx.write_vgpr64(8, lane, val);
    }
    fx.wf->set_exec(kExec);
    fx.wf->set_vcc(0xDEAD'BEEF'DEAD'BEEFULL);
    fx.write_sgpr_pair(2, kModulus);

    uint64_t expected = 0;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      if (((kExec >> lane) & 1ULL) && kModulus > fx.read_vgpr64(8, lane))
        expected |= 1ULL << lane;
    }

    fx.execute(kCmpGtU64Vcc);

    EXPECT_EQ(fx.wf->vcc(), expected) << "force_scalar=" << force_scalar;
    EXPECT_EQ(fx.read_sgpr_pair(2), kModulus) << "force_scalar=" << force_scalar;
  }
}

TEST(Cdna3PredicateMaskTest, AddmodCompareCndmaskLshlAddSequence) {
  // Sequence emitted for alg_addmod's `(n > s) ? s : s - n` selection:
  //   v_cmp_gt_u64_e32 vcc, s[2:3], v[8:9]
  //   v_cndmask_b32_e64 v5, -1, 0, vcc
  //   v_cndmask_b32_e64 v4, v1, 0, vcc
  //   v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
  constexpr std::array<uint32_t, 2> kCmpGtU64Vcc = {0x7DD81002u, 0u};
  constexpr std::array<uint32_t, 2> kCndHigh = {0xD1000005u, 0x01A900C1u};
  constexpr std::array<uint32_t, 2> kCndLow = {0xD1000004u, 0x01A90101u};
  constexpr std::array<uint32_t, 2> kLshlAdd = {0xD2080004u, 0x04210104u};
  constexpr uint64_t kExec = 0xF0F0'0F0F'A5A5'8001ULL;
  constexpr uint64_t kModulus = 0x00000000FF65CF7BULL;
  constexpr uint64_t kNegMod = 0xFFFF'FFFF'009A3085ULL;
  constexpr uint64_t kSentinel = 0x13579BDF2468ACE0ULL;
  ForceScalarGuard guard;

  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const uint64_t s = (lane % 8 == 0)   ? kModulus - 1
                         : (lane % 8 == 1) ? kModulus
                         : (lane % 8 == 2) ? kModulus + 1
                         : (lane % 8 == 3) ? 0
                         : (lane % 8 == 4) ? 1
                         : (lane % 8 == 5) ? (2 * kModulus) - 1
                         : (lane % 8 == 6) ? 0x1'0000'0000ULL
                                           : 0xFFFF'FFFF'FFFF'FFFFULL;
      fx.write_vgpr64(8, lane, s);
      fx.write_vgpr64(4, lane, kSentinel);
    }
    fx.wf->set_exec(kExec);
    fx.wf->set_vcc(0);
    fx.write_sgpr_pair(2, kModulus);
    fx.write_vgpr(1, 0, static_cast<uint32_t>(kNegMod));
    for (uint32_t lane = 1; lane < WF_SIZE; ++lane)
      fx.write_vgpr(1, lane, static_cast<uint32_t>(kNegMod));

    fx.execute(kCmpGtU64Vcc);
    fx.execute(kCndHigh);
    fx.execute(kCndLow);
    fx.execute(kLshlAdd);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const uint64_t s = fx.read_vgpr64(8, lane);
      const uint64_t expected =
          ((kExec >> lane) & 1ULL) == 0 ? kSentinel : (kModulus > s ? s : s + kNegMod);
      EXPECT_EQ(fx.read_vgpr64(4, lane), expected)
          << "force_scalar=" << force_scalar << " lane " << lane;
    }
  }
}

TEST(Cdna3PredicateMaskTest, AddmodFullLshlAddReductionSequence) {
  // Full inline alg_addmod arithmetic from the probe:
  //   v_lshl_add_u64 v[8:9], v[2:3], 0, v[28:29]  ; s = a + b
  //   v_cmp_gt_u64_e32 vcc, s[2:3], v[8:9]
  //   v_cndmask_b32_e64 v5, -1, 0, vcc
  //   v_cndmask_b32_e64 v4, v1, 0, vcc
  //   v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
  constexpr std::array<uint32_t, 2> kAddInputs = {0xD2080008u, 0x04710102u};
  constexpr std::array<uint32_t, 2> kCmpGtU64Vcc = {0x7DD81002u, 0u};
  constexpr std::array<uint32_t, 2> kCndHigh = {0xD1000005u, 0x01A900C1u};
  constexpr std::array<uint32_t, 2> kCndLow = {0xD1000004u, 0x01A90101u};
  constexpr std::array<uint32_t, 2> kReduce = {0xD2080004u, 0x04210104u};
  constexpr uint64_t kExec = 0xF0F0'0F0F'A5A5'8001ULL;
  constexpr uint64_t kModulus = 0x00000000FF65CF7BULL;
  constexpr uint64_t kNegMod = 0xFFFF'FFFF'009A3085ULL;
  constexpr uint64_t kSentinel = 0x13579BDF2468ACE0ULL;
  ForceScalarGuard guard;

  for (bool force_scalar : {true, false}) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx;
    ASSERT_NE(fx.wf, nullptr);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const uint64_t a = (lane % 8 == 0)   ? 0
                         : (lane % 8 == 1) ? 1
                         : (lane % 8 == 2) ? 2
                         : (lane % 8 == 3) ? kModulus - 1
                         : (lane % 8 == 4) ? kModulus - 2
                         : (lane % 8 == 5) ? 0x1'0000'0000ULL
                         : (lane % 8 == 6) ? 0x7FFF'FFFFULL
                                           : 0xFFFF'FFFFULL;
      const uint64_t b = (lane % 8 == 0)   ? kModulus - 1
                         : (lane % 8 == 1) ? kModulus - 1
                         : (lane % 8 == 2) ? kModulus - 1
                         : (lane % 8 == 3) ? 1
                         : (lane % 8 == 4) ? 3
                         : (lane % 8 == 5) ? 1
                         : (lane % 8 == 6) ? 0x8000'0000ULL
                                           : 2;
      fx.write_vgpr64(2, lane, a);
      fx.write_vgpr64(28, lane, b);
      fx.write_vgpr64(4, lane, kSentinel);
      fx.write_vgpr64(8, lane, kSentinel);
    }
    fx.wf->set_exec(kExec);
    fx.wf->set_vcc(0);
    fx.write_sgpr_pair(2, kModulus);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
      fx.write_vgpr(1, lane, static_cast<uint32_t>(kNegMod));

    fx.execute(kAddInputs);
    fx.execute(kCmpGtU64Vcc);
    fx.execute(kCndHigh);
    fx.execute(kCndLow);
    fx.execute(kReduce);

    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      const bool active = ((kExec >> lane) & 1ULL) != 0;
      const uint64_t a = fx.read_vgpr64(2, lane);
      const uint64_t b = fx.read_vgpr64(28, lane);
      const uint64_t s = a + b;
      const uint64_t expected_sum = active ? s : kSentinel;
      const uint64_t expected_reduced = active ? (kModulus > s ? s : s + kNegMod) : kSentinel;
      EXPECT_EQ(fx.read_vgpr64(8, lane), expected_sum)
          << "force_scalar=" << force_scalar << " lane " << lane;
      EXPECT_EQ(fx.read_vgpr64(4, lane), expected_reduced)
          << "force_scalar=" << force_scalar << " lane " << lane;
    }
  }
}

} // namespace
