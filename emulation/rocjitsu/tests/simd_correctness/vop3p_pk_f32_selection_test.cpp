// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3p_pk_f32_selection_test.cpp
/// @brief Compare packed FP32 ADD/MUL/FMA source selection and floating-point
/// policies between scalar and SIMD execution on CDNA2 through CDNA5.
/// Only FMA NaN sign/payload selection may differ; both results must be quiet NaNs.

#include "decode_test_util.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <array>
#include <cfenv>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <tuple>

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace {
using namespace rocjitsu;

struct HostState {
  std::fenv_t environment;
  bool force_scalar = util::force_scalar();
  HostState() { std::fegetenv(&environment); }
  ~HostState() {
    std::fesetenv(&environment);
    util::set_force_scalar_for_testing(force_scalar);
  }
};

// Distinct source halves and lanes cover cancellation, rounding boundaries,
// overflow, both signs of zero/subnormals/infinity, and quiet/signaling NaNs.
constexpr std::array<uint32_t, 20> kBits = {
    0,          0x80000000, 0x3f800000, 0xbf800000, 0x3f800001, 0x33800000, 0x00800000,
    0x007fffff, 1,          0x80000001, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
    0x7fc12345, 0x7f812345, 0xffc54321, 0xff854321, 0x3f000000, 0xbf000000};
constexpr uint32_t kSentinel = 0xdeadbeef;

enum class SourceKind { Vgpr, Sgpr, InlineFloat, Literal };

struct Machine {
  amdgpu::GpuMemory memory{"packed_f32_selection_memory"};
  amdgpu::L2Cache cache{"packed_f32_selection_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave;

  explicit Machine(rj_code_arch_t arch) {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cache.set_backing_memory(&memory);
    cu = amdgpu::ComputeUnitCore::create("packed_f32_selection", config, &memory, &cache);
    decoder = Decoder::create(arch);
    wave = cu->dispatch_wf(0, 0, 106, 256);
  }
  ~Machine() { wave->halt(); }

  using Result = std::array<std::array<uint32_t, 2>, 64>;
  Result run(Instruction &inst, bool scalar, uint32_t mode, uint64_t exec, uint32_t dst) {
    util::set_force_scalar_for_testing(scalar);
    const uint32_t base = wave->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
      for (uint32_t reg = 0; reg < 6; ++reg)
        cu->write_vgpr(base + reg, lane, kBits[(lane + 3 * reg) % kBits.size()]);
      for (uint32_t reg = 6; reg < 8; ++reg)
        cu->write_vgpr(base + reg, lane, kSentinel);
    }
    wave->debug_write_sgpr(8, 0x3f800001);
    wave->debug_write_sgpr(9, 0x33800000);
    wave->set_mode_raw(mode);
    wave->set_exec(exec);
    EXPECT_TRUE(cu->execute_instruction(&inst, *wave).succeeded());
    Result result{};
    for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
      for (uint32_t half = 0; half < 2; ++half)
        result[lane][half] = cu->read_vgpr(base + dst + half, lane);
    return result;
  }
};

std::array<uint32_t, 4> encoding(rj_code_arch_t arch, unsigned operation, uint8_t lo, uint8_t hi,
                                 uint8_t neg, uint8_t neg_hi, uint8_t clamp, uint8_t dst,
                                 SourceKind source_kind) {
  uint16_t src0 = 256;
  uint16_t src1 = 258;
  switch (source_kind) {
  case SourceKind::Vgpr:
    break;
  case SourceKind::Sgpr:
    src0 = 8;
    break;
  case SourceKind::InlineFloat:
    src0 = 242; // Inline 1.0f.
    break;
  case SourceKind::Literal:
    src1 = 255;
    break;
  }
  // CDNA2 through CDNA4 share these opcode numbers.
  const std::array<uint16_t, 3> older_ops = {cdna2::kVPkAddF32Vop3p, cdna2::kVPkMulF32Vop3p,
                                             cdna2::kVPkFmaF32Vop3p};
  const auto older = [&](auto builder, auto fields) {
    fields.vdst = dst;
    fields.neg_hi = neg_hi;
    fields.op_sel = lo;
    fields.op_sel_hi_2 = hi >> 2;
    fields.clamp = clamp;
    fields.src0 = src0;
    fields.src1 = src1;
    fields.src2 = 260;
    fields.op_sel_hi = hi & 3;
    fields.neg = neg;
    return builder(older_ops[operation], fields);
  };
  std::array<uint32_t, 2> words{};
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA2:
    words = older(cdna2::build_vop3p, cdna2::Vop3pBuilderFields{});
    break;
  case ROCJITSU_CODE_ARCH_CDNA3:
    words = older(cdna3::build_vop3p, cdna3::Vop3pBuilderFields{});
    break;
  case ROCJITSU_CODE_ARCH_CDNA4:
    words = older(cdna4::build_vop3p, cdna4::Vop3pBuilderFields{});
    break;
  case ROCJITSU_CODE_ARCH_CDNA5: {
    const std::array<uint16_t, 3> ops = {cdna5::kVPkAddF32Vop3p, cdna5::kVPkMulF32Vop3p,
                                         cdna5::kVPkFmaF32Vop3p};
    words = cdna5::build_vop3p(ops[operation], {.vdst = dst,
                                                .neg_hi = neg_hi,
                                                .opsel = lo,
                                                .opsel_hi_2 = uint8_t(hi >> 2),
                                                .clamp = clamp,
                                                .src0 = src0,
                                                .src1 = src1,
                                                .src2 = 260,
                                                .opsel_hi = uint8_t(hi & 3),
                                                .neg = neg});
    break;
  }
  default:
    ADD_FAILURE() << "unsupported architecture";
  }
  return {words[0], words[1], 0x3f000000, 0};
}

bool is_nan(uint32_t bits) { return (bits & 0x7fffffff) > 0x7f800000; }

class PackedF32Selection : public testing::TestWithParam<std::tuple<rj_code_arch_t, unsigned>> {
protected:
  void compare(Machine &machine, uint8_t lo, uint8_t hi, uint8_t neg, uint8_t neg_hi, uint32_t mode,
               uint8_t clamp, uint8_t dst, SourceKind source_kind, uint64_t exec) {
    const auto [arch, operation] = GetParam();
    SCOPED_TRACE(testing::Message() << "arch=" << arch << " op=" << operation << " lo="
                                    << unsigned(lo) << " hi=" << unsigned(hi) << " mode=" << mode
                                    << " source=" << static_cast<unsigned>(source_kind));
    const auto words = encoding(arch, operation, lo, hi, neg, neg_hi, clamp, dst, source_kind);
    std::unique_ptr<Instruction> inst(decode_valid(*machine.decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    const auto scalar = machine.run(*inst, true, mode, exec, dst);
    const auto simd = machine.run(*inst, false, mode, exec, dst);
    for (uint32_t lane = 0; lane < machine.wave->wf_size(); ++lane) {
      for (uint32_t half = 0; half < 2; ++half) {
        // As in the existing FMA SIMD contract, NaN payload choice can differ.
        // Both paths must produce a quiet NaN; non-NaN results must agree exactly.
        if ((exec & (uint64_t{1} << lane)) && operation == 2 && is_nan(scalar[lane][half]) &&
            is_nan(simd[lane][half])) {
          EXPECT_NE(scalar[lane][half] & 0x00400000, 0u);
          EXPECT_NE(simd[lane][half] & 0x00400000, 0u);
        } else {
          ASSERT_EQ(simd[lane][half], scalar[lane][half]) << "lane=" << lane << " half=" << half;
        }
        if (dst == 6 && !(exec & (uint64_t{1} << lane))) {
          ASSERT_EQ(simd[lane][half], kSentinel);
        }
      }
    }
  }
};

TEST_P(PackedF32Selection, AllSelectionsPreservePolicies) {
  HostState saved;
  Machine machine(std::get<0>(GetParam()));
  const unsigned selections = std::get<1>(GetParam()) == 2 ? 8 : 4;
  for (uint8_t lo = 0; lo < selections; ++lo)
    for (uint8_t hi = 0; hi < selections; ++hi)
      for (uint32_t round = 0; round < 4; ++round)
        for (uint32_t denorm = 0; denorm < 4; ++denorm)
          for (uint8_t clamp : {0, 1})
            for (uint32_t dx10_clamp : {0u, 0x100u})
              compare(machine, lo, hi, 5, 2, 0xc0 | round | (denorm << 4) | dx10_clamp, clamp, 6,
                      SourceKind::Vgpr, 0xd6a53cf0a5c3697bULL);
}

TEST_P(PackedF32Selection, ScalarSourcesAliasingAndEmptyExec) {
  HostState saved;
  Machine machine(std::get<0>(GetParam()));
  // Only CDNA5 admits a literal source in this packed-F32 encoding.
  for (SourceKind source :
       {SourceKind::Vgpr, SourceKind::Sgpr, SourceKind::InlineFloat, SourceKind::Literal}) {
    if (source == SourceKind::Literal && std::get<0>(GetParam()) != ROCJITSU_CODE_ARCH_CDNA5)
      continue;
    // Packed FP32 register pairs must start at even VGPR addresses.
    for (uint8_t dst : {0, 2, 4, 6})
      for (uint8_t neg = 0; neg < 8; ++neg)
        for (uint64_t exec : {uint64_t{0}, uint64_t{0xaaaaaaaa55555555}, ~uint64_t{0}})
          compare(machine, 3, 2, neg, neg ^ 7, 0xf0, 0, dst, source, exec);
  }
}

TEST_P(PackedF32Selection, RestoresHostEnvironment) {
  HostState saved;
  Machine machine(std::get<0>(GetParam()));
  for (int rounding : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(rounding), 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(_mm_getcsr() | (1u << 6) | (1u << 15));
    const uint32_t mxcsr = _mm_getcsr();
#endif
    const int exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    compare(machine, 0, 2, 0, 0, 0xf1, 0, 6, SourceKind::Vgpr, ~uint64_t{0});
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
#if defined(__x86_64__) || defined(__i386__)
    EXPECT_EQ(_mm_getcsr(), mxcsr);
#endif
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cdna, PackedF32Selection,
    testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
                                     ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5),
                     testing::Values(0u, 1u, 2u)));
} // namespace
