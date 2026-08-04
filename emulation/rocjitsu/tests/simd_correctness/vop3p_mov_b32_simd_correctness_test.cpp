// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3p_mov_b32_simd_correctness_test.cpp
/// @brief Golden selector check for v_pk_mov_b32_vop3p on CDNA targets.
///
/// Hardware-observed behavior for the assembler's default op_sel_hi value:
/// op_sel[0] selects the low output dword from src0.{lo,hi}, and op_sel[1]
/// selects the high output dword from src1.{lo,hi}. Each case runs twice in
/// the same process -- once forcing the scalar body, once allowing the SIMD
/// fast path -- and both paths must match the same golden 64-bit result. A
/// mixed SGPR-pair/VGPR-pair case covers the legal scalar source encoding.
/// In-process inactive lanes must keep the sentinel.

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "util/simd.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t kDstVgpr = 8; // pair occupies kDstVgpr..kDstVgpr+1
constexpr uint32_t DST_SENTINEL = 0xCDCDCDCDu;
constexpr uint32_t kMixedSgprLo = 0x11111111u;
constexpr uint32_t kMixedSgprHi = 0x22222222u;
constexpr uint32_t kMixedVgprLo = 0x55555555u;
constexpr uint32_t kMixedVgprHi = 0x66666666u;

struct ArchCase {
  rj_code_arch_t arch;
  const char *name;
};

constexpr std::array<ArchCase, 3> kArchCases{{
    {ROCJITSU_CODE_ARCH_CDNA2, "cdna2"},
    {ROCJITSU_CODE_ARCH_CDNA3, "cdna3"},
    {ROCJITSU_CODE_ARCH_CDNA4, "cdna4"},
}};

constexpr void vop3p_encode(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1,
                            uint32_t op_sel, uint32_t words[2]) {
  // op_sel_hi = 3 is what LLVM emits for v_pk_mov_b32 op_sel:[x,y].
  // neg / neg_hi / clamp = 0.
  words[0] = (vdst & 0xFFu) | ((op_sel & 0x3u) << 11) | ((op & 0x7Fu) << 16) | (0x1A7u << 23);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | (0x3u << 27);
}

const std::array<uint32_t, 14> kVals = {{
    0x00000000u,
    0xFFFFFFFFu,
    0x12345678u,
    0xDEADBEEFu,
    0xCAFEBABEu,
    0xA5A5A5A5u,
    0x5A5A5A5Au,
    0x80000000u,
    0x7FFFFFFFu,
    0xAAAAAAAAu,
    0x55555555u,
    0x00010001u,
    0xFEDCBA98u,
    0x13579BDFu,
}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  explicit Fixture(const ArchCase &arch)
      : gpu_mem(std::string("vop3p_mov_b32_mem_") + arch.name),
        l2(std::string("vop3p_mov_b32_l2_") + arch.name) {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch.arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create(std::string("cu_vop3p_mov_b32_") + arch.name, cfg,
                                         &gpu_mem, &l2);
    decoder = Decoder::create(arch.arch);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint32_t rot, uint64_t exec) {
    uint32_t sb = wf->sgpr_alloc().base;
    uint32_t vb = wf->vgpr_alloc().base;
    cu->write_sgpr(sb + 0, kMixedSgprLo);
    cu->write_sgpr(sb + 1, kMixedSgprHi);
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      // src0 pair at v0:v1, src1 pair at v2:v3.
      cu->write_vgpr(vb + 0, lane, kVals[lane % kVals.size()]);
      cu->write_vgpr(vb + 1, lane, kVals[(lane + 1) % kVals.size()]);
      cu->write_vgpr(vb + 2, lane, kVals[(lane + rot) % kVals.size()]);
      cu->write_vgpr(vb + 3, lane, kVals[(lane + rot + 1) % kVals.size()]);
      cu->write_vgpr(vb + 4, lane, kMixedVgprLo);
      cu->write_vgpr(vb + 5, lane, kMixedVgprHi);
      cu->write_vgpr(vb + kDstVgpr + 0, lane, DST_SENTINEL);
      cu->write_vgpr(vb + kDstVgpr + 1, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
  }

  std::array<uint64_t, WF_SIZE> run(Instruction *inst, uint32_t rot, uint64_t exec) {
    seed_inputs(rot, exec);
    cu->execute_instruction(inst, *wf);
    return read_output();
  }

  template <typename Inst>
  std::array<uint64_t, WF_SIZE> run_simd_probe_impl(Instruction *inst, const ArchCase &arch,
                                                    uint32_t rot, uint64_t exec, uint32_t op_sel) {
    auto *typed_inst = dynamic_cast<Inst *>(inst);
    EXPECT_NE(typed_inst, nullptr) << arch.name << " decoded unexpected instruction type";
    if (typed_inst == nullptr)
      return {};
    seed_inputs(rot, exec);
    EXPECT_TRUE(amdgpu::try_execute_vop3p_mov_b32_simd(*typed_inst, *wf))
        << arch.name << " v_pk_mov_b32_vop3p op_sel=" << op_sel
        << " did not take the SIMD fast path";
    return read_output();
  }

  std::array<uint64_t, WF_SIZE> run_simd_probe(Instruction *inst, const ArchCase &arch,
                                               uint32_t rot, uint64_t exec, uint32_t op_sel) {
    switch (arch.arch) {
    case ROCJITSU_CODE_ARCH_CDNA2:
      return run_simd_probe_impl<cdna2::VPkMovB32Vop3p>(inst, arch, rot, exec, op_sel);
    case ROCJITSU_CODE_ARCH_CDNA3:
      return run_simd_probe_impl<cdna3::VPkMovB32Vop3p>(inst, arch, rot, exec, op_sel);
    case ROCJITSU_CODE_ARCH_CDNA4:
      return run_simd_probe_impl<cdna4::VPkMovB32Vop3p>(inst, arch, rot, exec, op_sel);
    default:
      ADD_FAILURE() << "unsupported arch case " << arch.name;
      return {};
    }
  }

  std::array<uint64_t, WF_SIZE> read_output() {
    std::array<uint64_t, WF_SIZE> out{};
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint64_t lo = cu->read_vgpr(vb + kDstVgpr + 0, lane);
      uint64_t hi = cu->read_vgpr(vb + kDstVgpr + 1, lane);
      out[lane] = lo | (hi << 32);
    }
    return out;
  }
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

std::array<uint64_t, WF_SIZE> expected(uint32_t rot, uint64_t exec, uint32_t op_sel) {
  std::array<uint64_t, WF_SIZE> out{};
  const uint64_t sentinel = uint64_t{DST_SENTINEL} | (uint64_t{DST_SENTINEL} << 32);
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    const bool active = (exec >> lane) & 1ULL;
    if (!active) {
      out[lane] = sentinel;
      continue;
    }
    const uint32_t src0_lo = kVals[lane % kVals.size()];
    const uint32_t src0_hi = kVals[(lane + 1) % kVals.size()];
    const uint32_t src1_lo = kVals[(lane + rot) % kVals.size()];
    const uint32_t src1_hi = kVals[(lane + rot + 1) % kVals.size()];
    const uint32_t lo = (op_sel & 1u) ? src0_hi : src0_lo;
    const uint32_t hi = (op_sel & 2u) ? src1_hi : src1_lo;
    out[lane] = uint64_t{lo} | (uint64_t{hi} << 32);
  }
  return out;
}

void check(uint64_t exec) {
  ForceScalarGuard gate_guard;

  auto run_mode = [&](const ArchCase &arch, bool force_scalar, uint32_t rot,
                      uint32_t op_sel) -> std::array<uint64_t, WF_SIZE> {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx(arch);
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[2] = {0u, 0u};
    // src0 = VGPR 256 (pair v0:v1), src1 = VGPR 258 (pair v2:v3).
    vop3p_encode(/*op=*/51, kDstVgpr, /*src0=*/256, /*src1=*/258, op_sel, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << "v_pk_mov_b32_vop3p decode failed";
    auto out =
        force_scalar ? fx.run(inst, rot, exec) : fx.run_simd_probe(inst, arch, rot, exec, op_sel);
    delete inst;
    return out;
  };

  for (const auto &arch : kArchCases) {
    for (uint32_t rot = 0; rot < kVals.size(); ++rot) {
      for (uint32_t op_sel = 0; op_sel < 4; ++op_sel) {
        const auto golden = expected(rot, exec, op_sel);
        const auto scalar_out = run_mode(arch, /*force_scalar=*/true, rot, op_sel);
        const auto simd_out = run_mode(arch, /*force_scalar=*/false, rot, op_sel);

        EXPECT_EQ(scalar_out, golden)
            << arch.name << " v_pk_mov_b32_vop3p rot=" << rot << " op_sel=" << op_sel
            << ": scalar path diverged from hardware-observed semantics";
        EXPECT_EQ(simd_out, golden)
            << arch.name << " v_pk_mov_b32_vop3p rot=" << rot << " op_sel=" << op_sel
            << ": SIMD path diverged from hardware-observed semantics";

        for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
          const bool active = (exec >> lane) & 1ULL;
          if (!active) {
            const uint64_t sentinel = uint64_t{DST_SENTINEL} | (uint64_t{DST_SENTINEL} << 32);
            EXPECT_EQ(simd_out[lane], sentinel)
                << arch.name << " rot=" << rot << " op_sel=" << op_sel
                << ": clobbered inactive lane " << lane;
            EXPECT_EQ(scalar_out[lane], sentinel)
                << arch.name << " rot=" << rot << " op_sel=" << op_sel
                << ": clobbered inactive lane " << lane;
          }
        }
      }
    }
  }
}

TEST(Vop3pMovB32SimdCorrectness, FullExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check(/*exec=*/~0ULL);
}

TEST(Vop3pMovB32SimdCorrectness, PartialExec) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

TEST(Vop3pMovB32SimdCorrectness, ProductionSimdDispatchGolden) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }

  ForceScalarGuard gate_guard;
  util::set_force_scalar_for_testing(false);
  constexpr uint32_t kRot = 5;
  constexpr uint32_t kOpSel = 3;
  constexpr uint64_t kExec = 0xA5A5'F0F0'1234'8001ULL;
  const auto golden = expected(kRot, kExec, kOpSel);

  for (const auto &arch : kArchCases) {
    Fixture fx(arch);
    ASSERT_NE(fx.cu, nullptr);
    ASSERT_NE(fx.wf, nullptr);
    uint32_t words[2] = {0u, 0u};
    vop3p_encode(/*op=*/51, kDstVgpr, /*src0=*/256, /*src1=*/258, kOpSel, words);
    Instruction *inst = fx.decoder->decode(words);
    ASSERT_NE(inst, nullptr) << arch.name << " v_pk_mov_b32_vop3p decode failed";
    const auto out = fx.run(inst, kRot, kExec);
    delete inst;

    EXPECT_EQ(out, golden) << arch.name << ": production SIMD dispatch differed from golden";
  }
}

TEST(Vop3pMovB32SimdCorrectness, MixedSgprPairAndVgprPairGolden) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }

  ForceScalarGuard gate_guard;
  struct MixedSourceCase {
    const char *name;
    uint32_t src0;
    uint32_t src1;
    uint32_t op_sel;
    uint64_t golden;
  };
  constexpr std::array<MixedSourceCase, 2> kCases{{
      {"sgpr-vgpr", /*src0=*/0, /*src1=*/256 + 4, /*op_sel=*/1,
       uint64_t{kMixedSgprHi} | (uint64_t{kMixedVgprLo} << 32)},
      {"vgpr-sgpr", /*src0=*/256 + 4, /*src1=*/0, /*op_sel=*/2,
       uint64_t{kMixedVgprLo} | (uint64_t{kMixedSgprHi} << 32)},
  }};

  auto run_mode = [&](const ArchCase &arch, const MixedSourceCase &test_case, bool force_scalar) {
    util::set_force_scalar_for_testing(force_scalar);
    Fixture fx(arch);
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t words[2] = {0u, 0u};
    vop3p_encode(/*op=*/51, kDstVgpr, test_case.src0, test_case.src1, test_case.op_sel, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << arch.name << " " << test_case.name
                             << " v_pk_mov_b32_vop3p decode failed";
    auto out = force_scalar
                   ? fx.run(inst, /*rot=*/0, /*exec=*/~0ULL)
                   : fx.run_simd_probe(inst, arch, /*rot=*/0, /*exec=*/~0ULL, test_case.op_sel);
    delete inst;
    return out;
  };

  for (const auto &arch : kArchCases) {
    for (const auto &test_case : kCases) {
      const auto scalar_out = run_mode(arch, test_case, /*force_scalar=*/true);
      const auto simd_out = run_mode(arch, test_case, /*force_scalar=*/false);
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        EXPECT_EQ(scalar_out[lane], test_case.golden)
            << arch.name << " " << test_case.name
            << ": scalar mixed SGPR/VGPR result differs at lane " << lane;
        EXPECT_EQ(simd_out[lane], test_case.golden)
            << arch.name << " " << test_case.name
            << ": SIMD mixed SGPR/VGPR result differs at lane " << lane;
      }
    }
  }
}

} // namespace
