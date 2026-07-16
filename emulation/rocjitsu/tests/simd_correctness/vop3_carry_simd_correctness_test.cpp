// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_carry_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs scalar body) for the
/// VOP3 sdst-enc carry-bearing ops:
///   no-carry-in (CDNA4): v_add_co_u32, v_sub_co_u32, v_subrev_co_u32
///   src2-carry-in (CDNA4): v_addc_co_u32, v_subb_co_u32, v_subbrev_co_u32
///   src2-carry-in (RDNA3, sdst pre-bound to VCC by decoder):
///     v_add_co_ci_u32, v_sub_co_ci_u32, v_subrev_co_ci_u32
/// All six cin-form scalar bodies pull per-lane carry-in from
/// `inst.src2.read_scalar64(wf)` (SGPR-pair) and write co into
/// `inst.sdst.write_scalar64`, with inactive lanes zeroed in the
/// incoming VCC. Each (case, vcc_in, cin) runs TWICE in the same process -- once
/// forcing the scalar body, once the SIMD fast path, with identical inputs --
/// and the scalar-vs-SIMD equivalence on BOTH the destination VGPR AND the full
/// 64-bit SGPR-pair carry result is asserted with EXPECT_EQ
/// (util::set_force_scalar_for_testing flips the gate in-process), sweeping
/// {full, partial} EXEC × {VCC seeds} × {cin patterns}. In-process inactive dst
/// lanes must keep the sentinel. Inputs deliberately seed the 32-bit
/// carry/borrow boundary on the low lanes.

#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
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
#include <random>
#include <string>

namespace {

using namespace rocjitsu;

constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t DST_SENTINEL = 0xCAFEF00Du;

// CDNA4 / RDNA3 Vop3SdstEnc layout (identical bit fields, only the
// `encoding` marker differs: CDNA4 = 0x34, RDNA3 = 0x35):
//   word0: vdst[7:0] | sdst[14:8] | clamp[15] | op[25:16] | encoding[31:26]
//   word1: src0[8:0] | src1[17:9] | src2[26:18] | omod[28:27] | neg[31:29]
constexpr void vop3_sdstenc_encode(uint32_t op, uint32_t vdst, uint32_t sdst, uint32_t src0,
                                   uint32_t src1, uint32_t src2, uint32_t encoding_marker,
                                   uint32_t words[2]) {
  words[0] = (vdst & 0xFFu) | ((sdst & 0x7Fu) << 8) | ((op & 0x3FFu) << 16) |
             ((encoding_marker & 0x3Fu) << 26);
  words[1] = (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18);
}

constexpr uint32_t kCdna4Encoding = 0x34u;
constexpr uint32_t kRdna3Encoding = 0x35u;

const std::array<std::pair<uint32_t, uint32_t>, 14> kEdgePairs = {{
    {0xFFFFFFFFu, 0x00000001u},
    {0xFFFFFFFFu, 0x00000000u},
    {0x00000000u, 0x00000000u},
    {0x00000001u, 0x00000000u},
    {0x00000000u, 0x00000001u},
    {0x80000000u, 0x80000000u},
    {0x7FFFFFFFu, 0x00000001u},
    {0xFFFFFFFFu, 0xFFFFFFFFu},
    {0x00000000u, 0xFFFFFFFFu},
    {0xFFFFFFFEu, 0x00000001u},
    {0x12345678u, 0x12345678u},
    {0x12345679u, 0x12345678u},
    {0xAAAAAAAAu, 0x55555555u},
    {0x00000002u, 0x00000003u},
}};

const uint64_t kVccPatterns[] = {
    0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xAAAAAAAAAAAAAAAAULL,
    0x5555555555555555ULL, 0x0123456789ABCDEFULL,
};

// The cin word comes from an SGPR-pair (src2), which can hold *any* 64-bit
// pattern — so sweep cin independently from VCC. Same set as VCC; the
// orthogonal sweep catches cases where the glue accidentally reads VCC
// instead of src2 (or vice-versa).
const uint64_t kCinPatterns[] = {
    0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xAAAAAAAAAAAAAAAAULL,
    0x5555555555555555ULL, 0xCAFEBABEDEADBEEFULL,
};

template <int WaveSize> struct WaveFixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  WaveFixture(const char *mem_label, const char *l2_label, const char *cu_label,
              rj_code_arch_t arch)
      : gpu_mem(mem_label), l2(l2_label) {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create(cu_label, cfg, &gpu_mem, &l2);
    decoder = Decoder::create(arch);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void seed_inputs(uint64_t seed, uint64_t exec, uint64_t vcc_in, uint64_t sdst_in,
                   uint32_t sdst_sgpr, uint64_t cin_word, uint32_t cin_sgpr_pair) {
    std::mt19937_64 rng(seed);
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WaveSize; ++lane) {
      uint32_t r0, r1;
      if (lane < kEdgePairs.size()) {
        r0 = kEdgePairs[lane].first;
        r1 = kEdgePairs[lane].second;
      } else {
        r0 = static_cast<uint32_t>(rng());
        r1 = static_cast<uint32_t>(rng());
      }
      cu->write_vgpr(vbase + 0, lane, r0);
      cu->write_vgpr(vbase + 1, lane, r1);
      cu->write_vgpr(vbase + 2, lane, DST_SENTINEL);
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc_in);
    cu->write_sgpr(sdst_sgpr + 0, static_cast<uint32_t>(sdst_in));
    cu->write_sgpr(sdst_sgpr + 1, static_cast<uint32_t>(sdst_in >> 32));
    if (cin_sgpr_pair != 0u) {
      cu->write_sgpr(cin_sgpr_pair + 0, static_cast<uint32_t>(cin_word));
      cu->write_sgpr(cin_sgpr_pair + 1, static_cast<uint32_t>(cin_word >> 32));
    }
  }

  struct Result {
    std::array<uint32_t, WaveSize> dst{};
    uint64_t sdst = 0;
  };

  Result run(Instruction *inst, uint64_t seed, uint64_t exec, uint64_t vcc_in, uint64_t sdst_in,
             uint32_t sdst_sgpr, uint64_t cin_word, uint32_t cin_sgpr_pair) {
    seed_inputs(seed, exec, vcc_in, sdst_in, sdst_sgpr, cin_word, cin_sgpr_pair);
    cu->execute_instruction(inst, *wf);
    Result res;
    uint32_t vbase = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WaveSize; ++lane)
      res.dst[lane] = cu->read_vgpr(vbase + 2, lane);
    uint64_t lo = cu->read_sgpr(sdst_sgpr + 0);
    uint64_t hi = cu->read_sgpr(sdst_sgpr + 1);
    res.sdst = (hi << 32) | lo;
    return res;
  }
};

struct CarryCase {
  const char *label;
  uint32_t opcode;
  bool has_cin; ///< true for addc/subb/subbrev / add_co_ci / sub_co_ci / subrev_co_ci
};

const CarryCase kCdna4Cases[] = {
    // no-cin
    {"v_add_co_u32_vop3", 281, false},
    {"v_sub_co_u32_vop3", 282, false},
    {"v_subrev_co_u32_vop3", 283, false},
    // src2-cin
    {"v_addc_co_u32_vop3", 284, true},
    {"v_subb_co_u32_vop3", 285, true},
    {"v_subbrev_co_u32_vop3", 286, true},
};

const CarryCase kRdnaCases[] = {
    {"v_add_co_ci_u32_vop3", 288, true},
    {"v_sub_co_ci_u32_vop3", 289, true},
    {"v_subrev_co_ci_u32_vop3", 290, true},
};

// Restores the process force-scalar gate on scope exit so flipping it for an
// in-process A/B comparison cannot leak into later tests in the same process.
struct ForceScalarGuard {
  bool orig;
  ForceScalarGuard() : orig(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(orig); }
};

template <int WaveSize>
void check_case(const char *mem_label, const char *l2_label, const char *cu_label,
                rj_code_arch_t arch, const CarryCase &c, uint32_t encoding_marker, uint64_t exec) {
  ForceScalarGuard gate_guard;

  using Result = typename WaveFixture<WaveSize>::Result;

  constexpr uint64_t SEED = 0xC0FFEE'1234'5678ULL;
  // Seed the SGPR-pair sdst with a recognisable pattern so any "didn't write"
  // bug surfaces as a divergent (incoming vs scalar) result.
  constexpr uint64_t SDST_SEED = 0xDEADBEEFCAFEF00DULL;

  // Runs one (case, vcc_in, cin_word) in the requested execute mode (fresh
  // WaveFixture + decode per run isolates VGPR/SGPR/VCC state).
  auto run_mode = [&](bool force_scalar, uint64_t vcc_in, uint64_t cin_word) -> Result {
    util::set_force_scalar_for_testing(force_scalar);
    WaveFixture<WaveSize> fx(mem_label, l2_label, cu_label, arch);
    EXPECT_NE(fx.cu, nullptr);
    EXPECT_NE(fx.wf, nullptr);
    uint32_t sb = fx.wf->sgpr_alloc().base;
    // SGPR-pair carry-out target; must be an even-aligned SREG index.
    EXPECT_EQ(sb % 2u, 0u) << c.label << ": sgpr_alloc base not pair-aligned";
    // Second pair (sb+2) holds the cin word for cin-form ops; for no-cin
    // ops it is unused and the src2 field encodes 0 (the body ignores it).
    uint32_t cin_pair = c.has_cin ? (sb + 2u) : 0u;
    uint32_t src2_field = c.has_cin ? cin_pair : 0u;
    uint32_t words[2] = {0u, 0u};
    vop3_sdstenc_encode(c.opcode, /*vdst=*/2, /*sdst=*/sb,
                        /*src0=*/256, /*src1=*/257, /*src2=*/src2_field, encoding_marker, words);
    Instruction *inst = fx.decoder->decode(words);
    EXPECT_NE(inst, nullptr) << c.label << ": decode failed";
    auto out = fx.run(inst, SEED, exec, vcc_in, SDST_SEED, sb, cin_word, cin_pair);
    delete inst;
    return out;
  };

  for (uint64_t vcc_in : kVccPatterns) {
    // No-cin ops still sweep VCC (their glue seeds the sdst inactive bits
    // from VCC). Cin ops also sweep cin word from src2 SGPR-pair.
    const uint64_t *cin_set = c.has_cin ? kCinPatterns : kVccPatterns;
    const size_t cin_n = c.has_cin ? std::size(kCinPatterns) : 1u;
    for (size_t i = 0; i < cin_n; ++i) {
      uint64_t cin_word = c.has_cin ? cin_set[i] : 0ULL;
      const auto scalar_out = run_mode(/*force_scalar=*/true, vcc_in, cin_word);
      const auto simd_out = run_mode(/*force_scalar=*/false, vcc_in, cin_word);

      // Core A/B equivalence on BOTH the dst words and the full 64-bit sdst.
      EXPECT_EQ(scalar_out.dst, simd_out.dst)
          << c.label << " vcc=0x" << std::hex << vcc_in << " cin=0x" << cin_word
          << ": SIMD dst diverged from scalar body";
      EXPECT_EQ(scalar_out.sdst, simd_out.sdst)
          << c.label << " vcc=0x" << std::hex << vcc_in << " cin=0x" << cin_word
          << ": SIMD sdst diverged from scalar body";

      for (uint32_t lane = 0; lane < WaveSize; ++lane) {
        const bool active = (exec >> lane) & 1ULL;
        if (!active) {
          EXPECT_EQ(simd_out.dst[lane], DST_SENTINEL)
              << c.label << ": clobbered inactive dst lane " << lane;
          EXPECT_EQ(scalar_out.dst[lane], DST_SENTINEL)
              << c.label << ": clobbered inactive dst lane " << lane;
        }
      }
    }
  }
}

void run_cdna4(uint64_t exec) {
  for (const auto &c : kCdna4Cases)
    check_case<64>("vop3_carry_simd_mem", "vop3_carry_simd_l2", "cu_vop3_carry_simd",
                   ROCJITSU_CODE_ARCH_CDNA4, c, kCdna4Encoding, exec);
}

void run_rdna(uint64_t exec) {
  for (const auto &c : kRdnaCases)
    check_case<32>("vop3_carry_simd_rdna_mem", "vop3_carry_simd_rdna_l2", "cu_vop3_carry_simd_rdna",
                   ROCJITSU_CODE_ARCH_RDNA3, c, kRdna3Encoding, exec);
}

TEST(Vop3CarrySimdCorrectness, Cdna4FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  run_cdna4(/*exec=*/~0ULL);
}

TEST(Vop3CarrySimdCorrectness, Cdna4PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  run_cdna4(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

TEST(Vop3CarrySimdCorrectness, RdnaFullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // Wave32 — exec only meaningful in low 32 bits.
  run_rdna(/*exec=*/0x00000000'FFFFFFFFULL);
}

TEST(Vop3CarrySimdCorrectness, RdnaPartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  run_rdna(/*exec=*/0x00000000'A5A5F0F1ULL);
}

} // namespace
