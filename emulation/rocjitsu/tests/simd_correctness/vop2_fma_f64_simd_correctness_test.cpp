// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop2_fma_f64_simd_correctness_test.cpp
/// @brief CDNA4 V_FMAC_F64 decode, split-register, partial-EXEC, and MODE-aware
/// SIMD regressions.

#include "decode_test_util.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t kWaveSize = 64;
constexpr uint32_t kSgprsPerWave = 106;
constexpr uint32_t kVgprsPerWave = 256;

constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3fu) << 25) | ((vdst & 0xffu) << 17) | ((vsrc1 & 0xffu) << 9) |
         (src0 & 0x1ffu);
}

class ForceScalarGuard {
public:
  ForceScalarGuard() : original_(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(original_); }

private:
  bool original_;
};

class Vop2FmaF64Fixture {
public:
  amdgpu::GpuMemory gpu_memory{"vop2_fma_f64_simd_memory"};
  amdgpu::L2Cache l2{"vop2_fma_f64_simd_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Vop2FmaF64Fixture() {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_CDNA4;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = kSgprsPerWave;
    config.vgprs_per_wf = kVgprsPerWave;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("vop2_fma_f64_simd_cu", config, &gpu_memory, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, kSgprsPerWave, kVgprsPerWave);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t value) {
    const uint32_t base = wf->vgpr_alloc().base;
    cu->write_vgpr(base + reg, lane, static_cast<uint32_t>(value));
    cu->write_vgpr(base + reg + 1, lane, static_cast<uint32_t>(value >> 32));
  }

  uint64_t read64(uint32_t reg, uint32_t lane) const {
    const uint32_t base = wf->vgpr_alloc().base;
    return static_cast<uint64_t>(cu->read_vgpr(base + reg, lane)) |
           (static_cast<uint64_t>(cu->read_vgpr(base + reg + 1, lane)) << 32);
  }
};

struct RunResult {
  std::array<uint64_t, kWaveSize> output{};
  std::array<uint64_t, kWaveSize> accumulator{};
};

using RawInputs = std::array<std::array<uint64_t, 3>, kWaveSize>;

RunResult run_fmac(bool force_scalar, uint64_t exec, uint32_t mode, bool literal_src0 = false,
                   const RawInputs *raw_inputs = nullptr) {
  util::set_force_scalar_for_testing(force_scalar);
  Vop2FmaF64Fixture fixture;
  EXPECT_NE(fixture.cu, nullptr);
  EXPECT_NE(fixture.wf, nullptr);
  fixture.wf->set_exec(exec);
  fixture.wf->set_mode_raw(mode);

  RunResult result;
  constexpr uint64_t kLiteralBits = uint64_t{0x3df00000u} << 32;
  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    const double source0 = 1.0 + static_cast<double>(lane & 7u) * 0.125;
    const double source1 = literal_src0 ? 3.0 : 0x1p-53;
    const double accumulator =
        literal_src0 ? std::bit_cast<double>(kLiteralBits) : 1.0 + static_cast<double>(lane) * 0.25;
    const uint64_t source0_bits =
        raw_inputs ? (*raw_inputs)[lane][0] : std::bit_cast<uint64_t>(source0);
    const uint64_t source1_bits =
        raw_inputs ? (*raw_inputs)[lane][1] : std::bit_cast<uint64_t>(source1);
    fixture.write64(0, lane, source0_bits);
    fixture.write64(2, lane, source1_bits);
    result.accumulator[lane] =
        raw_inputs ? (*raw_inputs)[lane][2] : std::bit_cast<uint64_t>(accumulator);
    fixture.write64(4, lane, result.accumulator[lane]);
  }

  const uint32_t src0 = literal_src0 ? 255u : 256u;
  const uint32_t encoded = vop2_encode(/*opcode=*/4, /*vdst=*/4, /*vsrc1=*/2, src0);
  const uint32_t literal_high = static_cast<uint32_t>(kLiteralBits >> 32);
  uint32_t words[4] = {encoded, literal_src0 ? literal_high : 0u, 0u, 0u};
  std::unique_ptr<Instruction> instruction(decode_valid(*fixture.decoder, words));
  EXPECT_NE(instruction, nullptr);
  fixture.cu->execute_instruction(instruction.get(), *fixture.wf);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    result.output[lane] = fixture.read64(4, lane);
  return result;
}

TEST(Vop2FmaF64SimdCorrectness, InlineLiteralUsesEncodedHighWord) {
  ForceScalarGuard guard;
  const RunResult scalar = run_fmac(true, ~uint64_t{0}, /*mode=*/3u << 6, true);
  EXPECT_EQ(scalar.output[0], 0x3e10000000000000ULL);
  if constexpr (util::has_stdx_simd_64bit_lanes) {
    const RunResult simd = run_fmac(false, ~uint64_t{0}, /*mode=*/3u << 6, true);
    EXPECT_EQ(simd.output, scalar.output);
  }
}

TEST(Vop2FmaF64SimdCorrectness, PartialExecPreservesBothInactiveVgprWords) {
  ForceScalarGuard guard;
  constexpr uint64_t kExec = 0xa5a5f0f012348001ULL;
  const RunResult scalar = run_fmac(true, kExec, /*mode=*/3u << 6);
  for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
    if ((kExec & (uint64_t{1} << lane)) == 0) {
      EXPECT_EQ(scalar.output[lane], scalar.accumulator[lane]) << "scalar, inactive lane " << lane;
    }
  }
  if constexpr (util::has_stdx_simd_64bit_lanes) {
    const RunResult simd = run_fmac(false, kExec, /*mode=*/3u << 6);
    EXPECT_EQ(simd.output, scalar.output);
    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
      if ((kExec & (uint64_t{1} << lane)) == 0) {
        EXPECT_EQ(simd.output[lane], simd.accumulator[lane]) << "simd, inactive lane " << lane;
      }
    }
  }
}

TEST(Vop2FmaF64SimdCorrectness, AllRoundAndDenormModesMatchScalar) {
  if constexpr (!util::has_stdx_simd_64bit_lanes)
    GTEST_SKIP() << "64-bit-lane SIMD path unavailable; both runs would be the scalar one";
  ForceScalarGuard guard;
  for (uint32_t round = 0; round < 4; ++round) {
    for (uint32_t denorm = 0; denorm < 4; ++denorm) {
      const uint32_t mode = (round << 2) | (denorm << 6);
      const RunResult scalar = run_fmac(true, ~uint64_t{0}, mode);
      const RunResult simd = run_fmac(false, ~uint64_t{0}, mode);
      EXPECT_EQ(simd.output, scalar.output) << "round=" << round << " denorm=" << denorm;
    }
  }
}

TEST(Vop2FmaF64SimdCorrectness, SpecialValuesAndDenormBoundariesMatchScalarExactly) {
  if constexpr (!util::has_stdx_simd_64bit_lanes)
    GTEST_SKIP() << "64-bit-lane SIMD path unavailable; both runs would be the scalar one";
  ForceScalarGuard guard;

  constexpr uint64_t kPositiveZero = 0x0000'0000'0000'0000ULL;
  constexpr uint64_t kNegativeZero = 0x8000'0000'0000'0000ULL;
  constexpr uint64_t kOne = 0x3FF0'0000'0000'0000ULL;
  constexpr uint64_t kTwo = 0x4000'0000'0000'0000ULL;
  constexpr uint64_t kHalf = 0x3FE0'0000'0000'0000ULL;
  constexpr uint64_t kMinSubnormal = 0x0000'0000'0000'0001ULL;
  constexpr uint64_t kNegativeMinSubnormal = 0x8000'0000'0000'0001ULL;
  constexpr uint64_t kMaxSubnormal = 0x000F'FFFF'FFFF'FFFFULL;
  constexpr uint64_t kMinNormal = 0x0010'0000'0000'0000ULL;
  constexpr uint64_t kMaxFinite = 0x7FEF'FFFF'FFFF'FFFFULL;
  constexpr uint64_t kInfinity = 0x7FF0'0000'0000'0000ULL;
  constexpr uint64_t kQuietNan = 0x7FF8'0000'0000'0001ULL;
  constexpr uint64_t kSignalingNan = 0x7FF0'0000'0000'0001ULL;
  constexpr uint64_t kNegativeOneAndHalf = 0xBFF8'0000'0000'0000ULL;

  constexpr std::array<std::array<uint64_t, 3>, 12> cases = {{
      {kMinSubnormal, kOne, kPositiveZero},
      {kNegativeMinSubnormal, kOne, kPositiveZero},
      {kMaxSubnormal, kOne, kPositiveZero},
      {kMinNormal, kHalf, kPositiveZero},
      {kQuietNan, kOne, kPositiveZero},
      {kOne, kSignalingNan, kPositiveZero},
      {kInfinity, kPositiveZero, kPositiveZero},
      {kNegativeZero, kOne, kPositiveZero},
      {kMaxFinite, kTwo, kPositiveZero},
      {kNegativeOneAndHalf, kTwo, kOne},
      {kOne, kOne, kNegativeZero},
      {kMinNormal, kOne, kMinNormal | (uint64_t{1} << 63)},
  }};
  RawInputs inputs{};
  for (uint32_t lane = 0; lane < kWaveSize; ++lane)
    inputs[lane] = cases[lane % cases.size()];

  for (uint32_t round = 0; round < 4; ++round) {
    for (uint32_t denorm = 0; denorm < 4; ++denorm) {
      const uint32_t mode = (round << 2) | (denorm << 6);
      const RunResult scalar = run_fmac(true, ~uint64_t{0}, mode, false, &inputs);
      const RunResult simd = run_fmac(false, ~uint64_t{0}, mode, false, &inputs);
      EXPECT_EQ(simd.output, scalar.output) << "round=" << round << " denorm=" << denorm;
    }
  }
}

/// @brief Both paths against the architecturally correct result, not against
///        each other.
///
/// @details The tests above compare the scalar and SIMD paths. That catches a
/// divergence but not a shared mistake, and it names whichever path it is given
/// as the reference: when MODE.FP_ROUND stopped reaching the arithmetic, the
/// scalar path returned the round-to-nearest answer under every mode and the
/// failure read as "the two disagree" rather than "one is wrong".
///
/// These expectations are computed from the ISA definition rather than from the
/// model. Both operands are chosen so the result lands on a rounding boundary,
/// where the mode is the only thing that decides the answer:
///
///   a) 1.0 * 2^-53 + 1.0 is an exact tie, half an ulp above 1.0. Nearest
///      rounds to even and keeps 1.0; +inf must step up one ulp; -inf and zero
///      keep 1.0.
///   b) 1.125 * 2^-53 + 1.25 lands 0.5625 ulp above 1.25. Nearest is past the
///      halfway point and steps up, as does +inf; -inf and zero truncate.
///   c) the negative mirror of (b), 0.5625 ulp below -1.25. Nearest and -inf
///      take the more-negative neighbor; +inf and zero truncate toward -1.25.
///
/// (c) is what separates the four modes from each other. Both positive rows
/// give -inf and toward-zero the same answer, so on their own they would still
/// pass with FE_DOWNWARD and FE_TOWARDZERO transposed. Adding a negative row
/// splits that pair -- and splits nearest from +inf -- so each of the four
/// MODE.FP_ROUND values is pinned to exactly one host mode.
///
/// Because the reference is the ISA and not the other path, the scalar half
/// stands on its own and runs everywhere -- it is the half that caught this bug.
/// The SIMD half is gated on `has_stdx_simd_64bit_lanes`, not `has_stdx_simd`:
/// where the 64-bit-lane probes are compiled out, `run_fmac(false, ...)` falls
/// through to the scalar implementation, and asserting on it would report the
/// scalar path a second time as SIMD coverage.
TEST(Vop2FmaF64SimdCorrectness, DirectedRoundingMatchesTheIsaNotTheOtherPath) {
  ForceScalarGuard guard;

  constexpr uint64_t kOne = 0x3FF0000000000000ULL;
  constexpr uint64_t kOneAndAnEighth = 0x3FF2000000000000ULL;
  constexpr uint64_t kOneAndAQuarter = 0x3FF4000000000000ULL;
  constexpr uint64_t kTwoPowMinus53 = 0x3CA0000000000000ULL;
  constexpr uint64_t kNegativeOneAndAnEighth = 0xBFF2000000000000ULL;
  constexpr uint64_t kNegativeOneAndAQuarter = 0xBFF4000000000000ULL;

  struct Case {
    const char *what;
    uint64_t src0;
    uint64_t src1;
    uint64_t accumulator;
    // Indexed by MODE.FP_ROUND: 0 nearest-even, 1 +inf, 2 -inf, 3 zero.
    std::array<uint64_t, 4> expected;
  };

  const Case cases[] = {
      {"exact tie half an ulp above 1.0", kOne, kTwoPowMinus53, kOne, {kOne, kOne + 1, kOne, kOne}},
      {"0.5625 ulp above 1.25",
       kOneAndAnEighth,
       kTwoPowMinus53,
       kOneAndAQuarter,
       {kOneAndAQuarter + 1, kOneAndAQuarter + 1, kOneAndAQuarter, kOneAndAQuarter}},
      // Bits are sign-magnitude, so +1 here is the neighbor further from zero.
      {"0.5625 ulp below -1.25",
       kNegativeOneAndAnEighth,
       kTwoPowMinus53,
       kNegativeOneAndAQuarter,
       {kNegativeOneAndAQuarter + 1, kNegativeOneAndAQuarter, kNegativeOneAndAQuarter + 1,
        kNegativeOneAndAQuarter}},
  };

  for (const Case &test_case : cases) {
    RawInputs inputs{};
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
      inputs[lane] = {test_case.src0, test_case.src1, test_case.accumulator};

    for (uint32_t round = 0; round < 4; ++round) {
      // Denorm mode is irrelevant here; nothing in these cases is subnormal.
      const uint32_t mode = (round << 2) | (3u << 6);
      const uint64_t want = test_case.expected[round];
      const RunResult scalar = run_fmac(true, ~uint64_t{0}, mode, false, &inputs);
      for (uint32_t lane = 0; lane < kWaveSize; ++lane)
        EXPECT_EQ(scalar.output[lane], want)
            << "scalar, " << test_case.what << ", round=" << round << ", lane=" << lane;

      if constexpr (util::has_stdx_simd_64bit_lanes) {
        const RunResult simd = run_fmac(false, ~uint64_t{0}, mode, false, &inputs);
        for (uint32_t lane = 0; lane < kWaveSize; ++lane)
          EXPECT_EQ(simd.output[lane], want)
              << "simd, " << test_case.what << ", round=" << round << ", lane=" << lane;
      }
    }
  }
}

} // namespace
