// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vopc_simd_correctness_test.cpp
/// @brief Bit-identity check (SIMD fast path vs forced-scalar body) for the VOPC
/// compares wired into SIMD_VOPC / SIMD_VOPC64: the f32/f16/f64 relations
/// (eq/ge/gt/le/lg/lt/neq/nge/ngt/nle/nlg/nlt/o/u/f/tru) and the
/// i32/u32/i16/u16/i64/u64 relations (eq/ge/gt/le/lt/ne/f/t). Each writes one bit
/// per active EXEC lane into VCC, preserving inactive bits. The test runs both
/// modes in-process via amdgpu::simd_force_scalar() and asserts the full 64-bit
/// VCC agrees under full and partial EXEC, with inactive-lane bits preserved.
/// The 64-bit relations exercise the split lo/hi VGPR-pair read path. Inputs seed
/// NaN/±Inf/±0/denorm (floats) and signed/extreme boundaries (ints); float
/// compares are bit-exact in both modes (no NaN-skip carve-out needed).

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
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
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;

// VOPC: encoding[31:25]=0x3E, op[24:17], vsrc1[16:9], src0[8:0].
constexpr uint32_t vopc_encode(uint32_t op, uint32_t src0, uint32_t vsrc1) {
  return (0x3Eu << 25) | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

enum class Kind { F32, F16, INT32, F64, INT64 };

bool is_64bit(Kind k) { return k == Kind::F64 || k == Kind::INT64; }

// 32-bit edge sets ----------------------------------------------------------
// f32: ±0, ±1, ±Inf, NaN, denorm, pi, max.
const std::array<uint32_t, 10> kF32 = {{0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u,
                                        0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x00000001u,
                                        0x40490FDBu, 0x7F7FFFFFu}};
// f16 in the low 16 bits; high bits garbage to confirm both modes ignore them.
const std::array<uint32_t, 10> kF16 = {{0xDEAD0000u, 0xDEAD8000u, 0xDEAD3C00u, 0xDEADBC00u,
                                        0xDEAD7C00u, 0xDEADFC00u, 0xDEAD7E00u, 0xDEAD0001u,
                                        0xDEAD4248u, 0xDEAD7BFFu}};
// integer edges (i32/u32 directly, low 16 for i16/u16).
const std::array<uint32_t, 10> kInt32 = {{0x00000000u, 0x00000001u, 0xFFFFFFFFu, 0x7FFFFFFFu,
                                          0x80000000u, 0x0000FFFFu, 0x00008000u, 0x00007FFFu,
                                          0x12345678u, 0xAAAA5555u}};

// 64-bit edge sets ----------------------------------------------------------
// f64: ±0, ±1, ±Inf, NaN, denorm, pi, max.
const std::array<uint64_t, 10> kF64 = {
    {0x0000000000000000ull, 0x8000000000000000ull, 0x3FF0000000000000ull, 0xBFF0000000000000ull,
     0x7FF0000000000000ull, 0xFFF0000000000000ull, 0x7FF8000000000000ull, 0x0000000000000001ull,
     0x400921FB54442D18ull, 0x7FEFFFFFFFFFFFFFull}};
// integer edges (i64/u64).
const std::array<uint64_t, 10> kInt64 = {
    {0x0000000000000000ull, 0x0000000000000001ull, 0xFFFFFFFFFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull,
     0x8000000000000000ull, 0x00000000FFFFFFFFull, 0x0000000080000000ull, 0x000000007FFFFFFFull,
     0x123456789ABCDEF0ull, 0xAAAAAAAA55555555ull}};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vopc_simd_mem"), l2("vopc_simd_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vopc_simd", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  void write64(uint32_t reg, uint32_t lane, uint64_t v) {
    cu->write_vgpr(reg, lane, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, lane, static_cast<uint32_t>(v >> 32));
  }

  // Seed src0 / vsrc1 with all ordered pairs of the kind's edge set across the
  // lanes (a deterministic 10x10 grid, exercising both operand orders). For
  // 32-bit kinds src0=v0, vsrc1=v1; for 64-bit kinds src0=v0:v1, vsrc1=v2:v3.
  void seed_inputs(Kind k, uint64_t exec, uint64_t vcc_in) {
    uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t i = lane % 10;
      uint32_t j = (lane / 10 + lane) % 10;
      if (is_64bit(k)) {
        const auto &e = (k == Kind::F64) ? kF64 : kInt64;
        write64(vb + 0, lane, e[i]);
        write64(vb + 2, lane, e[j]);
      } else {
        const auto &e = (k == Kind::F32) ? kF32 : (k == Kind::F16) ? kF16 : kInt32;
        cu->write_vgpr(vb + 0, lane, e[i]);
        cu->write_vgpr(vb + 1, lane, e[j]);
      }
    }
    wf->set_exec(exec);
    wf->set_vcc(vcc_in);
  }

  uint64_t run(Instruction *inst, bool force_scalar, Kind k, uint64_t exec, uint64_t vcc_in) {
    amdgpu::simd_force_scalar() = force_scalar;
    seed_inputs(k, exec, vcc_in);
    cu->execute_instruction(inst, *wf);
    amdgpu::simd_force_scalar() = false;
    return wf->vcc();
  }
};

struct VopcCase {
  uint32_t opcode;
  Kind kind;
};

// Build the full case list from the regular opcode layout. Float blocks (f16
// base 32, f32 base 64, f64 base 96): 16 relations at offsets 0..15. Integer
// blocks (i16 160, u16 168, i32 192, u32 200, i64 224, u64 232): 8 relations at
// offsets 0..7.
std::vector<VopcCase> all_cases() {
  std::vector<VopcCase> cs;
  for (auto [base, k] : std::initializer_list<std::pair<uint32_t, Kind>>{
           {32u, Kind::F16}, {64u, Kind::F32}, {96u, Kind::F64}})
    for (uint32_t off = 0; off < 16; ++off)
      cs.push_back({base + off, k});
  for (auto [base, k] : std::initializer_list<std::pair<uint32_t, Kind>>{{160u, Kind::INT32},
                                                                         {168u, Kind::INT32},
                                                                         {192u, Kind::INT32},
                                                                         {200u, Kind::INT32},
                                                                         {224u, Kind::INT64},
                                                                         {232u, Kind::INT64}})
    for (uint32_t off = 0; off < 8; ++off)
      cs.push_back({base + off, k});
  return cs;
}

void check_all(uint64_t exec) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const uint64_t kVcc[] = {0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0xA5A5A5A5A5A5A5A5ULL};
  for (const auto &c : all_cases()) {
    // 64-bit operands span a VGPR pair: vsrc1 = v2:v3, else vsrc1 = v1.
    uint32_t vsrc1 = is_64bit(c.kind) ? 2u : 1u;
    uint32_t enc = vopc_encode(c.opcode, /*src0=*/256, vsrc1);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = fx.decoder->decode(words);
    ASSERT_NE(inst, nullptr) << "VOPC opcode " << c.opcode << " decode failed";
    for (uint64_t vcc_in : kVcc) {
      uint64_t scalar = fx.run(inst, /*force_scalar=*/true, c.kind, exec, vcc_in);
      uint64_t simd = fx.run(inst, /*force_scalar=*/false, c.kind, exec, vcc_in);
      EXPECT_EQ(scalar, simd) << "VOPC opcode " << c.opcode << " vcc_in=0x" << std::hex << vcc_in
                              << ": VCC divergence scalar=0x" << scalar << " simd=0x" << simd;
      const uint64_t inactive = ~exec;
      EXPECT_EQ(simd & inactive, vcc_in & inactive)
          << "VOPC opcode " << c.opcode << ": SIMD altered an inactive-lane VCC bit";
    }
    delete inst;
  }
}

TEST(VopcSimdCorrectness, FullExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  check_all(/*exec=*/~0ULL);
}

TEST(VopcSimdCorrectness, PartialExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable — scalar fallback in use";
    return;
  }
  // Alternating/sparse pattern crossing the SIMD chunk boundaries (16-wide for
  // 32-bit lanes, 8-wide for 64-bit).
  check_all(/*exec=*/0xA5A5'F0F0'1234'8001ULL);
}

} // namespace
