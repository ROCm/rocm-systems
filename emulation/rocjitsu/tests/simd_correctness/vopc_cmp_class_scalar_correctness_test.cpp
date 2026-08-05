// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vopc_cmp_class_scalar_correctness_test.cpp
/// @brief Regression test for the scalar v_cmp_class_{f16,f32,f64} bodies.
/// v_cmp_class tests src0's IEEE float class against a 10-bit class mask in
/// vsrc1 and writes one VCC bit per lane (1 if src0's class is selected). A
/// prior codegen bug extracted the operand through a value-truncating round
/// trip (`bit_cast<float>(static_cast<uint32_t>(<value>))`) and always
/// classified in f32 precision, so e.g. the double 1.0 (0x3FF0000000000000)
/// was mangled to a tiny f32 denormal and misclassified.
///
/// Each row pairs a representative bit pattern with the single class bit it
/// must match. The op is run with mask = that bit (expect VCC set) and with
/// mask = the other nine bits (expect VCC clear), so a misclassification in
/// either direction is caught. f64 patterns all have a nonzero high word, so a
/// low-word-only read or f32-precision classify fails the +normal/-normal rows.

#include "rocjitsu/code/rj_code.h"
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

[[maybe_unused]] constexpr uint32_t WF_SIZE = 64;
constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t CLASS_ALL = 0x3FFu; // all 10 class bits

// VOPC: encoding[31:25]=0x3E, op[24:17], vsrc1[16:9], src0[8:0].
constexpr uint32_t vopc_encode(uint32_t op, uint32_t src0, uint32_t vsrc1) {
  return (0x3Eu << 25) | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

// The 10 IEEE float classes, low to high bit:
// 0x001 sNaN, 0x002 qNaN, 0x004 -Inf, 0x008 -normal, 0x010 -denormal,
// 0x020 -0, 0x040 +0, 0x080 +denormal, 0x100 +normal, 0x200 +Inf.
struct ClassRow {
  uint64_t bits; // src0 bit pattern (low 16/32 for f16/f32; full 64 for f64)
  uint32_t class_bit;
  const char *label;
};

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vopc_class_mem"), l2("vopc_class_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vopc_class", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
  }

  // src0 = v0[:v1], vsrc1 (mask) = v2. Returns VCC bit for lane 0.
  bool run(uint32_t op, uint64_t src_bits, uint32_t mask, bool is_f64) {
    uint32_t vb = wf->vgpr_alloc().base;
    wf->set_exec(~0ULL);
    wf->set_vcc(0);
    cu->write_vgpr(vb + 0, 0, static_cast<uint32_t>(src_bits));
    if (is_f64)
      cu->write_vgpr(vb + 1, 0, static_cast<uint32_t>(src_bits >> 32));
    cu->write_vgpr(vb + 2, 0, mask);
    uint32_t enc = vopc_encode(op, /*src0=*/256, /*vsrc1=*/2);
    uint32_t words[4] = {enc, 0u, 0u, 0u};
    Instruction *inst = decoder->decode(words);
    EXPECT_NE(inst, nullptr);
    cu->execute_instruction(inst, *wf);
    delete inst;
    return (wf->vcc() >> 0) & 1ULL;
  }
};

void check(uint32_t op, const std::array<ClassRow, 10> &rows, bool is_f64) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  for (const auto &r : rows) {
    // Matching its own class bit -> VCC set.
    EXPECT_TRUE(fx.run(op, r.bits, r.class_bit, is_f64))
        << r.label << ": expected match for class 0x" << std::hex << r.class_bit;
    // The complement (other nine bits) -> VCC clear (each value is one class).
    EXPECT_FALSE(fx.run(op, r.bits, CLASS_ALL & ~r.class_bit, is_f64))
        << r.label << ": unexpected match against other classes";
    // The full mask selects every class -> always set.
    EXPECT_TRUE(fx.run(op, r.bits, CLASS_ALL, is_f64)) << r.label << ": full mask should match";
  }
}

TEST(VopcCmpClassScalar, F32) {
  const std::array<ClassRow, 10> rows = {{
      {0x7F800001ull, 0x001, "sNaN"},
      {0x7FC00000ull, 0x002, "qNaN"},
      {0xFF800000ull, 0x004, "-Inf"},
      {0xBF800000ull, 0x008, "-normal"},
      {0x80000001ull, 0x010, "-denormal"},
      {0x80000000ull, 0x020, "-0"},
      {0x00000000ull, 0x040, "+0"},
      {0x00000001ull, 0x080, "+denormal"},
      {0x3F800000ull, 0x100, "+normal"},
      {0x7F800000ull, 0x200, "+Inf"},
  }};
  check(/*v_cmp_class_f32=*/16, rows, /*is_f64=*/false);
}

TEST(VopcCmpClassScalar, F16) {
  const std::array<ClassRow, 10> rows = {{
      {0x7C01ull, 0x001, "sNaN"},
      {0x7E00ull, 0x002, "qNaN"},
      {0xFC00ull, 0x004, "-Inf"},
      {0xBC00ull, 0x008, "-normal"},
      {0x8001ull, 0x010, "-denormal"},
      {0x8000ull, 0x020, "-0"},
      {0x0000ull, 0x040, "+0"},
      {0x0001ull, 0x080, "+denormal"},
      {0x3C00ull, 0x100, "+normal"},
      {0x7C00ull, 0x200, "+Inf"},
  }};
  check(/*v_cmp_class_f16=*/20, rows, /*is_f64=*/false);
}

TEST(VopcCmpClassScalar, F64) {
  const std::array<ClassRow, 10> rows = {{
      {0x7FF0000000000001ull, 0x001, "sNaN"},
      {0x7FF8000000000000ull, 0x002, "qNaN"},
      {0xFFF0000000000000ull, 0x004, "-Inf"},
      {0xBFF0000000000000ull, 0x008, "-normal"}, // -1.0, nonzero high word
      {0x8000000000000001ull, 0x010, "-denormal"},
      {0x8000000000000000ull, 0x020, "-0"},
      {0x0000000000000000ull, 0x040, "+0"},
      {0x0000000000000001ull, 0x080, "+denormal"},
      {0x3FF0000000000000ull, 0x100, "+normal"}, // 1.0, nonzero high word
      {0x7FF0000000000000ull, 0x200, "+Inf"},
  }};
  check(/*v_cmp_class_f64=*/18, rows, /*is_f64=*/true);
}

} // namespace
