// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vop3_shift64_scalar_correctness_test.cpp
/// @brief Regression test for the scalar (functional-sim) codegen of the
/// 64-bit shift family and v_lshl_add, covering the two bugs fixed alongside
/// it:
///
///  1. v_lshlrev_b64 / v_lshrrev_b64 / v_ashrrev_i64 / v_lshl_add_u64 take a
///     32-bit shift-count operand, but the b64/i64 dtype default forced
///     read_lane64 on it. read_lane64 reads VGPR[idx]+VGPR[idx+1], so an op
///     near the end of a VGPR block could read past the operand-bound
///     register. Each shift-count operand here is a single VGPR whose adjacent
///     VGPR is seeded with a poison word; the results must depend only on the
///     count register. (The arithmetic mask below hides the poison's *value*,
///     so this pins the single-register contract / out-of-bounds intent; a
///     regression that widened the read AND dropped the mask is caught.)
///
///  2. v_lshl_add_u32 / v_lshl_add_u64 lowered the shift with no mask, so a
///     count >= the element width was undefined behavior. HW masks the count
///     to the low 6 bits (64-bit forms) or low 5 bits (32-bit forms); the
///     cases below include counts of exactly the width and above it.
///
/// The 64-bit value and destination operands carry nonzero high words so a
/// low-word-only read or a high-word-dropping write is visible. No SIMD is
/// involved; this exercises the plain decode+execute scalar path.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace {

using namespace rocjitsu;

constexpr uint32_t SGPRS_PER_WF = 106;
constexpr uint32_t VGPRS_PER_WF = 256;
constexpr uint32_t POISON = 0xDEADBEEFu;
constexpr uint64_t DST_SENTINEL = 0xA5A5A5A5A5A5A5A5ull;

// VOP3 word0 = encoding[31:26]=0x34 | op[25:16] | vdst[7:0]; word1 packs the
// 9-bit src operand codes (see Vop3MachineInst). The 10-bit op numbers are the
// decoder's sub_decode_vop3 table indices for CDNA4.
constexpr uint32_t VOP3_PREFIX = 0x34u << 26; // 0xD0000000
constexpr uint32_t OP_LSHLREV_B64 = 655;
constexpr uint32_t OP_LSHRREV_B64 = 656;
constexpr uint32_t OP_ASHRREV_I64 = 657;
constexpr uint32_t OP_LSHL_ADD_U64 = 520;
constexpr uint32_t OP_LSHL_ADD_U32 = 509;

// VGPR operand code (v0 == 256 + 0). vdst is a raw VGPR number.
constexpr uint32_t vgpr_src(uint32_t n) { return 256u + n; }

struct Vop3Words {
  uint32_t w0, w1;
};
Vop3Words vop3(uint32_t op, uint32_t vdst, uint32_t src0, uint32_t src1, uint32_t src2) {
  return {VOP3_PREFIX | ((op & 0x3FFu) << 16) | (vdst & 0xFFu),
          (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9) | ((src2 & 0x1FFu) << 18)};
}

struct Fixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wf = nullptr;

  Fixture() : gpu_mem("vop3_shift64_mem"), l2("vop3_shift64_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_vop3_shift64", cfg, &gpu_mem, &l2);
    decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
    if (wf == nullptr)
      ADD_FAILURE() << "dispatch_wf returned nullptr";
  }

  void write32(uint32_t reg, uint32_t v) { cu->write_vgpr(reg, 0, v); }
  void write64(uint32_t reg, uint64_t v) {
    cu->write_vgpr(reg, 0, static_cast<uint32_t>(v));
    cu->write_vgpr(reg + 1, 0, static_cast<uint32_t>(v >> 32));
  }
  uint64_t read64(uint32_t reg) {
    return static_cast<uint64_t>(cu->read_vgpr(reg + 1, 0)) << 32 | cu->read_vgpr(reg, 0);
  }

  void exec(const Vop3Words &w, const char *expect_mnemonic) {
    if (wf == nullptr) {
      ADD_FAILURE() << "Fixture setup failed";
      return;
    }
    wf->set_exec(~0ULL);
    uint32_t words[4] = {w.w0, w.w1, 0u, 0u};
    Instruction *inst = decoder->decode(words);
    ASSERT_NE(inst, nullptr) << "decode failed for " << expect_mnemonic;
    EXPECT_EQ(inst->mnemonic(), expect_mnemonic) << "decoded wrong op";
    cu->execute_instruction(inst, *wf);
    delete inst;
  }

  // rev-shift shape: src0 = 32-bit count (v0, poison in v1), src1 = 64-bit
  // value (v2:v3), dst = v4:v5. Returns the 64-bit destination.
  uint64_t run_revshift(uint32_t op, const char *mnem, uint32_t count, uint64_t value) {
    uint32_t vb = wf->vgpr_alloc().base;
    write32(vb + 0, count);
    write32(vb + 1, POISON); // adjacent to the single-register shift count
    write64(vb + 2, value);
    write64(vb + 4, DST_SENTINEL);
    exec(vop3(op, /*vdst=*/4, vgpr_src(0), vgpr_src(2), /*src2=*/0), mnem);
    return read64(vb + 4);
  }

  // v_lshl_add_u64: src0 = 64-bit value (v0:v1), src1 = 32-bit count (v2,
  // poison in v3), src2 = 64-bit addend (v4:v5), dst = v6:v7.
  uint64_t run_lshl_add_u64(uint64_t value, uint32_t count, uint64_t addend) {
    uint32_t vb = wf->vgpr_alloc().base;
    write64(vb + 0, value);
    write32(vb + 2, count);
    write32(vb + 3, POISON); // adjacent to the single-register shift count
    write64(vb + 4, addend);
    write64(vb + 6, DST_SENTINEL);
    exec(vop3(OP_LSHL_ADD_U64, /*vdst=*/6, vgpr_src(0), vgpr_src(2), vgpr_src(4)),
         "v_lshl_add_u64");
    return read64(vb + 6);
  }

  // v_lshl_add_u32: src0 = value (v0), src1 = count (v1), src2 = addend (v2),
  // dst = v3. Returns the low 32 bits.
  uint32_t run_lshl_add_u32(uint32_t value, uint32_t count, uint32_t addend) {
    uint32_t vb = wf->vgpr_alloc().base;
    write32(vb + 0, value);
    write32(vb + 1, count);
    write32(vb + 2, addend);
    write32(vb + 3, static_cast<uint32_t>(DST_SENTINEL));
    exec(vop3(OP_LSHL_ADD_U32, /*vdst=*/3, vgpr_src(0), vgpr_src(1), vgpr_src(2)),
         "v_lshl_add_u32");
    return cu->read_vgpr(vb + 3, 0);
  }
};

// Shift counts spanning 0, mid-range, exactly the width, and above it. The
// >= width entries lock the HW low-6-bit (low-5-bit for u32) masking.
const uint32_t kCounts64[] = {0u, 1u, 31u, 32u, 63u, 64u, 65u, 70u, 127u};
const uint32_t kCounts32[] = {0u, 1u, 15u, 31u, 32u, 33u, 63u, 64u};

// 64-bit operands with nonzero high words so a low-word-only read/write shows.
const uint64_t kVals64[] = {
    0x0123456789ABCDEFull, 0xFEDCBA9876543210ull, 0x8000000000000001ull,
    0x00000000DEADBEEFull, 0xFFFFFFFF00000000ull,
};

TEST(Vop3Shift64Scalar, LshlrevB64) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  for (uint64_t v : kVals64)
    for (uint32_t c : kCounts64) {
      uint64_t got = fx.run_revshift(OP_LSHLREV_B64, "v_lshlrev_b64", c, v);
      uint64_t want = v << (c & 63u);
      EXPECT_EQ(got, want) << "v_lshlrev_b64 v=0x" << std::hex << v << " count=" << std::dec << c;
    }
}

TEST(Vop3Shift64Scalar, LshrrevB64) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  for (uint64_t v : kVals64)
    for (uint32_t c : kCounts64) {
      uint64_t got = fx.run_revshift(OP_LSHRREV_B64, "v_lshrrev_b64", c, v);
      uint64_t want = v >> (c & 63u);
      EXPECT_EQ(got, want) << "v_lshrrev_b64 v=0x" << std::hex << v << " count=" << std::dec << c;
    }
}

TEST(Vop3Shift64Scalar, AshrrevI64) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  for (uint64_t v : kVals64)
    for (uint32_t c : kCounts64) {
      uint64_t got = fx.run_revshift(OP_ASHRREV_I64, "v_ashrrev_i64", c, v);
      uint64_t want = static_cast<uint64_t>(static_cast<int64_t>(v) >> (c & 63u));
      EXPECT_EQ(got, want) << "v_ashrrev_i64 v=0x" << std::hex << v << " count=" << std::dec << c;
    }
}

TEST(Vop3Shift64Scalar, LshlAddU64) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  const uint64_t addend = 0x1111222233334444ull;
  for (uint64_t v : kVals64)
    for (uint32_t c : kCounts64) {
      uint64_t got = fx.run_lshl_add_u64(v, c, addend);
      uint64_t want = (v << (c & 63u)) + addend;
      EXPECT_EQ(got, want) << "v_lshl_add_u64 v=0x" << std::hex << v << " count=" << std::dec << c;
    }
}

TEST(Vop3Shift64Scalar, LshlAddU32) {
  Fixture fx;
  ASSERT_NE(fx.cu, nullptr);
  const uint32_t addend = 0x0BADF00Du;
  for (uint32_t v : {0x00000001u, 0x80000001u, 0xFFFFFFFFu, 0x0000FFFFu, 0xDEADBEEFu})
    for (uint32_t c : kCounts32) {
      uint32_t got = fx.run_lshl_add_u32(v, c, addend);
      uint32_t want = (v << (c & 31u)) + addend;
      EXPECT_EQ(got, want) << "v_lshl_add_u32 v=0x" << std::hex << v << " count=" << std::dec << c;
    }
}

} // namespace
