// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <bit>
#include <cfenv>
#include <gtest/gtest.h>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace rocjitsu;

struct ArithmeticCase {
  std::string name;
  rj_code_arch_t arch;
  std::array<uint32_t, 3> words;
  std::vector<std::pair<uint32_t, uint32_t>> sources;
  std::vector<std::pair<uint32_t, uint32_t>> expected;
  uint32_t mode;
  int host_rounding;
  uint32_t mxcsr_mask = 0;
  uint32_t mxcsr_bits = 0;
  uint64_t vcc = 0;
};

void PrintTo(const ArithmeticCase &test, std::ostream *stream) { *stream << test.name; }

// Encodings assembled with llvm-mc; expected values follow GPU MODE, independently of host MODE.
const std::vector<ArithmeticCase> kCases{
    {"Gfx1250VopdDenormFlush",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xc9080300u, 0x06060300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}, {7, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"Gfx1250VopdDenormPreserve",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xc9080300u, 0x06060300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000002u}, {7, 0x00000002u}},
     240u,
     FE_TONEAREST},
    {"Gfx1250VopdRoundUp",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xc9080300u, 0x06060300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}, {7, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"Gfx1250VopdHostRoundUp",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xc9080300u, 0x06060300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800000u}, {7, 0x3f800000u}},
     240u,
     FE_UPWARD},
    {"Gfx1250Vopd3F64RoundUp",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xcf848100u, 0x0002010au, 0x08000006u},
     {{0, 0x00000000u}, {1, 0x3ff00000u}, {2, 0x00000000u}, {3, 0x3ca00000u}, {10, 0x12345678u}},
     {{6, 0x00000001u}, {7, 0x3ff00000u}, {8, 0x12345678u}},
     244u,
     FE_TONEAREST},
    {"Gfx1250Vopd3F64DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xcf848100u, 0x0002010au, 0x08000006u},
     {{0, 0x00000001u}, {1, 0x00000000u}, {2, 0x00000001u}, {3, 0x00000000u}, {10, 0x12345678u}},
     {{6, 0x00000000u}, {7, 0x00000000u}, {8, 0x12345678u}},
     0u,
     FE_TONEAREST},
    {"Gfx950Vop3OmodControl",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd1010006u, 0x08020300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}},
     {{6, 0x40800000u}},
     0u,
     FE_TONEAREST},
    {"gfx1100VopdDenormFlush",
     ROCJITSU_CODE_ARCH_RDNA3,
     {0xc9080300u, 0x06060702u},
     {{0, 0x00000001u}, {1, 0x00000001u}, {2, 0x00000001u}, {3, 0x00000001u}},
     {{6, 0x00000000u}, {7, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1100VopdRoundUp",
     ROCJITSU_CODE_ARCH_RDNA3,
     {0xc9080300u, 0x06060702u},
     {{0, 0x3f800000u}, {1, 0x33800000u}, {2, 0x3f800000u}, {3, 0x33800000u}},
     {{6, 0x3f800001u}, {7, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1150VopdDenormFlush",
     ROCJITSU_CODE_ARCH_RDNA3_5,
     {0xc9080300u, 0x06060702u},
     {{0, 0x00000001u}, {1, 0x00000001u}, {2, 0x00000001u}, {3, 0x00000001u}},
     {{6, 0x00000000u}, {7, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1150VopdRoundUp",
     ROCJITSU_CODE_ARCH_RDNA3_5,
     {0xc9080300u, 0x06060702u},
     {{0, 0x3f800000u}, {1, 0x33800000u}, {2, 0x3f800000u}, {3, 0x33800000u}},
     {{6, 0x3f800001u}, {7, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1201VopdDenormFlush",
     ROCJITSU_CODE_ARCH_RDNA4,
     {0xc9080300u, 0x06060702u},
     {{0, 0x00000001u}, {1, 0x00000001u}, {2, 0x00000001u}, {3, 0x00000001u}},
     {{6, 0x00000000u}, {7, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1201VopdRoundUp",
     ROCJITSU_CODE_ARCH_RDNA4,
     {0xc9080300u, 0x06060702u},
     {{0, 0x3f800000u}, {1, 0x33800000u}, {2, 0x3f800000u}, {3, 0x33800000u}},
     {{6, 0x3f800001u}, {7, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1250AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0x060c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1250AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0x060c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1250AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xd5030006u, 0x02020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1250AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xd5030006u, 0x02020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1250FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xd6130006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1250FmaF32RneControl",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xd6130006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx950AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0x020c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx950AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0x020c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx950AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd1010006u, 0x00020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx950AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd1010006u, 0x00020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx950FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd1cb0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx950FmaF32RneControl",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd1cb0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx942AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA3,
     {0x020c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx942AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA3,
     {0x020c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx942AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_CDNA3,
     {0xd1010006u, 0x00020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx942AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA3,
     {0xd1010006u, 0x00020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx942FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA3,
     {0xd1cb0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx942FmaF32RneControl",
     ROCJITSU_CODE_ARCH_CDNA3,
     {0xd1cb0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx90aAddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA2,
     {0x020c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx90aAddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA2,
     {0x020c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx90aAddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_CDNA2,
     {0xd1010006u, 0x00020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx90aAddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA2,
     {0xd1010006u, 0x00020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx90aFmaF32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA2,
     {0xd1cb0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx90aFmaF32RneControl",
     ROCJITSU_CODE_ARCH_CDNA2,
     {0xd1cb0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx908AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA1,
     {0x020c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx908AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA1,
     {0x020c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx908AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_CDNA1,
     {0xd1010006u, 0x00020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx908AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA1,
     {0xd1010006u, 0x00020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx908FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA1,
     {0xd1cb0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx908FmaF32RneControl",
     ROCJITSU_CODE_ARCH_CDNA1,
     {0xd1cb0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx1100AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA3,
     {0x060c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1100AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA3,
     {0x060c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1100AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_RDNA3,
     {0xd5030006u, 0x02020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1100AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA3,
     {0xd5030006u, 0x02020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1100FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA3,
     {0xd6130006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1100FmaF32RneControl",
     ROCJITSU_CODE_ARCH_RDNA3,
     {0xd6130006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx1150AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA3_5,
     {0x060c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1150AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA3_5,
     {0x060c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1150AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_RDNA3_5,
     {0xd5030006u, 0x02020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1150AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA3_5,
     {0xd5030006u, 0x02020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1150FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA3_5,
     {0xd6130006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1150FmaF32RneControl",
     ROCJITSU_CODE_ARCH_RDNA3_5,
     {0xd6130006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx1201AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA4,
     {0x060c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1201AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA4,
     {0x060c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1201AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_RDNA4,
     {0xd5030006u, 0x02020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1201AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA4,
     {0xd5030006u, 0x02020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1201FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA4,
     {0xd6130006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1201FmaF32RneControl",
     ROCJITSU_CODE_ARCH_RDNA4,
     {0xd6130006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx1030AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA2,
     {0x060c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1030AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA2,
     {0x060c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1030AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_RDNA2,
     {0xd5030006u, 0x02020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1030AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA2,
     {0xd5030006u, 0x02020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1030FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA2,
     {0xd54b0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1030FmaF32RneControl",
     ROCJITSU_CODE_ARCH_RDNA2,
     {0xd54b0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx1010AddF32e32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA1,
     {0x060c0300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1010AddF32e32DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA1,
     {0x060c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1010AddF32e64RoundUp",
     ROCJITSU_CODE_ARCH_RDNA1,
     {0xd5030006u, 0x02020300u},
     {{0, 0x3f800000u}, {1, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1010AddF32e64DenormFlush",
     ROCJITSU_CODE_ARCH_RDNA1,
     {0xd5030006u, 0x02020300u},
     {{0, 0x00000001u}, {1, 0x00000001u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx1010FmaF32RoundUp",
     ROCJITSU_CODE_ARCH_RDNA1,
     {0xd54b0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800001u}},
     241u,
     FE_TONEAREST},
    {"gfx1010FmaF32RneControl",
     ROCJITSU_CODE_ARCH_RDNA1,
     {0xd54b0006u, 0x040a0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {2, 0x33800000u}},
     {{6, 0x3f800000u}},
     240u,
     FE_TONEAREST},
    {"gfx1250PackedFmaF16ModeControl",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0xcc0e4006u, 0x1c0a0300u},
     {{0, 0x3c003c00u}, {1, 0x3c003c00u}, {2, 0x10001000u}, {6, 0x00000000u}},
     {{6, 0x3c013c01u}},
     244u,
     FE_TONEAREST},
    {"gfx1250AddF64RoundUp",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0x040c0500u},
     {{0, 0x00000000u}, {1, 0x3ff00000u}, {2, 0x00000000u}, {3, 0x3ca00000u}},
     {{6, 0x00000001u}, {7, 0x3ff00000u}},
     244u,
     FE_TONEAREST},
    {"gfx1250AddF64DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA5,
     {0x040c0500u},
     {{0, 0x00000001u}, {1, 0x00000000u}, {2, 0x00000001u}, {3, 0x00000000u}},
     {{6, 0x00000000u}, {7, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"gfx950PackedFmaF16ModeControl",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd38e4006u, 0x1c0a0300u},
     {{0, 0x3c003c00u}, {1, 0x3c003c00u}, {2, 0x10001000u}, {6, 0x00000000u}},
     {{6, 0x3c013c01u}},
     244u,
     FE_TONEAREST},
    {"gfx950AddF64RoundUp",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd2800006u, 0x00020500u},
     {{0, 0x00000000u}, {1, 0x3ff00000u}, {2, 0x00000000u}, {3, 0x3ca00000u}},
     {{6, 0x00000001u}, {7, 0x3ff00000u}},
     244u,
     FE_TONEAREST},
    {"gfx950AddF64DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd2800006u, 0x00020500u},
     {{0, 0x00000001u}, {1, 0x00000000u}, {2, 0x00000001u}, {3, 0x00000000u}},
     {{6, 0x00000000u}, {7, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"Gfx950AddF16RoundUp",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0x3e0c0300u},
     {{0, 0x00003c00u}, {1, 0x00001000u}, {6, 0x00000000u}},
     {{6, 0x00003c01u}},
     244u,
     FE_TONEAREST},
    {"Gfx950AddF16DenormFlush",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0x3e0c0300u},
     {{0, 0x00000001u}, {1, 0x00000001u}, {6, 0x00000000u}},
     {{6, 0x00000000u}},
     0u,
     FE_TONEAREST},
    {"Gfx950AddF16TinyRoundUpE64",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd11f0006u, 0x00020300u},
     {{0, 0x3c00u}, {1, 1u}},
     {{6, 0x3c01u}},
     0xf4u,
     FE_TONEAREST},
    // OMOD scales the rounded half: 65504 * 2 overflows before div:2.
    {"Gfx950MulF16RoundBeforeScale",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd1220006u, 0x18020300u},
     {{0, 0x7bffu}, {1, 0x4000u}},
     {{6, 0x7c00u}},
     0x44u,
     FE_TONEAREST},
    {"Gfx950MulF32ScaleRoundZero",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd1050006u, 0x08020300u},
     {{0, 0x7f7fffffu}, {1, 0x3f800000u}},
     {{6, 0x7f7fffffu}},
     3u,
     FE_TONEAREST},
    {"Gfx950SubF32RoundDown",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0x040c0300u},
     {{0, 0xbf800000u}, {1, 0x33800000u}},
     {{6, 0xbf800001u}},
     0xf2u,
     FE_TONEAREST},
    {"Gfx950SubrevF32RoundDown",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0x060c0300u},
     {{1, 0xbf800000u}, {0, 0x33800000u}},
     {{6, 0xbf800001u}},
     0xf2u,
     FE_TONEAREST},
    {"Gfx950MulF32InputFlush",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0x0a0c0300u},
     {{0, 1u}, {1, 0x4b000000u}},
     {{6, 0u}},
     0u,
     FE_TONEAREST},
    {"Gfx950FmacF32RoundUp",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0x760c0300u},
     {{0, 0x3f800000u}, {1, 0x3f800000u}, {6, 0x33800000u}},
     {{6, 0x3f800001u}},
     0xf1u,
     FE_TONEAREST},

};

// These instruction results were checked independently on gfx1100/gfx1201.
// Encodings come from llvm-mc; the same rules cover the shared CDNA executors.
std::vector<ArithmeticCase> adjacent_fp_cases() {
  struct Encodings {
    const char *name;
    rj_code_arch_t arch;
    std::array<std::array<uint32_t, 3>, 7> words;
  };
  const std::array encodings{
      Encodings{"gfx1250",
                ROCJITSU_CODE_ARCH_CDNA5,
                {{
                    {0xd71c0006u, 0x02020300u},
                    {0xd72b0006u, 0x02020500u},
                    {0x7e0c8100u},
                    {0xd5c00006u, 0x02010100u},
                    {0x7e0c7f00u},
                    {0xd5bf0006u, 0x02010100u},
                    {0xd6540006u, 0x040a0300u},
                }}},
      Encodings{"gfx950",
                ROCJITSU_CODE_ARCH_CDNA4,
                {{
                    {0xd2880006u, 0x00020300u},
                    {0xd2840006u, 0x00020500u},
                    {0x7e0c6900u},
                    {0xd1740006u, 0x00000100u},
                    {0x7e0c6700u},
                    {0xd1730006u, 0x00000100u},
                    {0xd2070006u, 0x040a0300u},
                }}},
      Encodings{"gfx1201",
                ROCJITSU_CODE_ARCH_RDNA4,
                {{
                    {0xd71c0006u, 0x02020300u},
                    {0xd72b0006u, 0x02020500u},
                    {0x7e0c8100u},
                    {0xd5c00006u, 0x02010100u},
                    {0x7e0c7f00u},
                    {0xd5bf0006u, 0x02010100u},
                    {0xd6540006u, 0x040a0300u},
                }}},
      Encodings{"gfx1100",
                ROCJITSU_CODE_ARCH_RDNA3,
                {{
                    {0xd71c0006u, 0x02020300u},
                    {0xd72b0006u, 0x02020500u},
                    {0x7e0c8100u},
                    {0xd5c00006u, 0x02010100u},
                    {0x7e0c7f00u},
                    {0xd5bf0006u, 0x02010100u},
                    {0xd6540006u, 0x040a0300u},
                }}},
      Encodings{"gfx1150",
                ROCJITSU_CODE_ARCH_RDNA3_5,
                {{
                    {0xd71c0006u, 0x02020300u},
                    {0xd72b0006u, 0x02020500u},
                    {0x7e0c8100u},
                    {0xd5c00006u, 0x02010100u},
                    {0x7e0c7f00u},
                    {0xd5bf0006u, 0x02010100u},
                    {0xd6540006u, 0x040a0300u},
                }}},
      Encodings{"gfx942",
                ROCJITSU_CODE_ARCH_CDNA3,
                {{
                    {0xd2880006u, 0x00020300u},
                    {0xd2840006u, 0x00020500u},
                    {0x7e0c6900u},
                    {0xd1740006u, 0x00000100u},
                    {0x7e0c6700u},
                    {0xd1730006u, 0x00000100u},
                    {0xd2070006u, 0x040a0300u},
                }}},
      Encodings{"gfx90a",
                ROCJITSU_CODE_ARCH_CDNA2,
                {{
                    {0xd2880006u, 0x00020300u},
                    {0xd2840006u, 0x00020500u},
                    {0x7e0c6900u},
                    {0xd1740006u, 0x00000100u},
                    {0x7e0c6700u},
                    {0xd1730006u, 0x00000100u},
                    {0xd2070006u, 0x040a0300u},
                }}},
      Encodings{"gfx908",
                ROCJITSU_CODE_ARCH_CDNA1,
                {{
                    {0xd2880006u, 0x00020300u},
                    {0xd2840006u, 0x00020500u},
                    {0x7e0c6900u},
                    {0xd1740006u, 0x00000100u},
                    {0x7e0c6700u},
                    {0xd1730006u, 0x00000100u},
                    {0xd2070006u, 0x040a0300u},
                }}},
      Encodings{"gfx1030",
                ROCJITSU_CODE_ARCH_RDNA2,
                {{
                    {0xd7620006u, 0x02020300u},
                    {0xd5680006u, 0x02020500u},
                    {0x7e0c8100u},
                    {0xd5c00006u, 0x02010100u},
                    {0x7e0c7f00u},
                    {0xd5bf0006u, 0x02010100u},
                    {0xd75f0006u, 0x040a0300u},
                }}},
      Encodings{"gfx1010",
                ROCJITSU_CODE_ARCH_RDNA1,
                {{
                    {0xd7620006u, 0x02020300u},
                    {0xd5680006u, 0x02020500u},
                    {0x7e0c8100u},
                    {0xd5c00006u, 0x02010100u},
                    {0x7e0c7f00u},
                    {0xd5bf0006u, 0x02010100u},
                    {0xd75f0006u, 0x040a0300u},
                }}},
  };
  std::vector<ArithmeticCase> cases;
  for (const Encodings &encoding : encodings) {
    const auto add = [&](const std::string &name, size_t instruction,
                         std::vector<std::pair<uint32_t, uint32_t>> sources,
                         std::vector<std::pair<uint32_t, uint32_t>> expected, uint32_t mode,
                         int host_rounding = FE_TONEAREST) {
      if (instruction == 6 && encoding.arch != ROCJITSU_CODE_ARCH_CDNA1 &&
          encoding.arch != ROCJITSU_CODE_ARCH_CDNA2 && encoding.arch != ROCJITSU_CODE_ARCH_CDNA3 &&
          encoding.arch != ROCJITSU_CODE_ARCH_CDNA4)
        expected.front().second |= 0xdead0000u;
      cases.push_back({std::string(encoding.name) + name, encoding.arch,
                       encoding.words[instruction], std::move(sources), std::move(expected), mode,
                       host_rounding});
    };
    for (uint32_t rounding = 0; rounding < 4; ++rounding) {
      const uint32_t mode = 0xf0u | rounding | (rounding << 2);
      const std::string suffix = "Round" + std::to_string(rounding);
      add("LdexpF32Tiny" + suffix, 0, {{0, 0x00800000u}, {1, uint32_t(-24)}},
          {{6, rounding == 1 ? 1u : 0u}}, mode);
      add("LdexpF64Tiny" + suffix, 1, {{0, 0u}, {1, 0x00100000u}, {2, uint32_t(-53)}},
          {{6, rounding == 1 ? 1u : 0u}, {7, 0u}}, mode);
      add("LdexpF32NegativeTiny" + suffix, 0, {{0, 0x80800000u}, {1, uint32_t(-24)}},
          {{6, rounding == 2 ? 0x80000001u : 0x80000000u}}, mode);
      add("LdexpF64NegativeTiny" + suffix, 1, {{0, 0u}, {1, 0x80100000u}, {2, uint32_t(-53)}},
          {{6, rounding == 2 ? 1u : 0u}, {7, 0x80000000u}}, mode);
      const bool to_infinity = rounding == 0 || rounding == 1;
      add("FixupF16Overflow" + suffix, 6, {{0, 0x7e2au}, {1, 0x3c00u}, {2, 0x3c00u}},
          {{6, to_infinity ? 0x7c00u : 0x7bffu}}, mode);
      add("FixupF16NegativeOverflow" + suffix, 6, {{0, 0x7c00u}, {1, 0xbc00u}, {2, 0x3c00u}},
          {{6, rounding == 0 || rounding == 2 ? 0xfc00u : 0xfbffu}}, mode);
      add("LdexpF32Overflow" + suffix, 0, {{0, 0x7f7fffffu}, {1, 1u}},
          {{6, to_infinity ? 0x7f800000u : 0x7f7fffffu}}, mode);
      add("LdexpF64Overflow" + suffix, 1, {{0, 0xffffffffu}, {1, 0x7fefffffu}, {2, 1u}},
          {{6, to_infinity ? 0u : 0xffffffffu}, {7, to_infinity ? 0x7ff00000u : 0x7fefffffu}},
          mode);
    }
    add("LdexpF32OmodOverflow", 0, {{0, 0x7f7fffffu}, {1, 0u}}, {{6, 0x7f7fffffu}}, 0x0fu);
    cases.back().words[1] |= 0x08000000u; // mul:2
    add("LdexpF64OmodOverflow", 1, {{0, 0xffffffffu}, {1, 0x7fefffffu}, {2, 0u}},
        {{6, 0xffffffffu}, {7, 0x7fefffffu}}, 0x0fu);
    cases.back().words[1] |= 0x08000000u; // mul:2
    add("LdexpF32IgnoresHostRoundDown", 0, {{0, 0x00800000u}, {1, uint32_t(-24)}}, {{6, 1u}}, 0xf5u,
        FE_DOWNWARD);
    add("LdexpF64IgnoresHostRoundDown", 1, {{0, 0u}, {1, 0x00100000u}, {2, uint32_t(-53)}},
        {{6, 1u}, {7, 0u}}, 0xf5u, FE_DOWNWARD);
    add("LdexpF32InputFlush", 0, {{0, 1u}, {1, 23u}}, {{6, 0u}}, 0u);
    add("LdexpF32InputPreserve", 0, {{0, 1u}, {1, 23u}}, {{6, 0x00800000u}}, 0xf0u);
    add("LdexpF64InputFlush", 1, {{0, 1u}, {1, 0u}, {2, 52u}}, {{6, 0u}, {7, 0u}}, 0u);
    add("LdexpF64InputPreserve", 1, {{0, 1u}, {1, 0u}, {2, 52u}}, {{6, 0u}, {7, 0x00100000u}},
        0xf0u);
    add("LdexpF32HugeShift", 0, {{0, 0x3f800000u}, {1, 0x7fffffffu}}, {{6, 0x7f7fffffu}}, 0xffu);
    add("LdexpF64HugeNegativeShift", 1, {{0, 0u}, {1, 0x3ff00000u}, {2, 0x80000000u}},
        {{6, 1u}, {7, 0u}}, 0xf5u);
    for (size_t instruction = 2; instruction < 6; ++instruction) {
      const bool exponent = instruction >= 4;
      const std::string name = "Frexp" + std::to_string(instruction);
      add(name + "PreserveSubnormal", instruction, {{0, 1u}},
          {{6, exponent ? uint32_t(-148) : 0x3f000000u}}, 0xf0u);
      add(name + "FlushSubnormal", instruction, {{0, 1u}}, {{6, 0u}}, 0u);
      add(name + "FlushNegativeSubnormal", instruction, {{0, 0x80000001u}},
          {{6, exponent ? 0u : 0x80000000u}}, 0u);
      add(name + "SignalingNaN", instruction, {{0, 0xff80002au}},
          {{6, exponent ? 0u : 0xffc0002au}}, 0xf0u);
      add(name + "NegativeNormal", instruction, {{0, 0xbe800000u}},
          {{6, exponent ? uint32_t(-1) : 0xbf000000u}}, 0xf0u);
    }
    add("FixupF16Sign", 6, {{0, 0xc200u}, {1, 0x3c00u}, {2, 0x3c00u}}, {{6, 0x4200u}}, 0xf0u);
    add("FixupF16Invalid", 6, {{0, 0xc200u}, {1, 0u}, {2, 0u}}, {{6, 0xfe00u}}, 0xf0u);
    add("FixupF16PreservesProvisionalInfinity", 6, {{0, 0x7c00u}, {1, 0x3c00u}, {2, 0x3c00u}},
        {{6, 0x7c00u}}, 0xf0u);
    add("FixupF16FlushDenominator", 6, {{0, 0x3c00u}, {1, 1u}, {2, 0x3c00u}}, {{6, 0x7c00u}}, 0u);
    add("FixupF16FlushNumerator", 6, {{0, 0x3c00u}, {1, 0x3c00u}, {2, 0x8001u}}, {{6, 0x8000u}},
        0u);
    add("FixupF16PreserveDenominator", 6, {{0, 0x3c00u}, {1, 1u}, {2, 0x3c00u}}, {{6, 0x3c00u}},
        0xf0u);
    add("FixupF16ProvisionalNaN", 6, {{0, 0x7e2au}, {1, 0xbc00u}, {2, 0x3c00u}}, {{6, 0xfc00u}},
        0xf0u);
    add("FixupF16QuietNaNPayload", 6, {{0, 0u}, {1, 0x3c00u}, {2, 0xfc2au}}, {{6, 0xfe2au}}, 0xf0u);
  }
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4}) {
    for (uint32_t mode = 0; mode < 256; mode += 17) {
      cases.push_back(
          {std::string(arch == ROCJITSU_CODE_ARCH_RDNA3 ? "Gfx1100" : "Gfx1201") +
               "ScaleF64SignalingNaNMode" + std::to_string(mode),
           arch,
           {0xd6fd6a06u, 0x040a0100u},
           {{0, 0x510b2fa8u}, {1, 0x7ff19bdeu}, {2, 0xc874d413u}, {3, 0x90fe0a8fu}},
           {{6, 0x510b2fa8u}, {7, arch == ROCJITSU_CODE_ARCH_RDNA3 ? 0x7ff19bdeu : 0x7ff99bdeu}},
           mode,
           FE_TONEAREST});
    }
  }
  return cases;
}

std::vector<ArithmeticCase> ldexp_f16_mode_cases() {
  struct Encoding {
    rj_code_arch_t arch;
    const char *name;
    std::array<uint32_t, 3> words;
  };
  // llvm-mc encodings: v_ldexp_f16 v6, v0, v1 (low halves on true16 targets).
  const std::array encodings{
      Encoding{ROCJITSU_CODE_ARCH_CDNA4, "Gfx950E32", {0x660c0300u}},
      Encoding{ROCJITSU_CODE_ARCH_CDNA4, "Gfx950E64", {0xd1330006u, 0x00020300u}},
      Encoding{ROCJITSU_CODE_ARCH_CDNA5, "Gfx1250E32", {0x760c0300u}},
      Encoding{ROCJITSU_CODE_ARCH_CDNA5, "Gfx1250E64", {0xd53b0006u, 0x02020300u}},
      Encoding{ROCJITSU_CODE_ARCH_RDNA4, "Gfx1201E32", {0x760c0300u}},
      Encoding{ROCJITSU_CODE_ARCH_RDNA4, "Gfx1201E64", {0xd53b0006u, 0x02020300u}},
  };
  std::vector<ArithmeticCase> cases;
  for (const Encoding &encoding : encodings) {
    const auto add = [&](std::string name, uint32_t input, uint16_t exponent, uint32_t expected,
                         uint32_t mode, int host_rounding = FE_TONEAREST) {
      cases.push_back({encoding.name + name,
                       encoding.arch,
                       encoding.words,
                       {{0, input}, {1, exponent}, {6, 0u}},
                       {{6, expected}},
                       mode,
                       host_rounding});
    };
    for (uint32_t rounding = 0; rounding < 4; ++rounding) {
      const std::string suffix = "Round" + std::to_string(rounding);
      const uint32_t mode = 0xf0u | (rounding << 2);
      add("PositiveHalfway" + suffix, 0x0401u, 0xffffu, rounding == 1 ? 0x0201u : 0x0200u, mode,
          FE_DOWNWARD);
      add("NegativeHalfway" + suffix, 0x8401u, 0xffffu, rounding == 2 ? 0x8201u : 0x8200u, mode,
          FE_UPWARD);
      add("HugePositiveExponent" + suffix, 0x3c00u, 0x7fffu, rounding <= 1 ? 0x7c00u : 0x7bffu,
          mode);
      add("HugeNegativeExponent" + suffix, 0x3c00u, 0x8000u, rounding == 1 ? 0x0001u : 0x0000u,
          mode);
    }
    for (uint32_t denorm = 0; denorm < 4; ++denorm) {
      const std::string suffix = "Denorm" + std::to_string(denorm);
      add("Input" + suffix, 0x0001u, 10, (denorm & 1u) ? 0x0400u : 0, denorm << 6);
      add("Output" + suffix, 0x0400u, 0xffffu, (denorm & 2u) ? 0x0200u : 0, denorm << 6);
      add("NegativeOutput" + suffix, 0x8400u, 0xffffu, (denorm & 2u) ? 0x8200u : 0x8000u,
          denorm << 6);
    }
    // OMOD scales the result after rounding it to half, so div:2 cannot rescue
    // an overflow (gfx1201 captures).
    if (encoding.words[1] != 0) {
      add("OmodAfterOverflow", 0x7bffu, 1, 0x7c00u, 0);
      cases.back().words[1] |= 3u << 27; // div:2
    }
  }
  return cases;
}

std::vector<ArithmeticCase> dx9_fma_cases() {
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    const bool has_accumulator_form =
        arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
    for (uint32_t form = 0; form < (has_accumulator_form ? 3u : 1u); ++form)
      for (uint32_t denorm = 0; denorm < 4; ++denorm)
        for (uint32_t rounding = 0; rounding < 4; ++rounding)
          for (bool ieee : {false, true}) {
            const std::string prefix = "Arch" + std::to_string(arch) + "Form" +
                                       std::to_string(form) + "Denorm" + std::to_string(denorm) +
                                       "Round" + std::to_string(rounding) + "Ieee" +
                                       std::to_string(ieee);
            // v_fma_dx9_zero_f32 v6,v0,v1,v2 and v_fmac_dx9_zero_f32 v6,v0,v1,
            // assembled with llvm-mc. RDNA4/CDNA5 expose only the three-source form.
            const std::array<uint32_t, 3> words =
                form == 0   ? std::array{0xd6090006u, 0x040a0300u, 0u}
                : form == 1 ? std::array{0x0c0c0300u, 0u, 0u}
                            : std::array{0xd5060006u, 0x00020300u, 0u};
            const auto add = [&](const std::string &name, uint32_t a, uint32_t b, uint32_t c,
                                 uint32_t expected, uint32_t omod = 0) {
              auto encoding = words;
              encoding[1] |= omod << 27;
              cases.push_back({prefix + name,
                               arch,
                               encoding,
                               {{0, a}, {1, b}, {form == 0 ? 2u : 6u, c}},
                               {{6, expected}},
                               (uint32_t(ieee) << 9) | (denorm << 4) | rounding,
                               FE_TONEAREST});
            };
            add("ZeroInf", 0u, 0x7f800000u, 0x3f800000u, 0x3f800000u);
            add("NanZero", 0x7fc01234u, 0x80000000u, 0x3f800000u, 0x3f800000u);
            add("InputFlush", 1u, 0x4b000000u, 0u, 0u);
            add("OutputFlush", 0x00800000u, 0x3f000000u, 0u, 0u);
            const uint32_t zero_sum = rounding == 2 ? 0x80000000u : 0u;
            add("AccumulatorFlush", 0u, 0x7f800000u, 0x80000001u, zero_sum);
            add("NegativeZeroAccumulator", 0u, 0x7f800000u, 0x80000000u, zero_sum);
            add("NegativeZeroProduct", 0x80000000u, 0x3f800000u, 0x80000000u, zero_sum);
            add("NormalControl", 0x3f800000u, 0x3f800000u, 0x3f800000u, 0x40000000u);
            // Captured gfx1100/gfx1201 boundary and payload witnesses.
            add("TinyBeforePacking", 0x80800000u, 0x33800000u, 0x00800000u, 0u);
            add("NormalSignificand", 0x80800000u, 0x33000000u, 0x00800000u,
                rounding < 2 ? 0x00800000u : 0u);
            add("FirstNan", 0x7fc00111u, 0x7fc00222u, 0x7fc00333u, 0x7fc00111u);
            const uint32_t nan = (ieee || !has_accumulator_form) ? 0x7fc00012u : 0x7f800012u;
            add("SignalingFirst", 0x7f800012u, 0x3f800000u, 0x3f800000u, nan);
            add("SignalingSecond", 0x3f800000u, 0x7f800012u, 0x3f800000u, nan);
            add("SignalingAccumulator", 0x3f800000u, 0x3f800000u, 0x7f800012u, nan);
            add("ZeroProductSignalingAccumulator", 0u, 0x7f800000u, 0x7f800012u, nan);
            if (form != 1)
              for (uint32_t omod = 1; omod < 4; ++omod) {
                // Mandatory output flushing keeps OMOD enabled regardless of
                // denormal MODE. RDNA3/3.5 still disable it in IEEE mode.
                const bool enabled = !has_accumulator_form || !ieee;
                const uint32_t scaled = !enabled    ? 0x40000000u
                                        : omod == 1 ? 0x40800000u
                                        : omod == 2 ? 0x41000000u
                                                    : 0x3f800000u;
                add("Omod" + std::to_string(omod), 0x3f800000u, 0x3f800000u, 0x3f800000u, scaled,
                    omod);
                add("OmodZero" + std::to_string(omod), 0u, 0x7f800000u, 0x80000000u,
                    enabled ? 0u : zero_sum, omod);
                if (omod == 3)
                  add("OmodUnderflow", 0x80800000u, 0x3f800000u, 0u,
                      enabled ? 0x80000000u : 0x80800000u, omod);
              }
          }
  }
  return cases;
}

std::vector<ArithmeticCase> binary_f32_policy_cases() {
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
        ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
        ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool rdna_encoding = arch >= ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA5;
    const bool always_quiets = arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
    for (bool multiply : {false, true})
      for (bool e64 : {false, true})
        for (uint32_t denorm = 0; denorm < 4; ++denorm)
          for (uint32_t rounding = 0; rounding < 4; ++rounding)
            for (bool ieee : {false, true}) {
              // Assembled for each target with llvm-mc. Expected raw bits were
              // captured on gfx1100/gfx1201; the shared MODE policy also applies
              // to earlier RDNA/CDNA arithmetic and CDNA5's always-quiet policy.
              const uint32_t first = e64 ? (rdna_encoding ? (multiply ? 0xd5080006u : 0xd5030006u)
                                                          : (multiply ? 0xd1050006u : 0xd1010006u))
                                         : (rdna_encoding ? (multiply ? 0x100c0300u : 0x060c0300u)
                                                          : (multiply ? 0x0a0c0300u : 0x020c0300u));
              const std::array<uint32_t, 3> words{
                  first, e64 ? (rdna_encoding ? 0x02020300u : 0x00020300u) : 0u, 0u};
              const std::string prefix = "Arch" + std::to_string(arch) +
                                         (multiply ? "Mul" : "Add") + (e64 ? "E64" : "E32") +
                                         "Denorm" + std::to_string(denorm) + "Round" +
                                         std::to_string(rounding) + "Ieee" + std::to_string(ieee);
              auto add = [&](const char *name, uint32_t a, uint32_t b, uint32_t expected) {
                cases.push_back({prefix + name,
                                 arch,
                                 words,
                                 {{0, a}, {1, b}},
                                 {{6, expected}},
                                 (uint32_t(ieee) << 9) | (denorm << 4) | rounding,
                                 FE_UPWARD});
              };
              const uint32_t nan = ieee || always_quiets ? 0x7fc01abcu : 0x7f801abcu;
              add("FirstNan", 0x7f801abcu, 0xffc00222u, nan);
              add("SecondNan", 0x3f800000u, 0x7f801abcu, nan);
              add("ZeroSign", 0x80000000u, multiply ? 0x3f800000u : 0x80000000u, 0x80000000u);
              add("OppositeZeros", 0u, 0x80000000u, multiply || rounding == 2 ? 0x80000000u : 0u);
              if (multiply) {
                const uint32_t positive = !(denorm & 2u) ? 0u
                                          : rounding < 2 ? 0x00800000u
                                                         : 0x007fffffu;
                const uint32_t negative = !(denorm & 2u)                   ? 0x80000000u
                                          : rounding == 0 || rounding == 2 ? 0x80800000u
                                                                           : 0x807fffffu;
                add("PositiveTinySignificand", 0x00800000u, 0x3f7fffffu, positive);
                add("NegativeTinySignificand", 0x80800000u, 0x3f7fffffu, negative);
              }
            }
  }
  return cases;
}

std::vector<ArithmeticCase> omod_underflow_cases() {
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (uint32_t op = 0; op < 3; ++op)
      for (uint32_t denorm = 0; denorm < 4; ++denorm)
        for (uint32_t rounding = 0; rounding < 4; ++rounding)
          for (bool ieee : {false, true}) {
            // FMA, FMAC and ADD with div:2. Physical gfx1100/gfx1201 preserve
            // the sign when scaling underflows a negative normal result.
            const std::array<uint32_t, 3> words = op == 0 ? std::array{0xd6130006u, 0x1c0a0300u, 0u}
                                                  : op == 1
                                                      ? std::array{0xd52b0006u, 0x18020300u, 0u}
                                                      : std::array{0xd5030006u, 0x18020300u, 0u};
            const bool enabled = arch == ROCJITSU_CODE_ARCH_RDNA4 ||
                                 arch == ROCJITSU_CODE_ARCH_CDNA5 || (!ieee && !(denorm & 2u));
            cases.push_back({"Arch" + std::to_string(arch) + "Op" + std::to_string(op) + "Denorm" +
                                 std::to_string(denorm) + "Round" + std::to_string(rounding) +
                                 "Ieee" + std::to_string(ieee),
                             arch,
                             words,
                             {{0, 0x80800000u}, {1, op == 2 ? 0u : 0x3f800000u}, {2, 0u}, {6, 0u}},
                             {{6, enabled ? 0x80000000u : 0x80800000u}},
                             (uint32_t(ieee) << 9) | (denorm << 4) | rounding,
                             FE_TONEAREST});
            auto boundary = cases.back();
            boundary.name += "BeforePacking";
            boundary.sources = {{0, 0x807fffffu},
                                {1, op == 2 ? 0x80800000u : 0x3f800000u},
                                {2, 0x80800000u},
                                {6, 0x80800000u}};
            boundary.expected = {{6, enabled         ? 0x80000000u
                                     : (denorm & 1u) ? 0x80ffffffu
                                                     : 0x80800000u}};
            cases.push_back(std::move(boundary));
          }
  return cases;
}

std::vector<ArithmeticCase> f16_fma_nan_cases() {
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (bool packed : {false, true})
      for (bool ieee : {false, true})
        for (bool clamp : {false, true})
          for (uint32_t denorm = 0; denorm < 4; ++denorm) {
            const bool modern =
                arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
            const auto add = [&](const char *name, uint16_t a, uint16_t b, uint16_t c,
                                 uint16_t nan) {
              const uint32_t expected =
                  clamp && modern ? 0u : nan | ((modern || ieee) ? 0x0200u : 0u);
              const auto pair = [packed](uint32_t value) {
                return packed ? value * 0x10001u : value;
              };
              cases.push_back({"Arch" + std::to_string(arch) + "Packed" + std::to_string(packed) +
                                   "Ieee" + std::to_string(ieee) + "Clamp" + std::to_string(clamp) +
                                   "Denorm" + std::to_string(denorm) + name,
                               arch,
                               {(packed ? 0xcc0e4006u : 0xd6480006u) | (clamp ? 0x8000u : 0u),
                                packed ? 0x1c0a0300u : 0x040a0300u, 0u},
                               {{0, pair(a)}, {1, pair(b)}, {2, pair(c)}, {6, 0u}},
                               {{6, pair(expected)}},
                               (uint32_t(ieee) << 9) | (denorm << 6),
                               FE_TONEAREST});
            };
            // Raw witnesses captured on gfx1100/gfx1201, including the GCC
            // optimization-dependent payload selection seen in packed FMA.
            add("FirstPayload", 0x7fc1u, 0xff80u, 0xff80u, 0x7fc1u);
            add("SignalingFirst", 0x7c01u, 0x3c00u, 0u, 0x7c01u);
            add("SignalingSecond", 0x3c00u, 0xfc12u, 0x7e01u, 0xfc12u);
            add("SignalingAddend", 0x3c00u, 0x3c00u, 0x7c01u, 0x7c01u);
            add("InvalidProductFirst", 0u, 0x7c00u, 0x7e01u, 0xfe00u);
            add("InvalidFiniteAddend", 0u, 0x7c00u, 0x3c00u, 0xfe00u);
            add("OppositeInfinities", 0x7c00u, 0x3c00u, 0xfc00u, 0xfe00u);
            add("FlushedProductFirst", 1u, 0x7c00u, 0x7e01u, denorm & 1u ? 0x7e01u : 0xfe00u);
          }
  return cases;
}

std::vector<ArithmeticCase> f16_fma_omod_cases() {
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (bool ieee : {false, true})
      for (bool overflow : {false, true})
        for (bool clamp : {false, true})
          for (uint32_t denorm = 0; denorm < 4; ++denorm) {
            const bool enabled = arch == ROCJITSU_CODE_ARCH_RDNA4 ||
                                 arch == ROCJITSU_CODE_ARCH_CDNA5 || (!ieee && !(denorm & 2u));
            const auto add = [&](const char *name, uint16_t a, uint16_t b, uint16_t c,
                                 uint32_t omod, uint16_t expected) {
              cases.push_back(
                  {"Arch" + std::to_string(arch) + "Ieee" + std::to_string(ieee) + "Overflow" +
                       std::to_string(overflow) + "Clamp" + std::to_string(clamp) + "Denorm" +
                       std::to_string(denorm) + name,
                   arch,
                   {0xd6480006u | (clamp ? 0x8000u : 0u), 0x040a0300u | (omod << 27), 0u},
                   {{0, a}, {1, b}, {2, c}, {6, 0u}},
                   {{6, expected}},
                   (uint32_t(ieee) << 9) | (uint32_t(overflow) << 23) | (denorm << 6),
                   FE_UPWARD});
            };
            // Both physical cards round the arithmetic before applying OMOD.
            add("NegativeUnderflow", 0x0400u, 0xbc00u, 0u, 3,
                clamp     ? 0u
                : enabled ? 0x8000u
                          : 0x8400u);
            add("OverflowBeforeDivision", 0x3c00u, 0x7bffu, 0x7bffu, 3,
                clamp      ? 0x3c00u
                : overflow ? (enabled ? 0x77ffu : 0x7bffu)
                           : 0x7c00u);
            add("TinyBeforeMultiplication", 0x0400u, 0x3800u, 0u, 2,
                enabled || !(denorm & 2u) ? 0u : 0x0200u);
            add("NegativeTinyBeforeMultiplication", 0x0400u, 0xb800u, 0u, 2,
                enabled || clamp ? 0u
                : (denorm & 2u)  ? 0x8200u
                                 : 0x8000u);
          }
  // Captured FMA and FMAC agree on both cards for every rounding mode.
  const uint16_t negative_tiny[] = {0x8000u, 0u, 0x8000u, 0u};
  const uint16_t positive_overflow[] = {0x77ffu, 0x7c00u, 0x77ffu, 0x77ffu};
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (bool accumulate : {false, true})
      for (uint32_t round = 0; round < 4; ++round) {
        const auto add = [&](const char *name, uint16_t a, uint16_t b, uint16_t c,
                             uint16_t expected) {
          cases.push_back(
              {"DirectedArch" + std::to_string(arch) + "Accumulate" + std::to_string(accumulate) +
                   "Round" + std::to_string(round) + name,
               arch,
               {accumulate ? 0xd5360006u : 0xd6480006u, accumulate ? 0x1a020300u : 0x1c0a0300u, 0u},
               {{0, a}, {1, b}, {2, c}, {6, c}},
               {{6, expected}},
               0x40u | (round << 2),
               FE_TOWARDZERO});
        };
        // RNE retains a negative normal until div:2 makes it tiny; RTZ first
        // flushes the arithmetic result, so the modifier receives a signed zero.
        add("NegativeUnderflow", 1u, 0x3400u, 0x8400u, negative_tiny[round]);
        add("OverflowBeforeDivision", 0x0400u, 0x0400u, 0x7bffu, positive_overflow[round]);
      }
  return cases;
}

std::vector<ArithmeticCase> host_mxcsr_cases() {
  std::vector<ArithmeticCase> cases;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  constexpr uint32_t kDazMask = 1u << 6;
  constexpr uint32_t kFtzMask = 1u << 15;
  constexpr uint32_t kControlMask = _MM_ROUND_MASK | kDazMask | kFtzMask;
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5})
    for (bool packed : {false, true})
      for (bool trap_invalid : {false, true}) {
        const auto pair = [packed](uint32_t value) { return packed ? value * 0x10001u : value; };
        const auto add = [&](const char *name, uint16_t a, uint16_t b, uint16_t c,
                             uint16_t expected) {
          cases.push_back(
              {"F16FmaArch" + std::to_string(arch) + "Packed" + std::to_string(packed) + "Trap" +
                   std::to_string(trap_invalid) + name,
               arch,
               {packed ? 0xcc0e4006u : 0xd6480006u, packed ? 0x1c0a0300u : 0x040a0300u, 0u},
               {{0, pair(a)}, {1, pair(b)}, {2, pair(c)}},
               {{6, packed ? pair(expected) : 0xdead0000u | expected}},
               0xf0u,
               FE_UPWARD,
               0xffffu,
               trap_invalid ? 0x1f00u : 0x1f80u});
        };
        // The residual must not expose Inf-Inf to the caller's exception state.
        add("InfiniteProduct", 0x7c00u, 0x3c00u, 0u, 0x7c00u);
        add("InfiniteAddend", 0x3c00u, 0x3c00u, 0x7c00u, 0x7c00u);
        add("InvalidProduct", 0u, 0x7c00u, 0x3c00u, 0xfe00u);
        add("InvalidSum", 0x7c00u, 0x3c00u, 0xfc00u, 0xfe00u);
      }
  // Change MXCSR rounding independently of the x87 rounding reported by fegetround().
  for (uint32_t rounding : {_MM_ROUND_UP, _MM_ROUND_DOWN, _MM_ROUND_TOWARD_ZERO}) {
    const bool upward = rounding == _MM_ROUND_UP;
    cases.push_back({"Cdna4AddRound" + std::to_string(rounding),
                     ROCJITSU_CODE_ARCH_CDNA4,
                     {0x020c0300u},
                     {{0, 0x3f800000u}, {1, upward ? 0x33800000u : 0x34400000u}},
                     {{6, upward ? 0x3f800000u : 0x3f800002u}},
                     0xf0u,
                     FE_TONEAREST,
                     kControlMask,
                     rounding});
  }
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    // V_DUAL_MUL_DX9_ZERO_F32 in both slots, writing v6 and v7.
    const std::array<uint32_t, 3> words{0xc9ce0300u, 0x06060702u};
    for (uint32_t daz : {0u, kDazMask}) {
      const std::string prefix =
          "VopdMulDx9Arch" + std::to_string(arch) + "Daz" + std::to_string(daz);
      const auto add = [&](const char *name, uint32_t lhs, uint32_t rhs, uint32_t expected,
                           uint32_t mode) {
        cases.push_back({prefix + name,
                         arch,
                         words,
                         {{0, lhs}, {1, rhs}, {2, lhs}, {3, rhs}},
                         {{6, expected}, {7, expected}},
                         mode,
                         FE_TONEAREST,
                         kControlMask,
                         daz});
      };
      add("Preserve", 0x00000001u, 0x4b000000u, 0x00800000u, 0xf0u);
      add("PreserveNegative", 0x80000001u, 0x4b000000u, 0x80800000u, 0xf0u);
      add("GuestFlush", 0x00000001u, 0x4b000000u, 0u, 0u);
      add("ZeroInf", 0u, 0x7f800000u, 0u, 0xf0u);
    }
  }
  // Also preserve pre-existing exception flags instead of merely clearing them.
  const size_t unflagged_cases = cases.size();
  for (size_t i = 0; i < unflagged_cases; ++i) {
    ArithmeticCase flagged = cases[i];
    flagged.name += "WithFlags";
    flagged.mxcsr_bits |= _MM_EXCEPT_INEXACT;
    cases.push_back(std::move(flagged));
  }
#endif
  return cases;
}

std::vector<ArithmeticCase> modifier_environment_cases() {
  std::vector<ArithmeticCase> cases;
  struct MulArch {
    rj_code_arch_t arch;
    uint32_t word;
  };
  // V_MUL_LEGACY_F32 on CDNA4, V_MUL_DX9_ZERO_F32 on RDNA3/4 and CDNA5.
  for (const MulArch &architecture : {MulArch{ROCJITSU_CODE_ARCH_CDNA4, 0xd2a10006u},
                                      MulArch{ROCJITSU_CODE_ARCH_RDNA3, 0xd5070006u},
                                      MulArch{ROCJITSU_CODE_ARCH_RDNA4, 0xd5070006u},
                                      MulArch{ROCJITSU_CODE_ARCH_CDNA5, 0xd5070006u}}) {
    for (uint32_t rounding = 0; rounding < 4; ++rounding) {
      for (bool negative : {false, true}) {
        for (bool clamp : {false, true}) {
          for (int host_rounding : {FE_TONEAREST, FE_TOWARDZERO}) {
            const uint32_t sign = negative ? 0x80000000u : 0u;
            const bool infinity =
                rounding == 0 || (rounding == 1 && !negative) || (rounding == 2 && negative);
            const uint32_t expected = clamp ? (negative ? 0u : 0x3f800000u)
                                            : sign | (infinity ? 0x7f800000u : 0x7f7fffffu);
            cases.push_back({"MulArch" + std::to_string(architecture.arch) + "Round" +
                                 std::to_string(rounding) + "Negative" + std::to_string(negative) +
                                 "Clamp" + std::to_string(clamp) + "Host" +
                                 std::to_string(host_rounding),
                             architecture.arch,
                             {architecture.word | (clamp ? 0x8000u : 0u), 0x0a020300u},
                             {{0, sign | 0x7f7fffffu}, {1, 0x3f800000u}},
                             {{6, expected}},
                             rounding,
                             host_rounding});
          }
        }
      }
    }
  }
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  constexpr uint32_t kDazMask = 1u << 6;
  constexpr uint32_t kControlMask = _MM_ROUND_MASK | kDazMask | (1u << 15);
  for (uint32_t daz : {0u, kDazMask}) {
    for (bool clamp : {false, true}) {
      for (bool negative : {false, true}) {
        const uint32_t sign = negative ? 0x80000000u : 0u;
        const std::string suffix = "Daz" + std::to_string(daz) + "Clamp" + std::to_string(clamp) +
                                   "Negative" + std::to_string(negative);
        cases.push_back({"LdexpF32" + suffix,
                         ROCJITSU_CODE_ARCH_CDNA4,
                         {0xd2880006u | (clamp ? 0x8000u : 0u), 0x00020300u},
                         {{0, sign | 0x00800000u}, {1, 0xffffffffu}},
                         {{6, negative && clamp ? 0u : sign | 0x00400000u}},
                         0xf0u,
                         FE_TONEAREST,
                         kControlMask,
                         daz});
        cases.push_back({"LdexpF64" + suffix,
                         ROCJITSU_CODE_ARCH_CDNA4,
                         {0xd2840006u | (clamp ? 0x8000u : 0u), 0x00020500u},
                         {{0, 0u}, {1, sign | 0x00100000u}, {2, 0xffffffffu}},
                         {{6, 0u}, {7, negative && clamp ? 0u : sign | 0x00080000u}},
                         0xf0u,
                         FE_TONEAREST,
                         kControlMask,
                         daz});
      }
    }
  }
#endif
  return cases;
}

std::vector<ArithmeticCase> fma_mix_f32_cases() {
  struct Sample {
    const char *name;
    uint32_t a, b, c, mode, expected;
  };
  // Raw gfx1100/gfx1201 captures; the older targets share the RDNA3 MODE policy.
  constexpr Sample samples[] = {
      {"RoundUp", 0x3f800000, 0x3f800000, 0x33800000, 1, 0x3f800001},
      {"RoundDown", 0xbf800000, 0x3f800000, 0xb3800000, 2, 0xbf800001},
      {"Nearest", 0x3f800000, 0x3f800000, 0x33800000, 0, 0x3f800000},
      {"TowardZero", 0x3f800000, 0x3f800000, 0x33800000, 3, 0x3f800000},
      {"InputFlush", 0x00000001, 0x4b000000, 0, 0, 0},
      {"InputPreserve", 0x00000001, 0x4b000000, 0, 0x30, 0x00800000},
      {"OutputFlush", 0x00800000, 0x3f000000, 0, 0, 0},
      {"OutputPreserve", 0x00800000, 0x3f000000, 0, 0x30, 0x00400000},
      {"NegativeZero", 0x3f800000, 0x3f800000, 0xbf800000, 2, 0x80000000},
      {"FusedCancellation", 0x3f800001, 0x3f7ffffe, 0xbf800000, 0, 0xa8800000},
      {"SignalingNan", 0x7f800012, 0x3f800000, 0x3f800000, 0, 0x7f800012},
      {"IeeeNan", 0x7f800012, 0x3f800000, 0x3f800000, 0x200, 0x7fc00012},
  };
  std::vector<ArithmeticCase> cases;
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (const auto &sample : samples) {
      uint32_t expected = sample.expected;
      if (expected == 0x7f800012 &&
          (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5))
        expected = 0x7fc00012;
      cases.push_back({"Arch" + std::to_string(arch) + sample.name,
                       arch,
                       {0xcc200006u, 0x040a0300u},
                       {{0, sample.a}, {1, sample.b}, {2, sample.c}},
                       {{6, expected}},
                       sample.mode,
                       FE_DOWNWARD,
                       0x9fc0u,
                       0x9f60u});
    }
  }
  return cases;
}

std::vector<ArithmeticCase> trig_fp_cases() {
  struct Target {
    const char *name;
    rj_code_arch_t arch;
    bool rdna_encoding;
    bool always_quiets;
  };
  const Target targets[] = {
      {"Cdna1", ROCJITSU_CODE_ARCH_CDNA1, false, false},
      {"Cdna2", ROCJITSU_CODE_ARCH_CDNA2, false, false},
      {"Cdna3", ROCJITSU_CODE_ARCH_CDNA3, false, false},
      {"Cdna4", ROCJITSU_CODE_ARCH_CDNA4, false, false},
      {"Rdna1", ROCJITSU_CODE_ARCH_RDNA1, true, false},
      {"Rdna2", ROCJITSU_CODE_ARCH_RDNA2, true, false},
      {"Rdna3", ROCJITSU_CODE_ARCH_RDNA3, true, false},
      {"Rdna35", ROCJITSU_CODE_ARCH_RDNA3_5, true, false},
      {"Rdna4", ROCJITSU_CODE_ARCH_RDNA4, true, true},
      {"Cdna5", ROCJITSU_CODE_ARCH_CDNA5, true, true},
  };
  std::vector<ArithmeticCase> cases;
  for (const auto &target : targets)
    for (unsigned cosine = 0; cosine < 2; ++cosine)
      for (unsigned e64 = 0; e64 < 2; ++e64) {
        // Assembled with llvm-mc for each target. The hardware captures use
        // gfx1100/gfx1201; the other targets check shared full-range and MODE
        // execution without claiming their finite approximations are identical.
        std::array<uint32_t, 3> words{};
        if (e64) {
          words[0] = (target.rdna_encoding ? 0xd5b50006u : 0xd1690006u) + cosine * 0x10000u;
          words[1] = target.rdna_encoding ? 0x02010100u : 0x00000100u;
        } else {
          words[0] = (target.rdna_encoding ? 0x7e0c6b00u : 0x7e0c5300u) + cosine * 0x200u;
        }
        const std::string prefix =
            std::string(target.name) + (cosine ? "Cos" : "Sin") + (e64 ? "E64" : "E32");
        auto add = [&](const std::string &name, uint32_t input, uint32_t expected, uint32_t mode) {
          cases.push_back({prefix + name,
                           target.arch,
                           words,
                           {{0, input}},
                           {{6, expected}},
                           mode,
                           FE_UPWARD,
                           0x8040u,
                           0x8040u});
        };
        add("LargeFinite", 0xff7fffffu, cosine ? 0x3f800000u : 0u, 240u);
        for (uint32_t ieee = 0; ieee < 2; ++ieee)
          add("SignalingNan" + std::to_string(ieee), 0xff812345u,
              target.always_quiets || ieee ? 0xffc12345u : 0xff812345u, 240u | (ieee << 9));
        for (uint32_t denorm = 0; denorm < 4; ++denorm)
          for (uint32_t rounding = 0; rounding < 4; ++rounding) {
            uint32_t mode = 192u | (denorm << 4) | rounding;
            add("Subnormal" + std::to_string(mode), 0x80000001u,
                cosine        ? 0x3f800000u
                : denorm == 3 ? 0x80000006u
                              : 0x80000000u,
                mode);
          }
        if (target.arch == ROCJITSU_CODE_ARCH_RDNA3 || target.arch == ROCJITSU_CODE_ARCH_RDNA4) {
          for (uint32_t rounding = 0; rounding < 4; ++rounding) {
            add("CapturedOctant" + std::to_string(rounding), 0x3e000000u,
                cosine ? 0x3f3504f3u : 0x3f3504f4u, 240u | rounding);
            // Raw captures distinguish the quadratic stages and the
            // normalized/reflected boundaries in both instruction encodings.
            const uint32_t captured[][3] = {
                {0x3aab9885u, 0x3c06c500u, 0x3f7ffdc8u}, {0x3d2001fbu, 0x3e78d2d0u, 0x3f7853c7u},
                {0x3dc084adu, 0x3f0e9072u, 0x3f54a13bu}, {0x3e7fff08u, 0x3f800000u, 0x37c2c761u},
                {0x3e7ffa50u, 0x3f800000u, 0x390ef149u},
            };
            for (const auto &sample : captured)
              add("CapturedStages" + std::to_string(sample[0]) + "Round" + std::to_string(rounding),
                  sample[0], sample[cosine + 1], 240u | rounding);
          }
        }
      }
  return cases;
}

std::vector<ArithmeticCase> log_exp_policy_cases() {
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
        ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
        ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool gcn = arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
                     arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
    for (bool logarithm : {false, true})
      for (bool e64 : {false, true}) {
        // Assembled independently for all ten profiles with llvm-mc.
        std::array<uint32_t, 3> words{};
        if (e64) {
          words[0] =
              (gcn ? 0xd1600006u : 0xd5a50006u) + (logarithm ? (gcn ? 0x10000u : 0x20000u) : 0u);
          words[1] = gcn ? 0x00000100u : 0x02010100u;
        } else {
          words[0] = (gcn ? 0x7e0c4100u : 0x7e0c4b00u) + (logarithm ? (gcn ? 0x200u : 0x400u) : 0u);
        }
        for (uint32_t ieee = 0; ieee < 2; ++ieee)
          for (uint32_t rounding = 0; rounding < 4; ++rounding) {
            const uint32_t mode = 240u | (ieee << 9) | rounding;
            const std::string prefix = "Arch" + std::to_string(arch) + (logarithm ? "Log" : "Exp") +
                                       (e64 ? "E64" : "E32") + "Mode" + std::to_string(mode);
            const auto add = [&](const char *name, uint32_t input, uint32_t expected) {
              // Preserve flags and FTZ/DAZ under upward rounding, with the
              // host invalid-operation trap enabled.
              cases.push_back({prefix + name,
                               arch,
                               words,
                               {{0, input}},
                               {{6, expected}},
                               mode,
                               FE_UPWARD,
                               0x9fc0u,
                               0x9f60u});
            };
            const bool quiet =
                ieee || arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
            add("SignalingNan", 0x7f812345u, quiet ? 0x7fc12345u : 0x7f812345u);
            add("NegativeSignalingNan", 0xff812345u, quiet ? 0xffc12345u : 0xff812345u);
            add("QuietNan", 0xffc12345u, 0xffc12345u);
            add("Negative", 0xbf800000u, logarithm ? 0xffc00000u : 0x3f000000u);
            add("NegativeInfinity", 0xff800000u, logarithm ? 0xffc00000u : 0u);
            add("NegativeSubnormal", 0x80000001u, logarithm ? 0xff800000u : 0x3f800000u);
            if (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA4)
              add("CapturedRounding", logarithm ? 0x3f174424u : 0x3f0567ecu,
                  logarithm ? 0xbf425164u : 0x3fb7b03du);
            if (!logarithm &&
                (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA4)) {
              const uint32_t captured[][2] = {
                  {0x337fffffu, 0x3f800000u}, {0x33800000u, 0x3f800000u},
                  {0x33800001u, 0x3f800000u}, {0xb37fffffu, 0x3f800000u},
                  {0xb3800000u, 0x3f7fffffu}, {0xb3800001u, 0x3f7fffffu},
                  {0x42ffffffu, 0x7f7fffa7u}, {0x43000000u, 0x7f800000u},
                  {0xc2fc0000u, 0x00800000u}, {0xc2fc0001u, 0x00000000u},
                  {0x3f0567ecu, 0x3fb7b03du}, {0xc114ed44u, 0x3acecc1eu},
                  {0x35500000u, 0x3f800004u}, {0x3e000090u, 0x3f8b95d0u},
                  {0x3e80001eu, 0x3f9837f6u}, {0x3ec0001au, 0x3fa5fedcu},
                  {0x3f000013u, 0x3fb504fcu}, {0x3f20002au, 0x3fc56740u},
                  {0x3f40006fu, 0x3fd7453eu}, {0x3f600022u, 0x3feac0dcu},
              };
              for (const auto &sample : captured) {
                const std::string name = "CapturedExp" + std::to_string(sample[0]);
                add(name.c_str(), sample[0], sample[1]);
              }
            }
            if (logarithm && e64) {
              // LOG always disables output denormals, so preservation MODE
              // cannot suppress scaling. IEEE mode still gates older profiles.
              for (uint32_t omod : {1u, 2u, 3u}) {
                auto scaled_words = words;
                scaled_words[1] |= omod << 27;
                const bool active =
                    !ieee || arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
                const uint32_t scaled[] = {0x40000000u, 0x40800000u, 0x41000000u, 0x3f800000u};
                for (uint32_t denorm = 0; denorm < 4; ++denorm)
                  cases.push_back({prefix + "LogScale" + std::to_string(omod) + "Denorm" +
                                       std::to_string(denorm),
                                   arch,
                                   scaled_words,
                                   {{0, 0x40800000u}},
                                   {{6, scaled[active ? omod : 0]}},
                                   192u | (ieee << 9) | (denorm << 4) | rounding,
                                   FE_UPWARD,
                                   0x9fc0u,
                                   0x9f60u});
              }
              // Modifiers run after the integer LOG mapping. Exercise both
              // preserved signaling NaNs and invalid-operation results while
              // host invalid traps, flags, rounding and flush controls persist.
              for (uint32_t dx10 : {0u, 1u})
                for (uint32_t modifier : {0u, 1u, 2u, 3u, 4u}) {
                  auto modified_words = words;
                  if (modifier == 4)
                    modified_words[0] |= 0x8000u;
                  else
                    modified_words[1] |= modifier << 27;
                  const bool nan_to_zero =
                      modifier == 4 && (dx10 || arch == ROCJITSU_CODE_ARCH_RDNA4 ||
                                        arch == ROCJITSU_CODE_ARCH_CDNA5);
                  for (uint32_t input : {0xbf800000u, 0x7f800123u}) {
                    const uint32_t result = input == 0xbf800000u ? 0xffc00000u
                                            : quiet              ? 0x7fc00123u
                                                                 : 0x7f800123u;
                    cases.push_back({prefix + "LogModifier" + std::to_string(modifier) + "Dx10" +
                                         std::to_string(dx10) + "Input" + std::to_string(input),
                                     arch,
                                     modified_words,
                                     {{0, input}},
                                     {{6, nan_to_zero ? 0u : result}},
                                     mode | (dx10 << 8),
                                     FE_UPWARD,
                                     0x9fc0u,
                                     0x9f60u});
                  }
                }
            }
            if (!logarithm && e64) {
              // OMOD overflows after EXP returns. Its rounding and exception
              // flags must remain independent of the host environment.
              for (uint32_t omod : {1u, 2u, 3u}) {
                auto scaled_words = words;
                scaled_words[1] |= omod << 27;
                for (uint32_t denorm = 0; denorm < 4; ++denorm) {
                  const bool active =
                      !ieee || arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
                  const uint32_t unscaled = omod == 3 ? 0x00800000u : 0x7f000000u;
                  const uint32_t expected = active ? (omod == 3 ? 0u : 0x7f800000u) : unscaled;
                  for (int host_round : {FE_DOWNWARD, FE_UPWARD})
                    cases.push_back({prefix + "Scale" + std::to_string(omod) + "Denorm" +
                                         std::to_string(denorm) + "Host" +
                                         std::to_string(host_round),
                                     arch,
                                     scaled_words,
                                     {{0, omod == 3 ? 0xc2fc0000u : 0x42fe0000u}},
                                     {{6, expected}},
                                     192u | (ieee << 9) | (denorm << 4) | rounding,
                                     host_round,
                                     0x9fc0u,
                                     0x9b60u});
                }
              }
            }
            {
              std::array<uint32_t, 3> half_words{};
              if (e64) {
                half_words[0] = gcn ? (logarithm ? 0xd1800006u : 0xd1810006u)
                                    : (logarithm ? 0xd5d70006u : 0xd5d80006u);
                half_words[1] = gcn ? 0x00000100u : 0x02010100u;
              } else {
                half_words[0] = gcn ? (logarithm ? 0x7e0c8100u : 0x7e0c8300u)
                                    : (logarithm ? 0x7e0caf00u : 0x7e0cb100u);
              }
              const bool preserves_high_half =
                  !gcn && arch != ROCJITSU_CODE_ARCH_RDNA1 && arch != ROCJITSU_CODE_ARCH_RDNA2;
              const uint32_t half_cases[][2] = {
                  {0x7c01u, quiet ? 0x7e01u : 0x7c01u},
                  {0xfc01u, quiet ? 0xfe01u : 0xfc01u},
                  {0xbc00u, logarithm ? 0xfe00u : 0x3800u},
                  {0xfc00u, logarithm ? 0xfe00u : 0u},
              };
              for (bool fp16_ovfl : {false, true})
                for (const auto &sample : half_cases)
                  cases.push_back({prefix + "Half" + std::to_string(sample[0]) + "Ovfl" +
                                       std::to_string(fp16_ovfl),
                                   arch,
                                   half_words,
                                   {{0, sample[0]}},
                                   {{6, sample[1] | (preserves_high_half ? 0xdead0000u : 0u)}},
                                   mode | (fp16_ovfl ? (1u << 23) : 0u),
                                   FE_UPWARD,
                                   0x9fc0u,
                                   0x9f60u});
            }
          }
      }
  }
  return cases;
}

std::vector<ArithmeticCase> sdwa_log_exp_policy_cases() {
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2}) {
    const bool gcn = arch != ROCJITSU_CODE_ARCH_RDNA1 && arch != ROCJITSU_CODE_ARCH_RDNA2;
    // The older manuals specify SDWA OMOD by reference to VOP3. These exact
    // values test that shared policy, not unmeasured hardware approximations.
    // The SDWA words were assembled independently for each profile with llvm-mc.
    for (bool logarithm : {false, true})
      for (uint32_t ieee = 0; ieee < 2; ++ieee)
        for (uint32_t denorm = 0; denorm < 4; ++denorm)
          for (uint32_t rounding = 0; rounding < 4; ++rounding)
            for (uint32_t omod : {1u, 2u, 3u}) {
              const uint32_t mode = 192u | (ieee << 9) | (denorm << 4) | rounding;
              const uint32_t opcode = gcn ? (logarithm ? 0x7e0c42f9u : 0x7e0c40f9u)
                                          : (logarithm ? 0x7e0c4ef9u : 0x7e0c4af9u);
              const std::array<uint32_t, 3> words{opcode, 0x00060600u | (omod << 14), 0};
              const uint32_t log_results[] = {0x40000000u, 0x40800000u, 0x41000000u, 0x3f800000u};
              const uint32_t input = logarithm   ? 0x40800000u
                                     : omod == 3 ? 0xc2fc0000u
                                                 : 0x42fe0000u;
              const uint32_t unscaled = omod == 3 ? 0x00800000u : 0x7f000000u;
              const uint32_t expected = logarithm   ? log_results[ieee ? 0 : omod]
                                        : ieee      ? unscaled
                                        : omod == 3 ? 0u
                                                    : 0x7f800000u;
              for (bool nan : {false, true})
                for (int host_round : {FE_DOWNWARD, FE_UPWARD})
                  cases.push_back({"Arch" + std::to_string(arch) + (logarithm ? "Log" : "Exp") +
                                       "Mode" + std::to_string(mode) + "Omod" +
                                       std::to_string(omod) + "Nan" + std::to_string(nan) + "Host" +
                                       std::to_string(host_round),
                                   arch,
                                   words,
                                   {{0, nan ? 0x7f812345u : input}},
                                   {{6, nan ? (ieee ? 0x7fc12345u : 0x7f812345u) : expected}},
                                   mode,
                                   host_round,
                                   0x9fc0u,
                                   0x9b60u});
            }
  }
  return cases;
}

std::vector<ArithmeticCase> half_log_exp_arithmetic_cases() {
  struct Captured {
    const char *name;
    bool logarithm;
    uint32_t source;
    uint32_t mode;
    uint32_t modifier;
    uint32_t rdna3;
    uint32_t rdna4;
  };
  // Raw gfx1100/gfx1201 captures. Modifier 4 is CLAMP; 1..3 are OMOD.
  const Captured captured[] = {
      {"LogPreserveInput", true, 0x0001u, 240u, 0u, 0xce00u, 0xce00u},
      {"LogFlushInput", true, 0x0001u, 48u, 0u, 0xfc00u, 0xfc00u},
      {"LogFlushSaturate", true, 0x0001u, 8388656u, 0u, 0xfbffu, 0xfbffu},
      {"LogZeroSaturate", true, 0x0000u, 8388848u, 0u, 0xfbffu, 0xfbffu},
      {"LogInputInfinity", true, 0x7c00u, 8388848u, 0u, 0x7c00u, 0x7c00u},
      {"ExpFiniteOverflow", false, 0x4c00u, 8388848u, 0u, 0x7bffu, 0x7bffu},
      {"ExpInputInfinity", false, 0x7c00u, 8388848u, 0u, 0x7c00u, 0x7c00u},
      {"ExpPreserveOutput", false, 0xcb80u, 240u, 0u, 0x0200u, 0x0200u},
      {"ExpFlushOutput", false, 0xcb80u, 112u, 0u, 0x0000u, 0x0000u},
      {"ExpScaleFlushedResult", false, 0xcb80u, 240u, 2u, 0x0200u, 0x0000u},
      {"ExpScaleSaturatedResult", false, 0x4c00u, 8388656u, 3u, 0x77ffu, 0x77ffu},
      {"ExpScaleOverflow", false, 0x4c00u, 48u, 3u, 0x7c00u, 0x7c00u},
      {"ExpScaleMode", false, 0x4c00u, 8388848u, 3u, 0x7bffu, 0x77ffu},
      {"LogNanClamp", true, 0xfc01u, 48u, 4u, 0xfc01u, 0x0000u},
      {"ExpNanScale", false, 0x7c01u, 48u, 1u, 0x7c01u, 0x7e01u},
      {"LogInvalidClamp", true, 0xbc00u, 48u, 4u, 0xfe00u, 0x0000u},
      {"ExpSingleRounding", false, 0x11c5u, 240u, 0u, 0x3c01u, 0x3c01u},
  };
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2, ROCJITSU_CODE_ARCH_CDNA3,
        ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RDNA1,
        ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
        ROCJITSU_CODE_ARCH_RDNA4}) {
    const bool gcn = arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
                     arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
    const bool true16 =
        !gcn && arch != ROCJITSU_CODE_ARCH_RDNA1 && arch != ROCJITSU_CODE_ARCH_RDNA2;
    const bool always_omod = arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
    for (const auto &sample : captured) {
      // Other profiles check exact values and MODE policy, not unmeasured finite approximations.
      if (sample.source == 0x11c5u && arch != ROCJITSU_CODE_ARCH_RDNA3 &&
          arch != ROCJITSU_CODE_ARCH_RDNA4)
        continue;
      for (bool e64 : {false, true}) {
        if (!e64 && sample.modifier != 0)
          continue;
        std::array<uint32_t, 3> words{};
        if (e64) {
          words[0] = gcn ? (sample.logarithm ? 0xd1800006u : 0xd1810006u)
                         : (sample.logarithm ? 0xd5d70006u : 0xd5d80006u);
          words[1] = gcn ? 0x00000100u : 0x02010100u;
          if (sample.modifier == 4)
            words[0] |= 0x8000u;
          else
            words[1] |= sample.modifier << 27;
        } else {
          words[0] = gcn ? (sample.logarithm ? 0x7e0c8100u : 0x7e0c8300u)
                         : (sample.logarithm ? 0x7e0caf00u : 0x7e0cb100u);
        }
        for (uint32_t rounding = 0; rounding < 4; ++rounding) {
          const uint32_t expected =
              (always_omod ? sample.rdna4 : sample.rdna3) | (true16 ? 0xdead0000u : 0u);
          cases.push_back({"Arch" + std::to_string(arch) + sample.name + (e64 ? "E64" : "E32") +
                               "Round" + std::to_string(rounding),
                           arch,
                           words,
                           {{0, sample.source}},
                           {{6, expected}},
                           sample.mode | (rounding << 2),
                           FE_UPWARD,
                           0x9fc0u,
                           0x9f60u});
        }
      }
    }
    if (gcn || arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2) {
      // Assembled SDWA forms select the high source and destination halves.
      // MODE and register-placement contracts also apply on these older profiles.
      for (uint32_t rounding = 0; rounding < 4; ++rounding) {
        for (bool signaling_nan : {false, true}) {
          const uint32_t source = signaling_nan ? 0x7c01u : 0x3c00u;
          const uint32_t result = signaling_nan ? 0x7c01u : 0x4400u;
          cases.push_back({"Arch" + std::to_string(arch) + "SdwaExpNan" +
                               std::to_string(signaling_nan) + "Round" + std::to_string(rounding),
                           arch,
                           {gcn ? 0x7e0c82f9u : 0x7e0cb0f9u, 0x00055500u},
                           {{0, (source << 16) | 0x1234u}},
                           {{6, (result << 16) | 0xbeefu}},
                           48u | (rounding << 2),
                           FE_UPWARD,
                           0x9fc0u,
                           0x9f60u});
        }
        cases.push_back(
            {"Arch" + std::to_string(arch) + "SdwaLogClampRound" + std::to_string(rounding),
             arch,
             {gcn ? 0x7e0c80f9u : 0x7e0caef9u, 0x00053500u},
             {{0, 0x7c011234u}},
             {{6, 0x0000beefu}},
             304u | (rounding << 2),
             FE_UPWARD,
             0x9fc0u,
             0x9f60u});
      }
    }
  }
  return cases;
}

std::vector<ArithmeticCase> fma_mix_half_cases() {
  std::vector<ArithmeticCase> cases;
  for (auto arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (uint32_t mode = 0; mode < 4; ++mode) {
      for (bool high : {false, true}) {
        // Physical RDNA3/4 shader exports: the exact product is just below
        // 0x340c, but rounding it to F32 first produces that exact F16 value.
        const uint32_t result = mode < 2 ? 0x340cu : 0x340bu;
        cases.push_back({"Arch" + std::to_string(arch) + "MixHalf" + std::to_string(high) +
                             "Round" + std::to_string(mode),
                         arch,
                         {high ? 0xcc224006u : 0xcc214006u, 0x820204ffu, 0x3b333333u},
                         {{2, 0x42b90000u}},
                         {{6, high ? (result << 16) | 0xbeefu : 0xdead0000u | result}},
                         240u | (mode << 2),
                         FE_UPWARD,
                         0x9fc0u,
                         0x9f60u});
      }
    }
    // Both cards prioritize an invalid product over the addend NaN. Ordinary
    // multiple-NaN inputs still select the first source's payload.
    for (bool high : {false, true}) {
      for (bool clamp : {false, true}) {
        for (uint32_t source = 0; source < 3; ++source) {
          const uint32_t a = source == 0 ? 0 : source == 1 ? 0x7f800000u : 0x7fc12000u;
          const uint32_t b = source == 0 ? 0x7f800000u : source == 1 ? 0 : 0x3f800000u;
          uint32_t result = source < 2 ? 0xfe00u : 0x7e09u;
          if (clamp && (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5))
            result = 0;
          cases.push_back(
              {"Arch" + std::to_string(arch) + "MixHalf" + std::to_string(high) + "Nan" +
                   std::to_string(source) + "Clamp" + std::to_string(clamp),
               arch,
               {(high ? 0xcc220006u : 0xcc210006u) | (clamp ? 0x8000u : 0), 0x040a0300u},
               {{0, a}, {1, b}, {2, 0x7fe9a000u}},
               {{6, high ? (result << 16) | 0xbeefu : 0xdead0000u | result}},
               0,
               FE_UPWARD,
               0x9fc0u,
               0x9f60u});
        }
      }
    }
  }
  return cases;
}

// Raw gfx1100/gfx1201 witnesses for instruction policies around the math mapping.
// The fixture checks sparse/full EXEC, inactive lanes and a hostile host environment.
std::vector<ArithmeticCase> transcendental_policy_cases() {
  struct Captured {
    const char *name;
    uint32_t opcode, input, mode, omod, rdna3, rdna4;
    bool half;
  };
  const Captured captured[] = {
      {"RcpZeroSaturate", 0xd5d40006u, 0x0000u, 0x8000c0u, 0, 0x7bffu, 0x7bffu, true},
      {"RsqNegativeZeroSaturate", 0xd5d60006u, 0x8000u, 0x8000c0u, 0, 0xfbffu, 0xfbffu, true},
      {"RcpFlushInput", 0xd5d40006u, 0x03ffu, 0, 0, 0x7c00u, 0x7c00u, true},
      {"RcpPreserveInput", 0xd5d40006u, 0x03ffu, 0xc0u, 0, 0x7401u, 0x7401u, true},
      {"RcpFlushOutput", 0xd5d40006u, 0x7bffu, 0, 0, 0x0000u, 0x0000u, true},
      {"RcpPreserveOutput", 0xd5d40006u, 0x7bffu, 0xc0u, 0, 0x0100u, 0x0100u, true},
      {"RcpHalfNan", 0xd5d40006u, 0x7c01u, 0, 0, 0x7c01u, 0x7e01u, true},
      {"RsqHalfNan", 0xd5d60006u, 0x7c01u, 0x200u, 0, 0x7e01u, 0x7e01u, true},
      {"SqrtHalfNanScale", 0xd5d50006u, 0x7c01u, 0, 1, 0x7c01u, 0x7e01u, true},
      {"SinHalfNan", 0xd5e00006u, 0x7c01u, 0, 0, 0x7c01u, 0x7e01u, true},
      {"CosHalfNan", 0xd5e10006u, 0x7c01u, 0, 0, 0x7c01u, 0x7e01u, true},
      {"SinHalfFlushInput", 0xd5e00006u, 0x0001u, 0, 0, 0x0000u, 0x0000u, true},
      {"SinHalfPreserveInput", 0xd5e00006u, 0x0001u, 0xc0u, 0, 0x0006u, 0x0006u, true},
      {"RcpNan", 0xd5aa0006u, 0x7f800001u, 0, 0, 0x7f800001u, 0x7fc00001u, false},
      {"RsqNan", 0xd5ae0006u, 0x7f800001u, 0, 0, 0x7f800001u, 0x7fc00001u, false},
      {"RcpScaleDenormMode", 0xd5aa0006u, 0x3f800000u, 0xf0u, 1, 0x40000000u, 0x40000000u, false},
      {"RsqScaleDenormMode", 0xd5ae0006u, 0x3f800000u, 0xf0u, 1, 0x40000000u, 0x40000000u, false},
      {"CosScaleDenormMode", 0xd5b60006u, 0x00000000u, 0xf0u, 1, 0x40000000u, 0x40000000u, false},
      {"RcpScaleOverflow", 0xd5aa0006u, 0x00800000u, 0, 2, 0x7f800000u, 0x7f800000u, false},
  };
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4})
    for (const auto &sample : captured)
      cases.push_back({"Arch" + std::to_string(arch) + sample.name,
                       arch,
                       {sample.opcode, 0x00000100u | (sample.omod << 27)},
                       {{0, sample.input}},
                       {{6, (arch == ROCJITSU_CODE_ARCH_RDNA3 ? sample.rdna3 : sample.rdna4) |
                                (sample.half ? 0xdead0000u : 0u)}},
                       sample.mode,
                       FE_DOWNWARD,
                       0x9fc0u,
                       0x9b60u});
  return cases;
}

std::vector<ArithmeticCase> minimum_maximum_cases() {
  // Expected results from gfx1201: preserve the first NaN's sign and payload
  // while quieting it, and honor input flushing only.
  // Instruction encodings from llvm-mc -mcpu=gfx1201.
  const auto make = [](std::string name, rj_code_arch_t arch, std::array<uint32_t, 3> words,
                       std::vector<std::pair<uint32_t, uint32_t>> sources,
                       std::vector<std::pair<uint32_t, uint32_t>> expected, uint32_t mode) {
    return ArithmeticCase{std::move(name),     arch, words,       std::move(sources),
                          std::move(expected), mode, FE_TONEAREST};
  };
  constexpr rj_code_arch_t RDNA4 = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr std::array<uint32_t, 3> MAXIMUM_F32{0xd7660006u, 0x00020300u, 0u};
  constexpr std::array<uint32_t, 3> MINIMUM_F32{0xd7650006u, 0x00020300u, 0u};
  constexpr std::array<uint32_t, 3> MAXIMUM_F64{0xd7420006u, 0x00020500u, 0u};
  constexpr std::array<uint32_t, 3> MINIMUM_F64{0xd7410006u, 0x00020500u, 0u};
  constexpr std::array<uint32_t, 3> MAXIMUM_F16{0xd7680006u, 0x00020300u, 0u};
  constexpr std::array<uint32_t, 3> MINIMUM_F16{0xd7670006u, 0x00020300u, 0u};
  return {
      make("MaximumF32QuietsSignalingSrc0", RDNA4, MAXIMUM_F32, {{0, 0x7f800001u}, {1, 0u}},
           {{6, 0x7fc00001u}}, 0xf0u),
      // A quiet src0 wins over a signaling src1 and keeps its sign.
      make("MaximumF32PrefersSrc0Nan", RDNA4, MAXIMUM_F32, {{0, 0xffc00000u}, {1, 0x7f800001u}},
           {{6, 0xffc00000u}}, 0xf0u),
      make("MinimumF32QuietsSrc1", RDNA4, MINIMUM_F32, {{0, 1u}, {1, 0x7f800001u}},
           {{6, 0x7fc00001u}}, 0xf0u),
      make("MinimumF32FlushesInput", RDNA4, MINIMUM_F32, {{0, 0x807fffffu}, {1, 0u}},
           {{6, 0x80000000u}}, 0x00u),
      make("MinimumF32KeepsInput", RDNA4, MINIMUM_F32, {{0, 0x807fffffu}, {1, 0u}},
           {{6, 0x807fffffu}}, 0xf0u),
      // Only the input-flush MODE bit affects selection.
      make("MaximumF32FlushInOnly", RDNA4, MAXIMUM_F32, {{0, 1u}, {1, 0u}}, {{6, 0u}}, 0x20u),
      make("MaximumF32FlushOutOnly", RDNA4, MAXIMUM_F32, {{0, 1u}, {1, 0u}}, {{6, 1u}}, 0x10u),
      make("MaximumF64QuietsSignalingSrc0", RDNA4, MAXIMUM_F64,
           {{0, 1u}, {1, 0x7ff00000u}, {2, 0u}, {3, 0u}}, {{6, 1u}, {7, 0x7ff80000u}}, 0xf0u),
      make("MinimumF64FlushesInput", RDNA4, MINIMUM_F64,
           {{0, 0xffffffffu}, {1, 0x800fffffu}, {2, 0u}, {3, 0u}}, {{6, 0u}, {7, 0x80000000u}},
           0x00u),
      make("MaximumF16QuietsSignalingSrc0", RDNA4, MAXIMUM_F16, {{0, 0x7c01u}, {1, 0u}, {6, 0u}},
           {{6, 0x7e01u}}, 0xf0u),
      make("MinimumF16QuietsSrc1", RDNA4, MINIMUM_F16, {{0, 1u}, {1, 0x7c01u}, {6, 0u}},
           {{6, 0x7e01u}}, 0xf0u),
      make("MinimumF16FlushesInput", RDNA4, MINIMUM_F16, {{0, 0x83ffu}, {1, 0u}, {6, 0u}},
           {{6, 0x8000u}}, 0x00u),
      // Three-source forms apply the first operation to src0/src1, then combine src2.
      make("Maximum3F32QuietsSrc2", RDNA4, {0xd62e0006u, 0x040a0300u, 0u},
           {{0, 0u}, {1, 0u}, {2, 0x7f800001u}}, {{6, 0x7fc00001u}}, 0xf0u),
      make("Minimum3F32FlushesSrc2", RDNA4, {0xd62d0006u, 0x040a0300u, 0u},
           {{0, 0u}, {1, 0u}, {2, 0x807fffffu}}, {{6, 0x80000000u}}, 0x00u),
      // The inner selection's NaN keeps its payload when src2 is also NaN.
      make("MaximumMinimumF32PrefersInnerNan", RDNA4, {0xd66d0006u, 0x040a0300u, 0u},
           {{0, 0u}, {1, 0x7f800001u}, {2, 0x7fc00000u}}, {{6, 0x7fc00001u}}, 0xf0u),
      make("MinimumMaximumF32FlushesAll", RDNA4, {0xd66c0006u, 0x040a0300u, 0u},
           {{0, 1u}, {1, 0u}, {2, 1u}}, {{6, 0u}}, 0x00u),
      make("Maximum3F16KeepsNegativeNan", RDNA4, {0xd6300006u, 0x040a0300u, 0u},
           {{0, 0xfd2du}, {1, 0x21c8u}, {2, 0xb723u}, {6, 0u}}, {{6, 0xff2du}}, 0xf0u),
      make("Minimum3F16FlushesInput", RDNA4, {0xd62f0006u, 0x040a0300u, 0u},
           {{0, 0x83ffu}, {1, 0u}, {2, 0u}, {6, 0u}}, {{6, 0x8000u}}, 0x00u),
      make("MaximumMinimumF16QuietsSrc2", RDNA4, {0xd66f0006u, 0x040a0300u, 0u},
           {{0, 0u}, {1, 0u}, {2, 0x7c01u}, {6, 0u}}, {{6, 0x7e01u}}, 0xf0u),
      make("MinimumMaximumF16FlushesSrc2", RDNA4, {0xd66e0006u, 0x040a0300u, 0u},
           {{0, 0u}, {1, 0u}, {2, 1u}, {6, 0u}}, {{6, 0u}}, 0x00u),
  };
}

std::vector<ArithmeticCase> min_max_num_cases() {
  // Expected results from gfx1201: *_NUM prefers numbers over NaNs and orders
  // -0 below +0. V_MED3_NUM uses V_MIN3_NUM whenever an input is NaN.
  // Instruction encodings from llvm-mc -mcpu=gfx1201.
  const auto rdna4 = [](std::string name, std::array<uint32_t, 3> words,
                        std::vector<std::pair<uint32_t, uint32_t>> sources,
                        std::vector<std::pair<uint32_t, uint32_t>> expected, uint32_t mode) {
    return ArithmeticCase{std::move(name),    ROCJITSU_CODE_ARCH_RDNA4, words,
                          std::move(sources), std::move(expected),      mode,
                          FE_TONEAREST};
  };
  constexpr std::array<uint32_t, 3> MIN_NUM_F32_E64{0xd5150006u, 0x00020300u, 0u};
  constexpr std::array<uint32_t, 3> MED3_NUM_F32{0xd6310006u, 0x040a0300u, 0u};
  constexpr std::array<uint32_t, 3> MINMAX_NUM_F32{0xd6680006u, 0x040a0300u, 0u};
  return {
      // v_max_num_f32_e32 v6, v0, v1
      rdna4("MaxNumF32OrdersSignedZero", {0x2c0c0300u, 0u, 0u}, {{0, 0x80000000u}, {1, 0u}},
            {{6, 0u}}, 0xf0u),
      rdna4("MinNumF32BothNanQuietsSrc0", MIN_NUM_F32_E64, {{0, 0x7f800001u}, {1, 0x7fc00000u}},
            {{6, 0x7fc00001u}}, 0xf0u),
      rdna4("MinNumF32IgnoresSignalingNan", MIN_NUM_F32_E64, {{0, 0x7f800001u}, {1, 0x3f800000u}},
            {{6, 0x3f800000u}}, 0xf0u),
      // v_max_num_f64_e32 v[6:7], v[0:1], v[2:3]
      rdna4("MaxNumF64FlushesInput", {0x1c0c0500u, 0u, 0u}, {{0, 1u}, {1, 0u}, {2, 0u}, {3, 0u}},
            {{6, 0u}, {7, 0u}}, 0x00u),
      // v_min_num_f16_e32 v6, v0, v1
      rdna4("MinNumF16BothNanKeepsSrc0Sign", {0x600c0300u, 0u, 0u},
            {{0, 0xfe00u}, {1, 0x7e00u}, {6, 0u}}, {{6, 0xfe00u}}, 0xf0u),
      rdna4("Max3NumF32FlushesSrc2", {0xd62a0006u, 0x040a0300u, 0u},
            {{0, 0x80000000u}, {1, 0x80000000u}, {2, 1u}}, {{6, 0u}}, 0x00u),
      rdna4("Min3NumF16OrdersSignedZero", {0xd62b0006u, 0x040a0300u, 0u},
            {{0, 0u}, {1, 0x8000u}, {2, 0u}, {6, 0u}}, {{6, 0x8000u}}, 0xf0u),
      rdna4("Med3NumF32NanSelectsMin3", MED3_NUM_F32, {{0, 0x7fc00000u}, {1, 1u}, {2, 0u}},
            {{6, 0u}}, 0xf0u),
      rdna4("Med3NumF32NegativeZeroMedian", MED3_NUM_F32,
            {{0, 0x80000000u}, {1, 0u}, {2, 0x80000000u}}, {{6, 0x80000000u}}, 0xf0u),
      rdna4("Med3NumF16FlushesInput", {0xd6320006u, 0x040a0300u, 0u},
            {{0, 1u}, {1, 0u}, {2, 1u}, {6, 0u}}, {{6, 0u}}, 0x00u),
      rdna4("MinMaxNumF32IgnoresInnerNanPair", MINMAX_NUM_F32,
            {{0, 0x7f800001u}, {1, 0x7fc00000u}, {2, 0x3f800000u}}, {{6, 0x3f800000u}}, 0xf0u),
      rdna4("MaxMinNumF16OrdersSignedZero", {0xd66b0006u, 0x040a0300u, 0u},
            {{0, 0x8000u}, {1, 0u}, {2, 0x3c00u}, {6, 0u}}, {{6, 0u}}, 0xf0u),
      // v_dual_max_num_f32 v6, v0, v1 :: v_dual_mov_b32 v7, v2
      // Expected from V_MAX_NUM_F32 rules; VOPD was not captured on hardware.
      rdna4("DualMaxNumF32OrdersSignedZero", {0xca900300u, 0x06060102u, 0u},
            {{0, 0x80000000u}, {1, 0u}, {2, 0x12345678u}}, {{6, 0u}, {7, 0x12345678u}}, 0xf0u),
  };
}

struct MinmaxForm {
  const char *name;
  uint16_t op;
  unsigned width;
  unsigned sources;
};

// Every floating min/max operation routed through the common VOP3 stages.
constexpr std::array<MinmaxForm, 30> kMinmaxForms = {{
    {"MinNumF16", rdna4::kVMinNumF16Vop3, 16, 2},
    {"MaxNumF16", rdna4::kVMaxNumF16Vop3, 16, 2},
    {"MinimumF16", rdna4::kVMinimumF16Vop3, 16, 2},
    {"MaximumF16", rdna4::kVMaximumF16Vop3, 16, 2},
    {"MinNumF32", rdna4::kVMinNumF32Vop3, 32, 2},
    {"MaxNumF32", rdna4::kVMaxNumF32Vop3, 32, 2},
    {"MinimumF32", rdna4::kVMinimumF32Vop3, 32, 2},
    {"MaximumF32", rdna4::kVMaximumF32Vop3, 32, 2},
    {"MinNumF64", rdna4::kVMinNumF64Vop3, 64, 2},
    {"MaxNumF64", rdna4::kVMaxNumF64Vop3, 64, 2},
    {"MinimumF64", rdna4::kVMinimumF64Vop3, 64, 2},
    {"MaximumF64", rdna4::kVMaximumF64Vop3, 64, 2},
    {"Min3NumF16", rdna4::kVMin3NumF16Vop3, 16, 3},
    {"Max3NumF16", rdna4::kVMax3NumF16Vop3, 16, 3},
    {"Minimum3F16", rdna4::kVMinimum3F16Vop3, 16, 3},
    {"Maximum3F16", rdna4::kVMaximum3F16Vop3, 16, 3},
    {"MinmaxNumF16", rdna4::kVMinmaxNumF16Vop3, 16, 3},
    {"MaxminNumF16", rdna4::kVMaxminNumF16Vop3, 16, 3},
    {"MinimummaximumF16", rdna4::kVMinimummaximumF16Vop3, 16, 3},
    {"MaximumminimumF16", rdna4::kVMaximumminimumF16Vop3, 16, 3},
    {"Med3NumF16", rdna4::kVMed3NumF16Vop3, 16, 3},
    {"Min3NumF32", rdna4::kVMin3NumF32Vop3, 32, 3},
    {"Max3NumF32", rdna4::kVMax3NumF32Vop3, 32, 3},
    {"Minimum3F32", rdna4::kVMinimum3F32Vop3, 32, 3},
    {"Maximum3F32", rdna4::kVMaximum3F32Vop3, 32, 3},
    {"MinmaxNumF32", rdna4::kVMinmaxNumF32Vop3, 32, 3},
    {"MaxminNumF32", rdna4::kVMaxminNumF32Vop3, 32, 3},
    {"MinimummaximumF32", rdna4::kVMinimummaximumF32Vop3, 32, 3},
    {"MaximumminimumF32", rdna4::kVMaximumminimumF32Vop3, 32, 3},
    {"Med3NumF32", rdna4::kVMed3NumF32Vop3, 32, 3},
}};

std::vector<ArithmeticCase> minmax_input_flush_cases() {
  // Explicit expectations keep this independent of the helper's MODE decoding.
  // Set F32 and F16/F64 differently, then vary each output-flush bit alone.
  struct ModeCase {
    uint32_t mode;
    bool keep_f32;
    bool keep_f16_f64;
  };
  constexpr std::array<ModeCase, 8> modes = {{
      {0xf0u, true, true},
      {0x00u, false, false},
      {0x30u, true, false},
      {0xc0u, false, true},
      {0xe0u, false, true},
      {0xd0u, true, true},
      {0xb0u, true, false},
      {0x70u, true, true},
  }};
  struct Form {
    const char *name;
    std::array<uint32_t, 3> words;
    unsigned width;
    bool minimum;
  };
  const auto vop2 = [](uint16_t op, unsigned width) {
    const auto words =
        rdna4::build_vop2(op, {.src0 = 256, .vsrc1 = uint8_t(width == 64 ? 2 : 1), .vdst = 6});
    return std::array<uint32_t, 3>{words[0], 0u, 0u};
  };
  const auto vop3 = [](uint16_t op, unsigned width) {
    const auto words = rdna4::build_vop3(
        op, {.vdst = 6, .src0 = 256, .src1 = uint16_t(width == 64 ? 258 : 257), .src2 = 258});
    return std::array<uint32_t, 3>{words[0], words[1], 0u};
  };
  const std::array<Form, 22> forms = {{
      {"MinNumF16E32", vop2(rdna4::kVMinNumF16Vop2, 16), 16, true},
      {"MaxNumF16E32", vop2(rdna4::kVMaxNumF16Vop2, 16), 16, false},
      {"MinNumF32E32", vop2(rdna4::kVMinNumF32Vop2, 32), 32, true},
      {"MaxNumF32E32", vop2(rdna4::kVMaxNumF32Vop2, 32), 32, false},
      {"MinNumF64E32", vop2(rdna4::kVMinNumF64Vop2, 64), 64, true},
      {"MaxNumF64E32", vop2(rdna4::kVMaxNumF64Vop2, 64), 64, false},
      {"MinNumF16E64", vop3(rdna4::kVMinNumF16Vop3, 16), 16, true},
      {"MaxNumF16E64", vop3(rdna4::kVMaxNumF16Vop3, 16), 16, false},
      {"MinNumF32E64", vop3(rdna4::kVMinNumF32Vop3, 32), 32, true},
      {"MaxNumF32E64", vop3(rdna4::kVMaxNumF32Vop3, 32), 32, false},
      {"MinNumF64E64", vop3(rdna4::kVMinNumF64Vop3, 64), 64, true},
      {"MaxNumF64E64", vop3(rdna4::kVMaxNumF64Vop3, 64), 64, false},
      {"MinimumF16", vop3(rdna4::kVMinimumF16Vop3, 16), 16, true},
      {"MaximumF16", vop3(rdna4::kVMaximumF16Vop3, 16), 16, false},
      {"MinimumF32", vop3(rdna4::kVMinimumF32Vop3, 32), 32, true},
      {"MaximumF32", vop3(rdna4::kVMaximumF32Vop3, 32), 32, false},
      {"MinimumF64", vop3(rdna4::kVMinimumF64Vop3, 64), 64, true},
      {"MaximumF64", vop3(rdna4::kVMaximumF64Vop3, 64), 64, false},
      // Ternary forms use separate SIMD helpers from the binary forms.
      {"Min3NumF16", vop3(rdna4::kVMin3NumF16Vop3, 16), 16, true},
      {"Max3NumF16", vop3(rdna4::kVMax3NumF16Vop3, 16), 16, false},
      {"Min3NumF32", vop3(rdna4::kVMin3NumF32Vop3, 32), 32, true},
      {"Max3NumF32", vop3(rdna4::kVMax3NumF32Vop3, 32), 32, false},
  }};
  std::vector<ArithmeticCase> cases;
  for (const Form &form : forms)
    for (const ModeCase &mode : modes) {
      const bool keep = form.width == 32 ? mode.keep_f32 : mode.keep_f16_f64;
      // min(-tiny, +0) is -tiny or -0; max(+tiny, +0) is +tiny or +0.
      // The ternary cases add another +0 and have the same expected result.
      const uint32_t sign = form.minimum ? (form.width == 16 ? 0x8000u : 0x80000000u) : 0u;
      std::vector<std::pair<uint32_t, uint32_t>> sources;
      std::vector<std::pair<uint32_t, uint32_t>> expected;
      if (form.width == 64) {
        sources = {{0, 1u}, {1, sign}, {2, 0u}, {3, 0u}};
        expected = {{6, keep ? 1u : 0u}, {7, sign}};
      } else {
        // Poison the unused F16 halves, and check that the destination's high
        // half survives. A read of the wrong source half would select a NaN.
        const uint32_t high = form.width == 16 ? 0x7e000000u : 0u;
        sources = {{0, high | sign | 1u}, {1, high}, {2, high}};
        expected = {{6, (form.width == 16 ? 0xdead0000u : 0u) | sign | (keep ? 1u : 0u)}};
      }
      constexpr char HEX[] = "0123456789abcdef";
      const std::string suffix = {HEX[mode.mode >> 4], HEX[mode.mode & 0xfu]};
      cases.push_back({std::string(form.name) + "Mode" + suffix, ROCJITSU_CODE_ARCH_RDNA4,
                       form.words, std::move(sources), std::move(expected), mode.mode,
                       FE_TONEAREST});
    }
  // Equal inputs make every binary, nested and median form return that input.
  // Test both signs with OMOD disabled so its zeroing cannot mask a missed flush.
  for (const MinmaxForm &form : kMinmaxForms)
    for (const ModeCase &mode : modes)
      for (const bool negative : {false, true}) {
        const bool keep = form.width == 32 ? mode.keep_f32 : mode.keep_f16_f64;
        const uint64_t sign = negative ? uint64_t{1} << (form.width - 1) : 0;
        const uint64_t input = sign | 1u;
        const uint64_t result = sign | (keep ? 1u : 0u);
        std::vector<std::pair<uint32_t, uint32_t>> sources;
        std::vector<std::pair<uint32_t, uint32_t>> expected;
        if (form.width == 64) {
          sources = {{0, uint32_t(input)},
                     {1, uint32_t(input >> 32)},
                     {2, uint32_t(input)},
                     {3, uint32_t(input >> 32)}};
          expected = {{6, uint32_t(result)}, {7, uint32_t(result >> 32)}};
        } else {
          const uint32_t high = form.width == 16 ? 0x7e000000u : 0u;
          for (unsigned source = 0; source < form.sources; ++source)
            sources.emplace_back(source, high | uint32_t(input));
          expected = {{6, (form.width == 16 ? 0xdead0000u : 0u) | uint32_t(result)}};
        }
        cases.push_back({std::string(form.name) + "AllSources" +
                             (negative ? "Negative" : "Positive") + "Mode" +
                             std::to_string(mode.mode),
                         ROCJITSU_CODE_ARCH_RDNA4, vop3(form.op, form.width), std::move(sources),
                         std::move(expected), mode.mode, FE_TONEAREST});
      }
  return cases;
}

std::vector<ArithmeticCase> minmax_output_modifier_cases() {
  // Expected bits come from gfx1201 captures with OMOD/CLAMP. The directed
  // overflow cases were captured for all 30 min/max forms on 2026-10-02.
  // Equal inputs let every form reuse the same output-stage expectations.
  // The final samples explicitly check the order of combined modifiers.
  struct Sample {
    const char *name;
    uint8_t omod;
    bool clamp;
    uint32_t mode;
    uint64_t value;
    uint64_t expected;
    bool abs = false;
    bool neg = false;
  };
  constexpr std::array<Sample, 30> f16 = {{
      {"Mul2Subnormal", 1, false, 0xf0u, 0x0001u, 0x0000u},
      {"Mul2NegativeZero", 1, false, 0xf0u, 0x8000u, 0x0000u},
      {"Div2NegativeMinNormal", 3, false, 0xf0u, 0x8400u, 0x8000u},
      {"Mul4", 2, false, 0xf0u, 0x3c00u, 0x4400u},
      {"Div2", 3, false, 0xf0u, 0x3c00u, 0x3800u},
      {"Mul2OverflowNearest", 1, false, 0xf0u, 0x7bffu, 0x7c00u},
      {"Mul4OverflowTowardZero", 2, false, 0xffu, 0xfbffu, 0xfbffu},
      {"Mul2OverflowFp16Ovfl", 1, false, 0x8000f0u, 0x7bffu, 0x7bffu},
      // gfx1201 overflow captures: round toward +infinity / -infinity.
      {"Mul2PositiveOverflowRoundUp", 1, false, 0xf5u, 0x7bffu, 0x7c00u},
      {"Mul2PositiveOverflowRoundUpFp16Ovfl", 1, false, 0x8000f5u, 0x7bffu, 0x7bffu},
      {"Mul2PositiveOverflowRoundDown", 1, false, 0xfau, 0x7bffu, 0x7bffu},
      {"Mul2PositiveOverflowRoundDownFp16Ovfl", 1, false, 0x8000fau, 0x7bffu, 0x7bffu},
      {"Mul2NegativeOverflowRoundUp", 1, false, 0xf5u, 0xfbffu, 0xfbffu},
      {"Mul2NegativeOverflowRoundUpFp16Ovfl", 1, false, 0x8000f5u, 0xfbffu, 0xfbffu},
      {"Mul2NegativeOverflowRoundDown", 1, false, 0xfau, 0xfbffu, 0xfc00u},
      {"Mul2NegativeOverflowRoundDownFp16Ovfl", 1, false, 0x8000fau, 0xfbffu, 0xfbffu},
      {"Mul4PositiveOverflowRoundUp", 2, false, 0xf5u, 0x7bffu, 0x7c00u},
      {"Mul4PositiveOverflowRoundUpFp16Ovfl", 2, false, 0x8000f5u, 0x7bffu, 0x7bffu},
      {"Mul4PositiveOverflowRoundDown", 2, false, 0xfau, 0x7bffu, 0x7bffu},
      {"Mul4PositiveOverflowRoundDownFp16Ovfl", 2, false, 0x8000fau, 0x7bffu, 0x7bffu},
      {"Mul4NegativeOverflowRoundUp", 2, false, 0xf5u, 0xfbffu, 0xfbffu},
      {"Mul4NegativeOverflowRoundUpFp16Ovfl", 2, false, 0x8000f5u, 0xfbffu, 0xfbffu},
      {"Mul4NegativeOverflowRoundDown", 2, false, 0xfau, 0xfbffu, 0xfc00u},
      {"Mul4NegativeOverflowRoundDownFp16Ovfl", 2, false, 0x8000fau, 0xfbffu, 0xfbffu},
      {"Mul2Nan", 1, false, 0xf0u, 0x7e00u, 0x7e00u},
      {"ClampNan", 0, true, 0xf0u, 0x7e00u, 0x0000u},
      {"ClampAboveOne", 0, true, 0xf0u, 0x7bffu, 0x3c00u},
      {"ClampNegative", 0, true, 0xf0u, 0x83ffu, 0x0000u},
      // abs(-0.75) * 2 clamps to 1; -abs(-0.75) * 2 is -1.5.
      {"AbsMul2Clamp", 1, true, 0xf0u, 0xba00u, 0x3c00u, true},
      {"AbsNegMul2", 1, false, 0xf0u, 0xba00u, 0xbe00u, true, true},
  }};
  constexpr std::array<Sample, 30> f32 = {{
      {"Mul2Subnormal", 1, false, 0xf0u, 0x00000001u, 0x00000000u},
      {"Mul2NegativeZero", 1, false, 0xf0u, 0x80000000u, 0x00000000u},
      {"Div2NegativeMinNormal", 3, false, 0xf0u, 0x80800000u, 0x80000000u},
      {"Mul4", 2, false, 0xf0u, 0x3f800000u, 0x40800000u},
      {"Div2", 3, false, 0xf0u, 0x3f800000u, 0x3f000000u},
      {"Mul2OverflowNearest", 1, false, 0xf0u, 0x7f7fffffu, 0x7f800000u},
      {"Mul4OverflowTowardZero", 2, false, 0xffu, 0xff7fffffu, 0xff7fffffu},
      {"Mul2OverflowFp16Ovfl", 1, false, 0x8000f0u, 0x7f7fffffu, 0x7f800000u},
      // gfx1201 overflow captures: round toward +infinity / -infinity.
      {"Mul2PositiveOverflowRoundUp", 1, false, 0xf5u, 0x7f7fffffu, 0x7f800000u},
      {"Mul2PositiveOverflowRoundUpFp16Ovfl", 1, false, 0x8000f5u, 0x7f7fffffu, 0x7f800000u},
      {"Mul2PositiveOverflowRoundDown", 1, false, 0xfau, 0x7f7fffffu, 0x7f7fffffu},
      {"Mul2PositiveOverflowRoundDownFp16Ovfl", 1, false, 0x8000fau, 0x7f7fffffu, 0x7f7fffffu},
      {"Mul2NegativeOverflowRoundUp", 1, false, 0xf5u, 0xff7fffffu, 0xff7fffffu},
      {"Mul2NegativeOverflowRoundUpFp16Ovfl", 1, false, 0x8000f5u, 0xff7fffffu, 0xff7fffffu},
      {"Mul2NegativeOverflowRoundDown", 1, false, 0xfau, 0xff7fffffu, 0xff800000u},
      {"Mul2NegativeOverflowRoundDownFp16Ovfl", 1, false, 0x8000fau, 0xff7fffffu, 0xff800000u},
      {"Mul4PositiveOverflowRoundUp", 2, false, 0xf5u, 0x7f7fffffu, 0x7f800000u},
      {"Mul4PositiveOverflowRoundUpFp16Ovfl", 2, false, 0x8000f5u, 0x7f7fffffu, 0x7f800000u},
      {"Mul4PositiveOverflowRoundDown", 2, false, 0xfau, 0x7f7fffffu, 0x7f7fffffu},
      {"Mul4PositiveOverflowRoundDownFp16Ovfl", 2, false, 0x8000fau, 0x7f7fffffu, 0x7f7fffffu},
      {"Mul4NegativeOverflowRoundUp", 2, false, 0xf5u, 0xff7fffffu, 0xff7fffffu},
      {"Mul4NegativeOverflowRoundUpFp16Ovfl", 2, false, 0x8000f5u, 0xff7fffffu, 0xff7fffffu},
      {"Mul4NegativeOverflowRoundDown", 2, false, 0xfau, 0xff7fffffu, 0xff800000u},
      {"Mul4NegativeOverflowRoundDownFp16Ovfl", 2, false, 0x8000fau, 0xff7fffffu, 0xff800000u},
      {"Mul2Nan", 1, false, 0xf0u, 0x7fc00000u, 0x7fc00000u},
      {"ClampNan", 0, true, 0xf0u, 0x7fc00000u, 0x00000000u},
      {"ClampAboveOne", 0, true, 0xf0u, 0x7f7fffffu, 0x3f800000u},
      {"ClampNegative", 0, true, 0xf0u, 0x807fffffu, 0x00000000u},
      {"AbsMul2Clamp", 1, true, 0xf0u, 0xbf400000u, 0x3f800000u, true},
      {"AbsNegMul2", 1, false, 0xf0u, 0xbf400000u, 0xbfc00000u, true, true},
  }};
  constexpr std::array<Sample, 30> f64 = {{
      {"Mul2Subnormal", 1, false, 0xf0u, 0x1u, 0x0u},
      {"Mul2NegativeZero", 1, false, 0xf0u, 0x8000000000000000u, 0x0u},
      {"Div2NegativeMinNormal", 3, false, 0xf0u, 0x8010000000000000u, 0x8000000000000000u},
      {"Mul4", 2, false, 0xf0u, 0x3ff0000000000000u, 0x4010000000000000u},
      {"Div2", 3, false, 0xf0u, 0x3ff0000000000000u, 0x3fe0000000000000u},
      {"Mul2OverflowNearest", 1, false, 0xf0u, 0x7fefffffffffffffu, 0x7ff0000000000000u},
      {"Mul4OverflowTowardZero", 2, false, 0xffu, 0xffefffffffffffffu, 0xffefffffffffffffu},
      {"Mul2OverflowFp16Ovfl", 1, false, 0x8000f0u, 0x7fefffffffffffffu, 0x7ff0000000000000u},
      // gfx1201 overflow captures: round toward +infinity / -infinity.
      {"Mul2PositiveOverflowRoundUp", 1, false, 0xf5u, 0x7fefffffffffffffu, 0x7ff0000000000000u},
      {"Mul2PositiveOverflowRoundUpFp16Ovfl", 1, false, 0x8000f5u, 0x7fefffffffffffffu,
       0x7ff0000000000000u},
      {"Mul2PositiveOverflowRoundDown", 1, false, 0xfau, 0x7fefffffffffffffu, 0x7fefffffffffffffu},
      {"Mul2PositiveOverflowRoundDownFp16Ovfl", 1, false, 0x8000fau, 0x7fefffffffffffffu,
       0x7fefffffffffffffu},
      {"Mul2NegativeOverflowRoundUp", 1, false, 0xf5u, 0xffefffffffffffffu, 0xffefffffffffffffu},
      {"Mul2NegativeOverflowRoundUpFp16Ovfl", 1, false, 0x8000f5u, 0xffefffffffffffffu,
       0xffefffffffffffffu},
      {"Mul2NegativeOverflowRoundDown", 1, false, 0xfau, 0xffefffffffffffffu, 0xfff0000000000000u},
      {"Mul2NegativeOverflowRoundDownFp16Ovfl", 1, false, 0x8000fau, 0xffefffffffffffffu,
       0xfff0000000000000u},
      {"Mul4PositiveOverflowRoundUp", 2, false, 0xf5u, 0x7fefffffffffffffu, 0x7ff0000000000000u},
      {"Mul4PositiveOverflowRoundUpFp16Ovfl", 2, false, 0x8000f5u, 0x7fefffffffffffffu,
       0x7ff0000000000000u},
      {"Mul4PositiveOverflowRoundDown", 2, false, 0xfau, 0x7fefffffffffffffu, 0x7fefffffffffffffu},
      {"Mul4PositiveOverflowRoundDownFp16Ovfl", 2, false, 0x8000fau, 0x7fefffffffffffffu,
       0x7fefffffffffffffu},
      {"Mul4NegativeOverflowRoundUp", 2, false, 0xf5u, 0xffefffffffffffffu, 0xffefffffffffffffu},
      {"Mul4NegativeOverflowRoundUpFp16Ovfl", 2, false, 0x8000f5u, 0xffefffffffffffffu,
       0xffefffffffffffffu},
      {"Mul4NegativeOverflowRoundDown", 2, false, 0xfau, 0xffefffffffffffffu, 0xfff0000000000000u},
      {"Mul4NegativeOverflowRoundDownFp16Ovfl", 2, false, 0x8000fau, 0xffefffffffffffffu,
       0xfff0000000000000u},
      {"Mul2Nan", 1, false, 0xf0u, 0x7ff8000000000000u, 0x7ff8000000000000u},
      {"ClampNan", 0, true, 0xf0u, 0x7ff8000000000000u, 0x0u},
      {"ClampAboveOne", 0, true, 0xf0u, 0x7fefffffffffffffu, 0x3ff0000000000000u},
      {"ClampNegative", 0, true, 0xf0u, 0x800fffffffffffffu, 0x0u},
      {"AbsMul2Clamp", 1, true, 0xf0u, 0xbfe8000000000000u, 0x3ff0000000000000u, true},
      {"AbsNegMul2", 1, false, 0xf0u, 0xbfe8000000000000u, 0xbff8000000000000u, true, true},
  }};
  std::vector<ArithmeticCase> cases;
  for (const MinmaxForm &form : kMinmaxForms) {
    const auto &samples = form.width == 16 ? f16 : form.width == 32 ? f32 : f64;
    for (const Sample &sample : samples) {
      const auto words =
          rdna4::build_vop3(form.op, {.vdst = 6,
                                      .abs = uint8_t(sample.abs ? (1u << form.sources) - 1 : 0u),
                                      .clamp = uint8_t(sample.clamp),
                                      .src0 = 256,
                                      .src1 = uint16_t(form.width == 64 ? 258 : 257),
                                      .src2 = 258,
                                      .omod = sample.omod,
                                      .neg = uint8_t(sample.neg ? (1u << form.sources) - 1 : 0u)});
      std::vector<std::pair<uint32_t, uint32_t>> sources;
      std::vector<std::pair<uint32_t, uint32_t>> expected;
      if (form.width == 64) {
        sources = {{0, uint32_t(sample.value)},
                   {1, uint32_t(sample.value >> 32)},
                   {2, uint32_t(sample.value)},
                   {3, uint32_t(sample.value >> 32)}};
        expected = {{6, uint32_t(sample.expected)}, {7, uint32_t(sample.expected >> 32)}};
      } else {
        // The F16 destination keeps its high half.
        const uint32_t high = form.width == 16 ? 0x7e000000u : 0u;
        sources = {{0, high | uint32_t(sample.value)},
                   {1, high | uint32_t(sample.value)},
                   {2, high | uint32_t(sample.value)}};
        expected = {{6, (form.width == 16 ? 0xdead0000u : 0u) | uint32_t(sample.expected)}};
      }
      cases.push_back({std::string(form.name) + sample.name,
                       ROCJITSU_CODE_ARCH_RDNA4,
                       {words[0], words[1], 0u},
                       std::move(sources),
                       std::move(expected),
                       sample.mode,
                       FE_TONEAREST});
    }
  }
  return cases;
}

// VOP3 integral rounding completes before the shared OMOD/CLAMP stages.
// These are explicit wiring expectations, not additional hardware captures.
std::vector<ArithmeticCase> integral_rounding_modifier_cases() {
  struct Form {
    const char *name;
    uint16_t op;
    unsigned width;
    double positive_halved;  // round(1.5) / 2
    double negative_doubled; // round(-abs(-1.5)) * 2
  };
  constexpr std::array<Form, 8> forms = {{
      {"CeilF32", rdna4::kVCeilF32Vop3, 32, 1.0, -2.0},
      {"FloorF32", rdna4::kVFloorF32Vop3, 32, 0.5, -4.0},
      {"TruncF32", rdna4::kVTruncF32Vop3, 32, 0.5, -2.0},
      {"RndneF32", rdna4::kVRndneF32Vop3, 32, 1.0, -4.0},
      {"CeilF64", rdna4::kVCeilF64Vop3, 64, 1.0, -2.0},
      {"FloorF64", rdna4::kVFloorF64Vop3, 64, 0.5, -4.0},
      {"TruncF64", rdna4::kVTruncF64Vop3, 64, 0.5, -2.0},
      {"RndneF64", rdna4::kVRndneF64Vop3, 64, 1.0, -4.0},
  }};
  std::vector<ArithmeticCase> cases;
  for (const auto &form : forms) {
    const auto bits = [&](double value) -> uint64_t {
      return form.width == 32 ? std::bit_cast<uint32_t>(static_cast<float>(value))
                              : std::bit_cast<uint64_t>(value);
    };
    const uint64_t sign = form.width == 32 ? 0x80000000u : 0x8000000000000000ull;
    const uint64_t infinity = form.width == 32 ? 0x7f800000u : 0x7ff0000000000000ull;
    const uint64_t quiet = form.width == 32 ? 0x00400000u : 0x0008000000000000ull;
    const auto add = [&](const std::string &name, uint64_t input, uint64_t result,
                         rdna4::Vop3BuilderFields fields, uint32_t mode = 0xf0u,
                         int host_rounding = FE_TONEAREST) {
      fields.vdst = 6;
      fields.src0 = 256;
      const auto words = rdna4::build_vop3(form.op, fields);
      std::vector<std::pair<uint32_t, uint32_t>> sources{{0, uint32_t(input)}};
      std::vector<std::pair<uint32_t, uint32_t>> expected{{6, uint32_t(result)}};
      if (form.width == 64) {
        sources.emplace_back(1, uint32_t(input >> 32));
        expected.emplace_back(7, uint32_t(result >> 32));
      }
      cases.push_back({std::string(form.name) + name,
                       ROCJITSU_CODE_ARCH_RDNA4,
                       {words[0], words[1], 0u},
                       std::move(sources),
                       std::move(expected),
                       mode,
                       host_rounding});
    };
    add("RoundThenHalf", bits(1.5), bits(form.positive_halved), {.omod = 3});
    add("AbsNegThenRound", bits(-1.5), bits(form.negative_doubled),
        {.abs = 1, .omod = 1, .neg = 1});
    add("ScaleThenClamp", bits(2.0), bits(1.0), {.clamp = 1, .omod = 3});
    add("PreserveNegativeZero", sign, sign, {});
    add("ScaleNegativeZero", sign, 0u, {.omod = 1});
    add("QuietNanBeforeScale", infinity | 0x42u, infinity | quiet | 0x42u, {.omod = 2});
    add("ClampNan", infinity | quiet | 0x42u, 0u, {.clamp = 1, .omod = 1});
    // MODE flushes the tiny input first, so every form gives +0: on gfx1201
    // ceil(+0) * 2 is +0, not 2.
    add("TinyInput", 1u, 0u, {.omod = 1}, 0u);

    for (uint32_t rounding = 0; rounding < 4; ++rounding) {
      // Give the other format a different rounding mode to catch field mixups.
      const uint32_t other = (rounding + 1) % 4;
      const uint32_t mode =
          0xf0u | (form.width == 32 ? rounding | (other << 2) : other | (rounding << 2));
      const int host = rounding == 0 ? FE_TOWARDZERO : FE_TONEAREST;
      const uint64_t positive = rounding == 0 || rounding == 1 ? infinity : infinity - 1;
      const uint64_t negative = rounding == 0 || rounding == 2 ? infinity : infinity - 1;
      add("PositiveOverflowMode" + std::to_string(rounding), infinity - 1, positive, {.omod = 1},
          mode, host);
      add("NegativeOverflowMode" + std::to_string(rounding), sign | (infinity - 1), sign | negative,
          {.omod = 1}, mode, host);
    }
  }
  return cases;
}

// gfx1201 applies OMOD, then CLAMP, after rounding a result to its destination
// format. Each case is a captured gfx1201 lane. F16 destinations keep the
// 0xa5a5 high half the capture initialized them with.
std::vector<ArithmeticCase> rounded_result_modifier_cases() {
  constexpr uint32_t kHigh = 0xa5a5a5a5u;
  constexpr uint16_t V0 = 256, V1 = 257, V2 = 258, V4 = 260;
  std::vector<ArithmeticCase> cases;
  const auto add = [&](const std::string &name, uint16_t op, rdna4::Vop3BuilderFields fields,
                       std::vector<std::pair<uint32_t, uint32_t>> sources,
                       std::vector<std::pair<uint32_t, uint32_t>> expected, uint32_t mode) {
    fields.vdst = 6;
    const auto words = rdna4::build_vop3(op, fields);
    cases.push_back({name,
                     ROCJITSU_CODE_ARCH_RDNA4,
                     {words[0], words[1], 0u},
                     std::move(sources),
                     std::move(expected),
                     mode,
                     FE_TONEAREST});
  };
  const auto f16 = [&](const std::string &name, uint16_t op, uint8_t omod, uint32_t a, uint32_t b,
                       uint32_t result, uint32_t mode) {
    add(name, op, {.src0 = V0, .src1 = V1, .omod = omod}, {{0, a}, {1, b}, {6, kHigh}},
        {{6, result}}, mode);
  };
  const auto f64 = [&](const std::string &name, uint16_t op, rdna4::Vop3BuilderFields fields,
                       std::vector<uint64_t> inputs, uint64_t result, uint32_t mode) {
    std::vector<std::pair<uint32_t, uint32_t>> sources;
    for (uint32_t i = 0; i < inputs.size(); ++i) {
      sources.emplace_back(2 * i, uint32_t(inputs[i]));
      sources.emplace_back(2 * i + 1, uint32_t(inputs[i] >> 32));
    }
    add(name, op, fields, std::move(sources), {{6, uint32_t(result)}, {7, uint32_t(result >> 32)}},
        mode);
  };

  // A zero or subnormal rounded result becomes +0; halving -min_normal keeps
  // its sign. The emulator previously scaled the wide sum before narrowing.
  f16("AddF16Mul2SubnormalSum", rdna4::kVAddF16Vop3, 1, 0x966883ffu, 0xe8b70000u, 0xa5a50000u,
      0xf0u);
  f16("AddF16Mul4SubnormalSum", rdna4::kVAddF16Vop3, 2, 0xae7c8981u, 0xac15077du, 0xa5a50000u, 0u);
  f16("AddF16Div2NegativeMinNormal", rdna4::kVAddF16Vop3, 3, 0x3c548400u, 0xffc00000u, 0xa5a58000u,
      0xffu);
  f16("LdexpF16Div2NegativeMinNormal", rdna4::kVLdexpF16Vop3, 3, 0x954b8400u, 0x35cd0000u,
      0xa5a58000u, 0u);
  f16("LdexpF16Mul4SubnormalResult", rdna4::kVLdexpF16Vop3, 2, 0x63c80400u, 0xde3dffffu,
      0xa5a50000u, 0u);
  f64("AddF64Div2NegativeMinNormal", rdna4::kVAddF64Vop3, {.src0 = V0, .src1 = V2, .omod = 3},
      {0x8010000000000000u, 0u}, 0x8000000000000000u, 0xf0u);
  f64("AddF64Mul4SubnormalSum", rdna4::kVAddF64Vop3, {.src0 = V0, .src1 = V2, .omod = 2},
      {0x800fffffffffffffu, 0u}, 0u, 0xffu);

  // Integral rounding of a half: OMOD overflow follows MODE, here toward zero.
  f16("TruncF16Mul4OverflowTowardZero", rdna4::kVTruncF16Vop3, 2, 0x6a017bffu, 0u, 0xa5a57bffu,
      0xffu);
  f16("RndneF16Mul2OverflowTowardZero", rdna4::kVRndneF16Vop3, 1, 0x2d4c7bffu, 0u, 0xa5a57bffu,
      0xffu);

  // TRANS results: halving -min_normal keeps its sign, and OMOD overflow gives
  // infinity even under round-toward-zero.
  f16("RcpF16Div2NegativeMinNormal", rdna4::kVRcpF16Vop3, 3, 0xf8c2f226u, 0u, 0xa5a58000u, 0xf0u);
  f16("SinF16Div2NegativeMinNormal", rdna4::kVSinF16Vop3, 3, 0xcafc813cu, 0u, 0xa5a58000u, 0xffu);
  f16("RcpF16Mul4OverflowTowardZero", rdna4::kVRcpF16Vop3, 2, 0xb0d70400u, 0u, 0xa5a57c00u, 0xffu);
  f16("RcpF16Mul4NegativeOverflowTowardZero", rdna4::kVRcpF16Vop3, 2, 0x474983ffu, 0u, 0xa5a5fc00u,
      0xffu);

  // Integer-to-F64 conversions take OMOD and CLAMP on the converted value.
  f64("CvtF64I32Clamp", rdna4::kVCvtF64I32Vop3, {.clamp = 1, .src0 = V0}, {0xffffffffu}, 0u, 0xf0u);
  f64("CvtF64I32Div2", rdna4::kVCvtF64I32Vop3, {.src0 = V0, .omod = 3}, {1u}, 0x3fe0000000000000u,
      0u);
  f64("CvtF64U32Mul2", rdna4::kVCvtF64U32Vop3, {.src0 = V0, .omod = 1}, {1u}, 0x4000000000000000u,
      0xffu);

  // DIV_FMAS scales its rounded result, then clamps. VCC is clear, so the
  // fused result is not rescaled.
  add("DivFmasF32Mul2", rdna4::kVDivFmasF32Vop3, {.src0 = V0, .src1 = V1, .src2 = V2, .omod = 1},
      {{0, 0x7f7fffffu}, {1, 1u}, {2, 0u}}, {{6, 0x357fffffu}}, 0xf0u);
  add("DivFmasF32ClampNan", rdna4::kVDivFmasF32Vop3,
      {.clamp = 1, .src0 = V0, .src1 = V1, .src2 = V2}, {{0, 0x7f800000u}, {1, 0u}, {2, 0u}},
      {{6, 0u}}, 0u);
  f64("DivFmasF64Mul2NegativeSubnormal", rdna4::kVDivFmasF64Vop3,
      {.src0 = V0, .src1 = V2, .src2 = V4, .omod = 1}, {0x800fffffffffffffu, 1u, 0u}, 0u, 0xffu);
  f64("DivFmasF64ClampNan", rdna4::kVDivFmasF64Vop3,
      {.clamp = 1, .src0 = V0, .src1 = V2, .src2 = V4},
      {0x7ff0000000000000u, 0x8000000000000000u, 0u}, 0u, 0xf0u);
  return cases;
}

// CEIL and FLOOR flush a subnormal source to a signed zero when MODE disables
// input denormals, so ceil(+tiny) = +0 and floor(-tiny) = -0. Results are
// gfx1201 captures. MODE 0x30 keeps F32 input denormals and flushes F16/F64;
// 0xc0 does the reverse, so each case catches a read of the wrong field.
std::vector<ArithmeticCase> integral_rounding_input_flush_cases() {
  struct Form {
    const char *name;
    uint16_t vop1;
    uint16_t vop3;
    unsigned width;
    bool ceil;
  };
  constexpr std::array<Form, 6> forms = {{
      {"CeilF16", rdna4::kVCeilF16Vop1, rdna4::kVCeilF16Vop3, 16, true},
      {"FloorF16", rdna4::kVFloorF16Vop1, rdna4::kVFloorF16Vop3, 16, false},
      {"CeilF32", rdna4::kVCeilF32Vop1, rdna4::kVCeilF32Vop3, 32, true},
      {"FloorF32", rdna4::kVFloorF32Vop1, rdna4::kVFloorF32Vop3, 32, false},
      {"CeilF64", rdna4::kVCeilF64Vop1, rdna4::kVCeilF64Vop3, 64, true},
      {"FloorF64", rdna4::kVFloorF64Vop1, rdna4::kVFloorF64Vop3, 64, false},
  }};
  constexpr uint32_t kHigh = 0xa5a50000u;
  std::vector<ArithmeticCase> cases;
  for (const Form &form : forms) {
    const uint64_t sign = uint64_t{1} << (form.width - 1);
    const uint64_t largest_subnormal = (uint64_t{1} << (form.width == 16   ? 10
                                                        : form.width == 32 ? 23
                                                                           : 52)) -
                                       1;
    const uint64_t one = form.width == 16   ? 0x3c00u
                         : form.width == 32 ? 0x3f800000u
                                            : 0x3ff0000000000000u;
    const bool f32_field = form.width == 32;
    const auto add = [&](const std::string &name, std::array<uint32_t, 2> words, uint64_t input,
                         uint32_t mode, bool flushed, uint64_t unflushed) {
      // ceil(+tiny) and floor(-tiny) round to +-1 unless the input is flushed.
      const uint64_t result = flushed ? (form.ceil ? 0u : sign) : unflushed;
      std::vector<std::pair<uint32_t, uint32_t>> sources{{0, uint32_t(input)}};
      std::vector<std::pair<uint32_t, uint32_t>> expected{{6, uint32_t(result)}};
      if (form.width == 16) {
        sources = {{0, kHigh | uint32_t(input)}, {6, kHigh}};
        expected = {{6, kHigh | uint32_t(result)}};
      } else if (form.width == 64) {
        sources.emplace_back(1, uint32_t(input >> 32));
        expected.emplace_back(7, uint32_t(result >> 32));
      }
      cases.push_back({std::string(form.name) + name,
                       ROCJITSU_CODE_ARCH_RDNA4,
                       {words[0], words[1], 0u},
                       std::move(sources),
                       std::move(expected),
                       mode,
                       FE_TONEAREST});
    };
    const uint64_t tiny = form.ceil ? 1u : sign | largest_subnormal;
    const uint64_t rounded = form.ceil ? one : sign | one;
    const auto vop1 = rdna4::build_vop1(form.vop1, {.src0 = 256, .vdst = 6});
    const auto vop3 = rdna4::build_vop3(form.vop3, {.vdst = 6, .src0 = 256});
    // NEG turns the opposite-signed tiny input into the one that rounds away.
    const auto vop3_neg = rdna4::build_vop3(form.vop3, {.vdst = 6, .src0 = 256, .neg = 1});
    for (const uint32_t mode : {0x30u, 0xc0u}) {
      const bool flushed = (mode == 0xc0u) == f32_field;
      const std::string suffix = mode == 0x30u ? "Mode30" : "ModeC0";
      add("Vop1" + suffix, {vop1[0], 0u}, tiny, mode, flushed, rounded);
      add("Vop3" + suffix, vop3, tiny, mode, flushed, rounded);
      add("Vop3Neg" + suffix, vop3_neg, tiny ^ sign, mode, flushed, rounded);
    }
  }
  return cases;
}

// An RDNA4 case with F64 operands in v[0:1], v[2:3] and v[4:5] and the result in v[6:7].
ArithmeticCase f64_case(const std::string &name, std::array<uint32_t, 2> words,
                        std::array<uint64_t, 3> sources, uint32_t mode, uint64_t result) {
  std::vector<std::pair<uint32_t, uint32_t>> registers;
  for (uint32_t i = 0; i < sources.size(); ++i) {
    registers.emplace_back(2 * i, uint32_t(sources[i]));
    registers.emplace_back(2 * i + 1, uint32_t(sources[i] >> 32));
  }
  return {name,
          ROCJITSU_CODE_ARCH_RDNA4,
          {words[0], words[1], 0u},
          std::move(registers),
          {{6, uint32_t(result)}, {7, uint32_t(result >> 32)}},
          mode,
          FE_TONEAREST};
}

// gfx1201 detects tininess after rounding: a result that rounds to the
// smallest normal only on the subnormal grid is still tiny, and a flushing
// output (or an active OMOD) turns it into zero. Results are gfx1201
// captures. MODE 0x30 flushes F16/F64 outputs and keeps F32 ones; 0xc0 does
// the reverse, so each pair catches a read of the wrong MODE field.
std::vector<ArithmeticCase> tiny_result_cases() {
  constexpr uint32_t kHigh = 0xa5a50000u;
  std::vector<ArithmeticCase> cases;
  // F16 operands in v0/v1; v6 keeps its 0xa5a5 high half.
  const auto add_f16 = [&](const std::string &name, std::array<uint32_t, 2> words, uint16_t a,
                           uint16_t b, uint32_t mode, uint16_t result) {
    cases.push_back({name,
                     ROCJITSU_CODE_ARCH_RDNA4,
                     {words[0], words[1], 0u},
                     {{0, kHigh | a}, {1, kHigh | b}, {6, kHigh}},
                     {{6, kHigh | result}},
                     mode,
                     FE_TONEAREST});
  };
  const auto mul_f16_vop2 =
      rdna4::build_vop2(rdna4::kVMulF16Vop2, {.src0 = 256, .vsrc1 = 1, .vdst = 6});
  const std::array<uint32_t, 2> mul_f16_e32{mul_f16_vop2[0], 0u};
  const auto mul_f16 =
      rdna4::build_vop3(rdna4::kVMulF16Vop3, {.vdst = 6, .src0 = 256, .src1 = 257});
  const auto mul_f16_mul2 =
      rdna4::build_vop3(rdna4::kVMulF16Vop3, {.vdst = 6, .src0 = 256, .src1 = 257, .omod = 1});
  // (1 - 2^-11) * 2^-14 is exact in F16 precision, so it is tiny in every
  // rounding mode although nearest-even subnormal rounding gives 0x0400.
  add_f16("MulF16Vop2TinyMode30", mul_f16_e32, 0x3bffu, 0x0400u, 0x30u, 0x0000u);
  add_f16("MulF16Vop2TinyModeC0", mul_f16_e32, 0x3bffu, 0x0400u, 0xc0u, 0x0400u);
  add_f16("MulF16Vop2NegativeTinyMode30", mul_f16_e32, 0x3bffu, 0x8400u, 0x30u, 0x8000u);
  add_f16("MulF16Vop2NegativeTinyModeC0", mul_f16_e32, 0x3bffu, 0x8400u, 0xc0u, 0x8400u);
  add_f16("MulF16Vop3TinyRoundUp", mul_f16, 0x3bffu, 0x0400u, 0x35u, 0x0000u);
  // (1 - 2^-20) * 2^-14 rounds to 2^-14 at F16 precision: not tiny, so it
  // survives the flush, except when rounding toward zero.
  add_f16("MulF16Vop3RoundsToNormalMode30", mul_f16, 0x3bfeu, 0x0401u, 0x30u, 0x0400u);
  add_f16("MulF16Vop3RoundsToNormalTowardZero", mul_f16, 0x3bfeu, 0x0401u, 0x3fu, 0x0000u);
  // OMOD flushes a tiny result even when MODE keeps output denormals.
  add_f16("MulF16Vop3Mul2TinyModeF0", mul_f16_mul2, 0x3bffu, 0x0400u, 0xf0u, 0x0000u);
  add_f16("MulF16Vop3Mul2RoundsToNormalModeF0", mul_f16_mul2, 0x3bfeu, 0x0401u, 0xf0u, 0x0800u);

  // F32 operands in v0/v1, result in v6.
  const auto add_f32 = [&](const std::string &name, std::array<uint32_t, 2> words, uint32_t a,
                           uint32_t b, uint32_t mode, uint32_t result) {
    cases.push_back({name,
                     ROCJITSU_CODE_ARCH_RDNA4,
                     {words[0], words[1], 0u},
                     {{0, a}, {1, b}},
                     {{6, result}},
                     mode,
                     FE_TONEAREST});
  };
  const auto dx9_vop2 =
      rdna4::build_vop2(rdna4::kVMulDx9ZeroF32Vop2, {.src0 = 256, .vsrc1 = 1, .vdst = 6});
  const std::array<uint32_t, 2> dx9_e32{dx9_vop2[0], 0u};
  const auto dx9 =
      rdna4::build_vop3(rdna4::kVMulDx9ZeroF32Vop3, {.vdst = 6, .src0 = 256, .src1 = 257});
  const auto dx9_mul2 = rdna4::build_vop3(rdna4::kVMulDx9ZeroF32Vop3,
                                          {.vdst = 6, .src0 = 256, .src1 = 257, .omod = 1});
  // (1 - 2^-24) * 2^-126 is exact in F32 precision: tiny, although
  // nearest-even subnormal rounding gives 0x00800000.
  add_f32("MulDx9F32Vop2TinyModeC0", dx9_e32, 0x3f7fffffu, 0x00800000u, 0xc0u, 0u);
  add_f32("MulDx9F32Vop2TinyMode30", dx9_e32, 0x3f7fffffu, 0x00800000u, 0x30u, 0x00800000u);
  add_f32("MulDx9F32Vop2NegativeTinyModeC0", dx9_e32, 0x3f7fffffu, 0x80800000u, 0xc0u, 0x80000000u);
  // MODE 0xd0 keeps the subnormal input but flushes the tiny product.
  add_f32("MulDx9F32Vop2SubnormalInputModeD0", dx9_e32, 0x3fffffffu, 0x00400000u, 0xd0u, 0u);
  add_f32("MulDx9F32Vop3Mul2TinyModeF0", dx9_mul2, 0x3f7fffffu, 0x00800000u, 0xf0u, 0u);
  // (1 - 2^-46) * 2^-126 rounds to 2^-126 at F32 precision: not tiny.
  add_f32("MulDx9F32Vop3Mul2RoundsToNormalModeF0", dx9_mul2, 0x3f7ffffeu, 0x00800001u, 0xf0u,
          0x01000000u);
  // A zero operand gives +0 even with a NaN or infinite partner.
  add_f32("MulDx9F32Vop3NegativeZeroTimesNanModeC0", dx9, 0x80000000u, 0x7fc00000u, 0xc0u, 0u);
  add_f32("MulDx9F32Vop3InfinityTimesZeroModeF0", dx9, 0xff800000u, 0u, 0xf0u, 0u);

  const auto add_f64 = [&](const std::string &name, std::array<uint32_t, 2> words,
                           std::array<uint64_t, 3> sources, uint32_t mode, uint64_t result) {
    cases.push_back(f64_case(name, words, sources, mode, result));
  };
  constexpr uint64_t kMinNormal64 = 0x0010000000000000u;
  constexpr uint64_t kSign64 = 0x8000000000000000u;
  constexpr uint64_t kOneMinusUlp64 = 0x3fefffffffffffffu; // 1 - 2^-53
  const auto mul_f64_vop2 =
      rdna4::build_vop2(rdna4::kVMulF64Vop2, {.src0 = 256, .vsrc1 = 2, .vdst = 6});
  const std::array<uint32_t, 2> mul_f64_e32{mul_f64_vop2[0], 0u};
  const auto mul_f64 =
      rdna4::build_vop3(rdna4::kVMulF64Vop3, {.vdst = 6, .src0 = 256, .src1 = 258});
  const auto mul_f64_mul2 =
      rdna4::build_vop3(rdna4::kVMulF64Vop3, {.vdst = 6, .src0 = 256, .src1 = 258, .omod = 1});
  add_f64("MulF64Vop2TinyMode30", mul_f64_e32, {kOneMinusUlp64, kMinNormal64, 0}, 0x30u, 0u);
  add_f64("MulF64Vop2TinyModeC0", mul_f64_e32, {kOneMinusUlp64, kMinNormal64, 0}, 0xc0u,
          kMinNormal64);
  add_f64("MulF64Vop2NegativeTinyMode30", mul_f64_e32, {kOneMinusUlp64, kSign64 | kMinNormal64, 0},
          0x30u, kSign64);
  // (1 - 2^-104) * 2^-1022 rounds to 2^-1022 at full precision, except
  // toward -infinity (MODE 0x7a).
  add_f64("MulF64Vop3RoundsToNormalMode30", mul_f64, {0x3feffffffffffffeu, kMinNormal64 + 1, 0},
          0x30u, kMinNormal64);
  add_f64("MulF64Vop3RoundsToNormalTowardNegative", mul_f64,
          {0x3feffffffffffffeu, kMinNormal64 + 1, 0}, 0x7au, 0u);
  // MODE 0xf0 is native, so this also covers the SIMD path's OMOD check.
  add_f64("MulF64Vop3Mul2TinyModeF0", mul_f64_mul2, {kOneMinusUlp64, kMinNormal64, 0}, 0xf0u, 0u);

  const auto fma_f64 =
      rdna4::build_vop3(rdna4::kVFmaF64Vop3, {.vdst = 6, .src0 = 256, .src1 = 258, .src2 = 260});
  const auto fma_f64_mul2 = rdna4::build_vop3(
      rdna4::kVFmaF64Vop3, {.vdst = 6, .src0 = 256, .src1 = 258, .src2 = 260, .omod = 1});
  add_f64("FmaF64TinyMode30", fma_f64, {kOneMinusUlp64, kMinNormal64, 0}, 0x30u, 0u);
  add_f64("FmaF64TinyModeC0", fma_f64, {kOneMinusUlp64, kMinNormal64, 0}, 0xc0u, kMinNormal64);
  // -(2^-1022 - 2^-1074) minus a tiny product rounds toward -infinity to the
  // smallest normal on the subnormal grid, but stays tiny at full precision.
  add_f64("FmaF64TinyTowardNegative", fma_f64,
          {0x800fffffffffffffu, kMinNormal64, 0x800fffffffffffffu}, 0x7au, kSign64);
  add_f64("FmaF64SubnormalModeF0", fma_f64,
          {0x800fffffffffffffu, kMinNormal64, 0x800fffffffffffffu}, 0xf0u, 0x800fffffffffffffu);
  add_f64("FmaF64Mul2TinyModeF0", fma_f64_mul2, {kOneMinusUlp64, kMinNormal64, 0}, 0xf0u, 0u);
  return cases;
}

// V_FMA_F64 NaN selection and output modifiers, from gfx1201 captures. An
// invalid product (0 * infinity, after MODE input flushing) gives the default
// NaN even when the addend is a NaN; otherwise the first NaN source is quieted.
// OMOD scales the rounded result like every other output modifier.
std::vector<ArithmeticCase> fma_f64_policy_cases() {
  constexpr uint64_t kInfinity = 0x7ff0000000000000u;
  constexpr uint64_t kDefaultNan = 0xfff8000000000000u;
  constexpr uint64_t kOne = 0x3ff0000000000000u;
  const auto fma = [](uint8_t omod) {
    return rdna4::build_vop3(rdna4::kVFmaF64Vop3,
                             {.vdst = 6, .src0 = 256, .src1 = 258, .src2 = 260, .omod = omod});
  };
  return {
      f64_case("ZeroTimesInfinitySignalingAddend", fma(0), {0u, kInfinity, 0x7ff4000000000001u},
               0xf0u, kDefaultNan),
      f64_case("InfinityTimesZeroQuietAddend", fma(0),
               {0xfff0000000000000u, 0x8000000000000000u, 0x7ff8000000000005u}, 0x00u, kDefaultNan),
      // MODE 0x30 flushes the F64 subnormal input, making the product invalid.
      f64_case("FlushedSubnormalTimesInfinityMode30", fma(0), {kInfinity, 1u, 0x7ff0000000000001u},
               0x30u, kDefaultNan),
      f64_case("SubnormalTimesInfinityModeF0", fma(0), {kInfinity, 1u, 0x7ff0000000000001u}, 0xf0u,
               0x7ff8000000000001u),
      f64_case("FirstNanIsQuieted", fma(0),
               {0x7ff4000000000007u, 0x7ff8000000000009u, 0x7ff800000000000bu}, 0xf0u,
               0x7ffc000000000007u),
      f64_case("SecondNanBeforeAddend", fma(0), {kOne, 0x7ff4000000000007u, 0x7ff800000000000bu},
               0x00u, 0x7ffc000000000007u),
      // Halving a normal with the smallest exponent gives a zero of its sign.
      f64_case("Div2NegativeUnderflow", fma(3), {0x8018000000000000u, kOne, 0u}, 0xf0u,
               0x8000000000000000u),
      f64_case("Div2PositiveUnderflow", fma(3), {0x0018000000000000u, kOne, 0u}, 0x00u, 0u),
      f64_case("Mul2Negative", fma(1), {0x8018000000000000u, kOne, 0u}, 0xf0u, 0x8028000000000000u),
      // A subnormal result becomes +0 under OMOD, also when MODE keeps it.
      f64_case("Div2Subnormal", fma(3), {0x000fffffffffffffu, kOne, 0u}, 0xf0u, 0u),
  };
}

// V_DIV_FMAS judges tininess after rounding like other arithmetic, including
// a post-scaled (VCC) result and under OMOD. Results are gfx1201 captures.
std::vector<ArithmeticCase> div_fmas_cases() {
  const auto fmas_f32 = [](uint8_t omod) {
    return rdna4::build_vop3(rdna4::kVDivFmasF32Vop3,
                             {.vdst = 6, .src0 = 256, .src1 = 257, .src2 = 258, .omod = omod});
  };
  const auto f32_case = [](const std::string &name, std::array<uint32_t, 2> words,
                           std::array<uint32_t, 3> sources, uint32_t mode, uint32_t result) {
    return ArithmeticCase{name,
                          ROCJITSU_CODE_ARCH_RDNA4,
                          {words[0], words[1], 0u},
                          {{0, sources[0]}, {1, sources[1]}, {2, sources[2]}},
                          {{6, result}},
                          mode,
                          FE_TONEAREST};
  };
  const auto fmas_f64 = rdna4::build_vop3(rdna4::kVDivFmasF64Vop3,
                                          {.vdst = 6, .src0 = 256, .src1 = 258, .src2 = 260});
  // -(2^-126 - 2^-149) minus a tiny product, rounded toward -infinity (MODE
  // 0x?a), reaches the smallest normal only on the subnormal grid.
  const std::array<uint32_t, 3> f32_boundary{0x807fffffu, 0x00000001u, 0x807fffffu};
  const std::array<uint64_t, 3> f64_boundary{0x800fffffffffffffu, 0x0010000000000000u,
                                             0x800fffffffffffffu};
  std::vector<ArithmeticCase> cases{
      f32_case("F32TinyTowardNegativeModeCa", fmas_f32(0), f32_boundary, 0xcau, 0x80000000u),
      f32_case("F32TinyTowardNegativeModeFa", fmas_f32(0), f32_boundary, 0xfau, 0x80800000u),
      f32_case("F32SubnormalModeF0", fmas_f32(0), f32_boundary, 0xf0u, 0x807fffffu),
      f64_case("F64TinyTowardNegativeMode7a", fmas_f64, f64_boundary, 0x7au, 0x8000000000000000u),
      f64_case("F64TinyTowardNegativeModeFa", fmas_f64, f64_boundary, 0xfau, 0x8010000000000000u),
  };
  // VCC post-scales by 2^-64; OMOD then flushes the tiny quotient.
  ArithmeticCase post_scaled = f32_case("F32PostScaledMul2TinyModeF0", fmas_f32(1),
                                        {0x3f8983e4u, 0x206e4950u, 0x00000001u}, 0xf0u, 0u);
  post_scaled.vcc = ~uint64_t{0};
  cases.push_back(std::move(post_scaled));
  return cases;
}

// V_DIV_FIXUP_F16 on raw halves, from gfx1201 captures. A nonfinite quotient
// for finite operands overflows under the F16 rounding mode, and FP16_OVFL
// (MODE bit 23) then gives the largest finite value; a true infinity stays.
// OMOD and CLAMP use the shared output modifiers.
std::vector<ArithmeticCase> div_fixup_f16_cases() {
  constexpr uint32_t kHigh = 0xa5a50000u, kSourceHigh = 0x12340000u;
  const auto fixup = [](uint8_t omod, uint8_t opsel = 0) {
    return rdna4::build_vop3(
        rdna4::kVDivFixupF16Vop3,
        {.vdst = 6, .opsel = opsel, .src0 = 256, .src1 = 257, .src2 = 258, .omod = omod});
  };
  const auto f16_case = [&](const std::string &name, std::array<uint32_t, 2> words,
                            std::array<uint16_t, 3> sources, uint32_t mode, uint16_t result) {
    return ArithmeticCase{name,
                          ROCJITSU_CODE_ARCH_RDNA4,
                          {words[0], words[1], 0u},
                          {{0, kSourceHigh | sources[0]},
                           {1, kSourceHigh | sources[1]},
                           {2, kSourceHigh | sources[2]},
                           {6, kHigh}},
                          {{6, kHigh | result}},
                          mode,
                          FE_TONEAREST};
  };
  constexpr uint32_t kOvfl = 0x00800000u;
  std::vector<ArithmeticCase> cases{
      f16_case("OverflowNearestEven", fixup(0), {0x7c00u, 0x3c00u, 0x3c00u}, 0xf0u, 0x7c00u),
      f16_case("OverflowFp16Ovfl", fixup(0), {0x7c00u, 0x3c00u, 0x3c00u}, kOvfl | 0xf0u, 0x7bffu),
      f16_case("OverflowTowardZero", fixup(0), {0x7c00u, 0x3c00u, 0x3c00u}, 0xffu, 0x7bffu),
      f16_case("NegativeOverflowFp16Ovfl", fixup(0), {0x7c00u, 0xbc00u, 0x3c00u}, kOvfl | 0xf0u,
               0xfbffu),
      f16_case("NegativeOverflowTowardPositive", fixup(0), {0x7c00u, 0xbc00u, 0x3c00u}, 0xf5u,
               0xfbffu),
      f16_case("DivideByZeroStaysInfiniteFp16Ovfl", fixup(0), {0x3c00u, 0x0000u, 0x3c00u},
               kOvfl | 0xf0u, 0x7c00u),
      f16_case("NumeratorNanQuieted", fixup(0), {0x3c00u, 0x3c00u, 0x7c01u}, 0xf0u, 0x7e01u),
      // MODE 0x30 flushes the F16 subnormal operands, so 0 / 0 gives the default NaN.
      f16_case("FlushedZeroOverZeroMode30", fixup(0), {0x3c00u, 0x0001u, 0x0001u}, 0x30u, 0xfe00u),
      f16_case("SubnormalOperandsModeF0", fixup(0), {0x3c00u, 0x0001u, 0x0001u}, 0xf0u, 0x3c00u),
      f16_case("Mul2SubnormalQuotient", fixup(1), {0x83ffu, 0x3c00u, 0x3c00u}, 0xf0u, 0x0000u),
      f16_case("Div2NegativeUnderflow", fixup(3), {0x0400u, 0xbc00u, 0x3c00u}, 0xf0u, 0x8000u),
      f16_case("Mul2OverflowTowardZero", fixup(1), {0x7bffu, 0x3c00u, 0x3c00u}, 0xffu, 0x7bffu),
      f16_case("Mul2OverflowNearestEven", fixup(1), {0x7bffu, 0x3c00u, 0x3c00u}, 0xf0u, 0x7c00u),
  };
  // OPSEL selects every source high half and writes the destination high half.
  cases.push_back({"HighHalvesFp16Ovfl",
                   ROCJITSU_CODE_ARCH_RDNA4,
                   {fixup(0, 0xf)[0], fixup(0, 0xf)[1], 0u},
                   {{0, 0x7c001234u}, {1, 0xbc001234u}, {2, 0x3c001234u}, {6, 0x0000a5a5u}},
                   {{6, 0xfbffa5a5u}},
                   kOvfl | 0xf0u,
                   FE_TONEAREST});
  return cases;
}

void expect_arithmetic_case(const ArithmeticCase &test) {
  amdgpu::GpuMemory memory("mode_memory");
  amdgpu::L2Cache cache("mode_cache");
  cache.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = test.arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu =
      amdgpu::ComputeUnitCore::create("valu_fp_mode", config, &memory, &cache);
  std::unique_ptr<Decoder> decoder = Decoder::create(test.arch);
  amdgpu::Wavefront *wave = cu->dispatch_wf(0, 0, 106, 256);
  ASSERT_NE(wave, nullptr);
  DecodeResult decoded = decoder->decode(test.words.data());
  ASSERT_FALSE(decoded.failed());
  std::unique_ptr<Instruction> instruction = std::move(decoded).value();
  const uint32_t base = wave->vgpr_alloc().base;
  const uint64_t full_exec = wave->wf_size() == 64 ? ~uint64_t{0} : 0xffffffffu;
  for (uint64_t exec : {uint64_t{1}, full_exec}) {
    wave->set_exec(exec);
    wave->set_vcc(test.vcc);
    wave->set_mode_raw(test.mode);
    for (const std::pair<uint32_t, uint32_t> &destination : test.expected)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        cu->write_vgpr(base + destination.first, lane, 0xdeadbeefu);
    for (const std::pair<uint32_t, uint32_t> &source : test.sources)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        cu->write_vgpr(base + source.first, lane, source.second);
    std::fenv_t saved_environment;
    ASSERT_EQ(std::fegetenv(&saved_environment), 0);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    const uint32_t saved_mxcsr = _mm_getcsr();
#endif
    ASSERT_EQ(std::fesetround(test.host_rounding), 0);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    // Sticky flags from earlier tests must not hide exception-state leaks.
    const uint32_t host_mxcsr =
        (_mm_getcsr() & ~(test.mxcsr_mask | _MM_EXCEPT_MASK)) | test.mxcsr_bits;
    _mm_setcsr(host_mxcsr);
#endif
    const int initial_rounding = std::fegetround();
    const bool succeeded = cu->execute_instruction(instruction.get(), *wave).succeeded();
    const int restored_rounding = std::fegetround();
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    const uint32_t restored_mxcsr = _mm_getcsr();
#endif
    const int restore_status = std::fesetenv(&saved_environment);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_setcsr(saved_mxcsr);
    if (test.mxcsr_mask != 0) {
      EXPECT_EQ(restored_mxcsr, host_mxcsr);
    }
#endif
    ASSERT_EQ(restore_status, 0);
    EXPECT_TRUE(succeeded);
    EXPECT_EQ(initial_rounding, test.host_rounding);
    EXPECT_EQ(restored_rounding, test.host_rounding);
    for (const std::pair<uint32_t, uint32_t> &destination : test.expected) {
      uint32_t inactive = 0xdeadbeefu;
      for (const std::pair<uint32_t, uint32_t> &source : test.sources)
        if (source.first == destination.first)
          inactive = source.second;
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
        EXPECT_EQ(cu->read_vgpr(base + destination.first, lane),
                  (exec & (uint64_t{1} << lane)) ? destination.second : inactive)
            << "register " << destination.first << " lane " << lane;
      }
    }
  }
  wave->halt();
}

class ValuFpModeTest : public testing::TestWithParam<ArithmeticCase> {};

TEST_P(ValuFpModeTest, HonorsModeAndPreservesInactiveLanes) { expect_arithmetic_case(GetParam()); }

// Restores the process force-scalar gate if an assertion leaves the test early.
struct ForceScalarGuard {
  bool original = util::force_scalar();
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(original); }
};

class ValuMinmaxFpModeTest : public testing::TestWithParam<ArithmeticCase> {};

TEST_P(ValuMinmaxFpModeTest, HonorsModeOnScalarAndSimdPaths) {
  ForceScalarGuard guard;
  for (const bool scalar : {true, false}) {
    SCOPED_TRACE(scalar ? "scalar" : "SIMD enabled");
    util::set_force_scalar_for_testing(scalar);
    expect_arithmetic_case(GetParam());
  }
}

class ValuIntegralRoundingModeTest : public testing::TestWithParam<ArithmeticCase> {};

TEST_P(ValuIntegralRoundingModeTest, ModifiersOnScalarAndSimdPaths) {
  ForceScalarGuard guard;
  for (const bool scalar : {true, false}) {
    SCOPED_TRACE(scalar ? "scalar" : "SIMD enabled");
    util::set_force_scalar_for_testing(scalar);
    expect_arithmetic_case(GetParam());
  }
}

INSTANTIATE_TEST_SUITE_P(OutputModifiers, ValuIntegralRoundingModeTest,
                         testing::ValuesIn(integral_rounding_modifier_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(InputFlush, ValuIntegralRoundingModeTest,
                         testing::ValuesIn(integral_rounding_input_flush_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

class ValuRoundedResultModifierTest : public testing::TestWithParam<ArithmeticCase> {};

TEST_P(ValuRoundedResultModifierTest, MatchesGfx1201OnScalarAndSimdPaths) {
  ForceScalarGuard guard;
  for (const bool scalar : {true, false}) {
    SCOPED_TRACE(scalar ? "scalar" : "SIMD enabled");
    util::set_force_scalar_for_testing(scalar);
    expect_arithmetic_case(GetParam());
  }
}

INSTANTIATE_TEST_SUITE_P(OutputModifiers, ValuRoundedResultModifierTest,
                         testing::ValuesIn(rounded_result_modifier_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Tininess, ValuRoundedResultModifierTest,
                         testing::ValuesIn(tiny_result_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(FmaF64, ValuRoundedResultModifierTest,
                         testing::ValuesIn(fma_f64_policy_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(DivFmas, ValuRoundedResultModifierTest,
                         testing::ValuesIn(div_fmas_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(DivFixupF16, ValuRoundedResultModifierTest,
                         testing::ValuesIn(div_fixup_f16_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(TranscendentalPolicy, ValuFpModeTest,
                         testing::ValuesIn(transcendental_policy_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(FmaMixF32, ValuFpModeTest, testing::ValuesIn(fma_mix_f32_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(FmaMixHalf, ValuFpModeTest, testing::ValuesIn(fma_mix_half_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(AllTargets, ValuFpModeTest, testing::ValuesIn(kCases),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Trigonometry, ValuFpModeTest, testing::ValuesIn(trig_fp_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(LogExp, ValuFpModeTest, testing::ValuesIn(log_exp_policy_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(SdwaLogExp, ValuFpModeTest, testing::ValuesIn(sdwa_log_exp_policy_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(HalfLogExp, ValuFpModeTest,
                         testing::ValuesIn(half_log_exp_arithmetic_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Adjacent, ValuFpModeTest, testing::ValuesIn(adjacent_fp_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(LdexpF16, ValuFpModeTest, testing::ValuesIn(ldexp_f16_mode_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Dx9Fma, ValuFpModeTest, testing::ValuesIn(dx9_fma_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(BinaryF32, ValuFpModeTest, testing::ValuesIn(binary_f32_policy_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(OmodUnderflow, ValuFpModeTest, testing::ValuesIn(omod_underflow_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(F16FmaOmod, ValuFpModeTest, testing::ValuesIn(f16_fma_omod_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(F16FmaNan, ValuFpModeTest, testing::ValuesIn(f16_fma_nan_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(HostMxcsr, ValuFpModeTest, testing::ValuesIn(host_mxcsr_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(ModifierEnvironment, ValuFpModeTest,
                         testing::ValuesIn(modifier_environment_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(MinimumMaximum, ValuMinmaxFpModeTest,
                         testing::ValuesIn(minimum_maximum_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(MinMaxNum, ValuMinmaxFpModeTest, testing::ValuesIn(min_max_num_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(MinmaxInputFlush, ValuMinmaxFpModeTest,
                         testing::ValuesIn(minmax_input_flush_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(MinmaxOutputModifiers, ValuMinmaxFpModeTest,
                         testing::ValuesIn(minmax_output_modifier_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

TEST(ValuFpModeHelpers, PackedF16NanSelection) {
  using namespace amdgpu::fp_mode;
  // Physical gfx1100/gfx1201 captures, with IEEE disabled/enabled. Entries
  // give arithmetic and min/max outputs for legacy-off, legacy-on and newer.
  struct Case {
    uint16_t a, b;
    std::array<uint16_t, 3> arithmetic, select;
  };
  constexpr Case cases[] = {
      {0x7fc1, 0xff80, {0x7fc1, 0x7fc1, 0x7fc1}, {0x7fc1, 0x7fc1, 0x7fc1}},
      {0xff80, 0x7fc1, {0xff80, 0xff80, 0xff80}, {0xff80, 0xff80, 0xff80}},
      {0x7c01, 0xff80, {0x7c01, 0x7e01, 0x7e01}, {0x7c01, 0x7e01, 0x7e01}},
      {0x7fc1, 0xfc02, {0x7fc1, 0x7fc1, 0x7fc1}, {0x7fc1, 0xfe02, 0x7fc1}},
      {0x3c00, 0x7c01, {0x7c01, 0x7e01, 0x7e01}, {0x3c00, 0x7e01, 0x3c00}},
  };
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4})
    for (bool ieee : {false, true}) {
      const unsigned profile = arch == ROCJITSU_CODE_ARCH_RDNA4 ? 2 : unsigned(ieee);
      for (auto op :
           {PackedBinaryOp::ADD, PackedBinaryOp::MUL, PackedBinaryOp::MIN, PackedBinaryOp::MAX})
        for (const auto &test : cases) {
          SCOPED_TRACE(::testing::Message() << int(arch) << ':' << ieee << ':' << int(op) << ':'
                                            << test.a << ':' << test.b);
          const uint16_t expected =
              (op == PackedBinaryOp::ADD || op == PackedBinaryOp::MUL ? test.arithmetic
                                                                      : test.select)[profile];
          EXPECT_EQ(packed_binary_f16(op, test.a, test.b, 0, 3, false, false, false, arch, ieee),
                    expected);
          EXPECT_EQ(packed_binary_f16(op, test.a, test.b, 0, 3, true, false, true, arch, ieee),
                    expected > 0x7c00u ? 0 : 0x3c00u);
        }
    }
}

TEST(ValuFpModeHelpers, RoundingAndSignedZero) {
  using amdgpu::fp_mode::Arithmetic;
  for (uint32_t mode = 0; mode < 4; ++mode) {
    const uint32_t expected = mode == 1 ? 0x3f800001u : 0x3f800000u;
    EXPECT_EQ(std::bit_cast<uint32_t>(
                  amdgpu::fp_mode::arithmetic<Arithmetic::ADD>(1.0f, 0x1p-24f, 0.0f, mode, 3)),
              expected);
    EXPECT_EQ(std::bit_cast<uint32_t>(
                  amdgpu::fp_mode::arithmetic<Arithmetic::SUB>(1.0f, 1.0f, 0.0f, mode, 3)),
              mode == 2 ? 0x80000000u : 0u);
    EXPECT_EQ(std::bit_cast<uint32_t>(
                  amdgpu::fp_mode::arithmetic<Arithmetic::MUL>(-0.0f, 1.0f, 0.0f, mode, 3)),
              0x80000000u);
  }
}

TEST(ValuFpModeHelpers, MixedF16FmaExactZeroSigns) {
  struct Case {
    uint32_t a, b, c;
    std::array<uint16_t, 4> expected;
  };
  constexpr std::array<Case, 8> cases{{
      {0, 0x3f800000, 0, {0, 0, 0, 0}},
      {0, 0x3f800000, 0x80000000, {0, 0, 0x8000, 0}},
      {0x80000000, 0x3f800000, 0, {0, 0, 0x8000, 0}},
      {0x80000000, 0x3f800000, 0x80000000, {0x8000, 0x8000, 0x8000, 0x8000}},
      {0, 0xbf800000, 0, {0, 0, 0x8000, 0}},
      {0, 0xbf800000, 0x80000000, {0x8000, 0x8000, 0x8000, 0x8000}},
      {0x80000000, 0xbf800000, 0, {0, 0, 0, 0}},
      {0x80000000, 0xbf800000, 0x80000000, {0, 0, 0x8000, 0}},
  }};
  for (const auto &test : cases)
    for (bool zero_second : {false, true})
      for (bool swap : {false, true})
        for (bool clamp : {false, true})
          for (uint32_t mode = 0; mode < 4; ++mode) {
            uint32_t a = test.a, b = zero_second ? test.b & 0x80000000u : test.b;
            if (swap)
              std::swap(a, b);
            amdgpu::fp_mode::ScopedEnvironment environment(0);
            const uint16_t actual = amdgpu::fp_mode::detail::fma_f32_to_f16_nearest_environment(
                std::bit_cast<float>(a), std::bit_cast<float>(b), std::bit_cast<float>(test.c),
                mode, clamp, false, false);
            EXPECT_EQ(actual, clamp ? 0u : test.expected[mode]);
          }
}

TEST(ValuFpModeHelpers, MixedF16FmaCancellationAcrossF32Range) {
  constexpr std::array<std::array<uint32_t, 3>, 5> cases{{
      {0x7f7fffff, 0x3f800000, 0xff7fffff},
      {1, 0x3f800000, 0x80000001},
      {0x00800000, 0x3f000000, 0x80400000},
      {0x5f800000, 0x1f800000, 0xbf800000},
      {0x00800000, 0x34000000, 0x80000001},
  }};
  for (const auto &test : cases)
    for (uint32_t mode = 0; mode < 4; ++mode) {
      amdgpu::fp_mode::ScopedEnvironment environment(0);
      EXPECT_EQ(amdgpu::fp_mode::detail::fma_f32_to_f16_nearest_environment(
                    std::bit_cast<float>(test[0]), std::bit_cast<float>(test[1]),
                    std::bit_cast<float>(test[2]), mode, false, false, false),
                mode == 2 ? 0x8000u : 0u);
    }
  // These products are far below F16 range, but are nonzero F64 values. A
  // sign-only repair must not turn directed underflow into exact cancellation.
  for (uint32_t mode = 0; mode < 4; ++mode) {
    amdgpu::fp_mode::ScopedEnvironment environment(0);
    EXPECT_EQ(
        amdgpu::fp_mode::detail::fma_f32_to_f16_nearest_environment(
            std::bit_cast<float>(1u), std::bit_cast<float>(1u), 0.0f, mode, false, false, false),
        mode == 1 ? 1u : 0u);
    EXPECT_EQ(amdgpu::fp_mode::detail::fma_f32_to_f16_nearest_environment(
                  std::bit_cast<float>(0x80000001u), std::bit_cast<float>(1u), 0.0f, mode, false,
                  false, false),
              mode == 2 ? 0x8001u : 0x8000u);
  }
}

TEST(ValuFpModeHelpers, F16FmaRetainsTinyProduct) {
  // Both physical cards round 65504 + 2^-48 upward to infinity. A host F64
  // addition alone loses the tiny product and incorrectly returns 65504.
  for (uint32_t round = 0; round < 4; ++round)
    for (uint32_t denorm = 0; denorm < 4; ++denorm)
      for (bool overflow : {false, true}) {
        const uint16_t expected = round == 1 && (denorm & 1u) && !overflow ? 0x7c00u : 0x7bffu;
        EXPECT_EQ(amdgpu::fp_mode::fma_f16(1, 1, 0x7bff, false, false, false, false, false, false,
                                           round, denorm, 0, false, overflow, false, true),
                  expected);
      }
}

TEST(ValuFpModeHelpers, F16FmaFlushesBeforePacking) {
  // Packing directly to a subnormal half can round these tiny intermediates
  // to minimum normal; hardware tests the rounded significand first.
  for (uint32_t round = 0; round < 4; ++round)
    for (uint16_t sign : {uint16_t{0}, uint16_t{0x8000}})
      EXPECT_EQ(amdgpu::fp_mode::fma_f16(1, sign | 0x03ff, sign | 0x03ff, false, false, false,
                                         false, false, false, round, 1, 0, false, false, false,
                                         true),
                sign);
}

TEST(ValuFpModeHelpers, FusedResultAndOutputFlush) {
  using amdgpu::fp_mode::Arithmetic;
  EXPECT_EQ(std::bit_cast<uint32_t>(amdgpu::fp_mode::arithmetic<Arithmetic::FMA>(
                0x1.000002p0f, 0x1.fffffcp-1f, -1.0f, 0, 3)),
            0xa8800000u);
  EXPECT_EQ(std::bit_cast<uint32_t>(
                amdgpu::fp_mode::arithmetic<Arithmetic::MUL>(-0x1p-126f, 0.5f, 0.0f, 0, 1)),
            0x80000000u);
}

TEST(ValuFpModeHelpers, OutputScalePrecedesF16Rounding) {
  EXPECT_EQ(amdgpu::fp_mode::finish_arithmetic_f16(65504.0 * 2.0, 0, 1, false, 3), 0x7bffu);
  EXPECT_EQ(amdgpu::fp_mode::finish_arithmetic_f16(1.0 + 0x1p-24, 1, 1, false, 1), 0x4001u);
  EXPECT_EQ(amdgpu::fp_mode::finish_arithmetic_f16(-0.0, 0, 1, false, 1), 0u);
}

TEST(ValuFpModeHelpers, HostFlushControlsDoNotOverrideGpuMode) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  using amdgpu::fp_mode::Arithmetic;
  const uint32_t saved_mxcsr = _mm_getcsr();
  const uint32_t host_mxcsr = saved_mxcsr | (1u << 6) | (1u << 15);
  _mm_setcsr(host_mxcsr);
  const bool native_matches = amdgpu::fp_mode::native_arithmetic_matches(0, 3);
  const uint32_t result = std::bit_cast<uint32_t>(amdgpu::fp_mode::arithmetic<Arithmetic::ADD>(
      std::bit_cast<float>(1u), std::bit_cast<float>(1u), 0.0f, 0, 3));
  const uint32_t scaled32 =
      std::bit_cast<uint32_t>(amdgpu::ldexp(std::bit_cast<float>(1u), 1, 1, 3));
  const uint64_t scaled64 =
      std::bit_cast<uint64_t>(amdgpu::ldexp(std::bit_cast<double>(uint64_t{1}), 1, 1, 3));
  const amdgpu::FrexpF32Result split = amdgpu::frexp_f32(std::bit_cast<float>(1u), 3);
  const uint32_t restored_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr);
  EXPECT_FALSE(native_matches);
  EXPECT_EQ(result, 2u);
  EXPECT_EQ(scaled32, 2u);
  EXPECT_EQ(scaled64, 2u);
  EXPECT_EQ(std::bit_cast<uint32_t>(split.mantissa), 0x3f000000u);
  EXPECT_EQ(split.exponent, -148);
  EXPECT_EQ(restored_mxcsr, host_mxcsr);
#endif
}

// Floating compares flush their inputs under MODE.FP_DENORM before the
// relation. Each case runs one decoded RDNA4 compare, built with the generated
// encoders, on lanes whose src0 cycles through +tiny, -tiny, +0 and 1.0
// with src1 = +0. V_CMP_EQ then selects lane 2 of each group of four when input
// denormals are kept, and lanes 0-2 when they are flushed. F16 sources hold a
// NaN in their high half, so only the .l halves may be read.
struct CompareInputFlushCase {
  std::string name;
  std::array<uint32_t, 2> words;
  unsigned width;
  bool vop3;
  uint32_t mode;
  uint32_t expected;
};

void PrintTo(const CompareInputFlushCase &test, std::ostream *stream) { *stream << test.name; }

std::vector<CompareInputFlushCase> compare_input_flush_cases() {
  constexpr uint32_t KEEP = 0x44444444u;
  constexpr uint32_t FLUSH = 0x77777777u;
  // MODE.FP_DENORM is [5:4] for F32 and [7:6] for F16/F64; bit 0 of each field
  // allows input denormals. The modes set the two fields differently and vary
  // the output bit alone, so reading the other format's field or the output
  // bit gives the wrong mask.
  constexpr std::array<std::pair<uint32_t, uint32_t>, 8> F32_MODES = {{
      {0xf0u, KEEP},
      {0x00u, FLUSH},
      {0x30u, KEEP},
      {0xc0u, FLUSH},
      {0xe0u, FLUSH},
      {0xd0u, KEEP},
      {0xb0u, KEEP},
      {0x70u, KEEP},
  }};
  constexpr std::array<std::pair<uint32_t, uint32_t>, 8> F16_F64_MODES = {{
      {0xf0u, KEEP},
      {0x00u, FLUSH},
      {0x30u, FLUSH},
      {0xc0u, KEEP},
      {0xe0u, KEEP},
      {0xd0u, KEEP},
      {0xb0u, FLUSH},
      {0x70u, KEEP},
  }};
  struct Form {
    const char *name;
    std::array<uint32_t, 2> words;
    unsigned width;
    bool vop3;
  };
  // Source operands encode VGPR n as 256 + n; VOPC vsrc1 is the VGPR index.
  constexpr uint16_t V0 = 256, V1 = 257, V2 = 258;
  const auto vopc = [](uint16_t op, uint8_t vsrc1) {
    return std::array<uint32_t, 2>{rdna4::build_vopc(op, {.src0 = V0, .vsrc1 = vsrc1})[0], 0u};
  };
  const auto vop3 = [](uint16_t op, uint16_t src1) {
    return rdna4::build_vop3(op, {.vdst = 6, .src0 = V0, .src1 = src1});
  };
  const std::array<Form, 6> forms = {{
      // v_cmp_eq_f32_e32 vcc_lo, v0, v1
      {"EqF32E32", vopc(rdna4::kVCmpEqF32Vopc, 1), 32, false},
      // v_cmp_eq_f32_e64 s6, v0, v1
      {"EqF32E64", vop3(rdna4::kVCmpEqF32Vop3, V1), 32, true},
      // v_cmp_eq_f16_e32 vcc_lo, v0.l, v1.l
      {"EqF16E32", vopc(rdna4::kVCmpEqF16Vopc, 1), 16, false},
      // v_cmp_eq_f16_e64 s6, v0.l, v1.l
      {"EqF16E64", vop3(rdna4::kVCmpEqF16Vop3, V1), 16, true},
      // v_cmp_eq_f64_e32 vcc_lo, v[0:1], v[2:3]
      {"EqF64E32", vopc(rdna4::kVCmpEqF64Vopc, 2), 64, false},
      // v_cmp_eq_f64_e64 s6, v[0:1], v[2:3]
      {"EqF64E64", vop3(rdna4::kVCmpEqF64Vop3, V2), 64, true},
  }};
  std::vector<CompareInputFlushCase> cases;
  for (const Form &form : forms)
    for (const auto &[mode, expected] : form.width == 32 ? F32_MODES : F16_F64_MODES) {
      constexpr char HEX[] = "0123456789abcdef";
      const std::string suffix = {HEX[mode >> 4], HEX[mode & 0xfu]};
      cases.push_back({std::string(form.name) + "Mode" + suffix, form.words, form.width, form.vop3,
                       mode, expected});
    }
  return cases;
}

class ValuCompareInputFlushTest : public testing::TestWithParam<CompareInputFlushCase> {};

TEST_P(ValuCompareInputFlushTest, HonorsModeOnScalarAndSimdPaths) {
  const CompareInputFlushCase &test = GetParam();
  amdgpu::GpuMemory memory("compare_memory");
  amdgpu::L2Cache cache("compare_cache");
  cache.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu =
      amdgpu::ComputeUnitCore::create("valu_compare", config, &memory, &cache);
  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  amdgpu::Wavefront *wave = cu->dispatch_wf(0, 0, 106, 256);
  ASSERT_NE(wave, nullptr);
  DecodeResult decoded = decoder->decode(test.words.data());
  ASSERT_FALSE(decoded.failed());
  std::unique_ptr<Instruction> instruction = std::move(decoded).value();
  const uint32_t vgpr = wave->vgpr_alloc().base;
  const uint32_t sgpr = wave->sgpr_alloc().base;
  const bool wave64 = wave->wf_size() == 64;
  const uint64_t full_exec = wave64 ? ~uint64_t{0} : 0xffffffffu;
  const uint64_t expected_lanes = test.expected | (uint64_t{test.expected} << 32);
  // +tiny, -tiny, +0 and 1.0 in each format.
  constexpr std::array<uint32_t, 4> F16_SOURCES = {0x7e000001u, 0x7e008001u, 0x7e000000u,
                                                   0x7e003c00u};
  constexpr std::array<uint32_t, 4> F32_SOURCES = {0x00000001u, 0x80000001u, 0u, 0x3f800000u};
  constexpr std::array<uint32_t, 4> F64_HIGH = {0u, 0x80000000u, 0u, 0x3ff00000u};
  constexpr std::array<uint32_t, 4> F64_LOW = {1u, 1u, 0u, 0u};
  ForceScalarGuard guard;
  for (const bool scalar : {true, false}) {
    util::set_force_scalar_for_testing(scalar);
    for (const uint64_t exec : {full_exec, full_exec & uint64_t{0x0f0f0f0f0f0f0f0f}}) {
      wave->set_exec(exec);
      wave->set_mode_raw(test.mode);
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane) {
        if (test.width == 64) {
          cu->write_vgpr(vgpr + 0, lane, F64_LOW[lane % 4]);
          cu->write_vgpr(vgpr + 1, lane, F64_HIGH[lane % 4]);
          cu->write_vgpr(vgpr + 2, lane, 0u);
          cu->write_vgpr(vgpr + 3, lane, 0u);
        } else {
          cu->write_vgpr(vgpr + 0, lane,
                         test.width == 16 ? F16_SOURCES[lane % 4] : F32_SOURCES[lane % 4]);
          cu->write_vgpr(vgpr + 1, lane, test.width == 16 ? 0x7e000000u : 0u);
        }
      }
      // Stale destination bits must be replaced, including inactive lanes.
      wave->set_vcc_mask(uint64_t{0xdeadbeefdeadbeef});
      cu->write_sgpr(sgpr + 6, 0xdeadbeefu);
      cu->write_sgpr(sgpr + 7, 0xdeadbeefu);
      ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
      uint64_t result = wave->vcc_mask();
      if (test.vop3)
        result = cu->read_sgpr(sgpr + 6) |
                 (wave64 ? uint64_t{cu->read_sgpr(sgpr + 7)} << 32 : uint64_t{0});
      EXPECT_EQ(result & full_exec, expected_lanes & exec)
          << (scalar ? "scalar" : "SIMD") << " exec 0x" << std::hex << exec;
    }
  }
  wave->halt();
}

INSTANTIATE_TEST_SUITE_P(CompareInputFlush, ValuCompareInputFlushTest,
                         testing::ValuesIn(compare_input_flush_cases()),
                         [](const testing::TestParamInfo<CompareInputFlushCase> &info) {
                           return info.param.name;
                         });

// SALU float min/max match their VALU counterparts (RDNA4 ISA section 6.8),
// including the gfx1201 NaN order. Each case is a gfx1201 capture. F16 forms
// read bits [15:0] and zero the destination's high half.
struct SaluMinmaxCase {
  const char *name;
  uint16_t op;
  uint32_t a;
  uint32_t b;
  uint32_t mode;
  uint32_t expected;
};

void PrintTo(const SaluMinmaxCase &test, std::ostream *stream) { *stream << test.name; }

const SaluMinmaxCase kSaluMinmaxCases[] = {
    {"MinNumF32NegativeZero", rdna4::kSMinNumF32Sop2, 0x00000000u, 0x80000000u, 0xf0u, 0x80000000u},
    {"MinNumF32BothQuietNans", rdna4::kSMinNumF32Sop2, 0x7fc00000u, 0xffc00000u, 0xf0u,
     0x7fc00000u},
    {"MinNumF32IgnoresSignalingNan", rdna4::kSMinNumF32Sop2, 0x3f800000u, 0x7f800001u, 0xf0u,
     0x3f800000u},
    {"MinNumF32KeepsDenormal", rdna4::kSMinNumF32Sop2, 0x80000001u, 0x00000000u, 0x30u,
     0x80000001u},
    {"MinNumF32FlushesDenormal", rdna4::kSMinNumF32Sop2, 0x80000001u, 0x00000000u, 0xc0u,
     0x80000000u},
    {"MaxNumF32PositiveZero", rdna4::kSMaxNumF32Sop2, 0x80000000u, 0x00000000u, 0xf0u, 0x00000000u},
    // ISA discrepancy, as for V_MINIMUM_F32: the quiet src0 beats a signaling src1.
    {"MinimumF32FirstNanWins", rdna4::kSMinimumF32Sop2, 0x7fc00000u, 0x7f800001u, 0xf0u,
     0x7fc00000u},
    {"MinimumF32KeepsNanPayload", rdna4::kSMinimumF32Sop2, 0x00000000u, 0xffc00456u, 0xf0u,
     0xffc00456u},
    {"MaximumF32QuietsSignalingNan", rdna4::kSMaximumF32Sop2, 0x7f800001u, 0x00000000u, 0xf0u,
     0x7fc00001u},
    {"MaximumF32FlushesDenormal", rdna4::kSMaximumF32Sop2, 0x007fffffu, 0x00000000u, 0xc0u,
     0x00000000u},
    {"MinNumF16FlushesDenormal", rdna4::kSMinNumF16Sop2, 0x5a5a8001u, 0x5a5a0000u, 0x30u,
     0x00008000u},
    {"MinNumF16KeepsDenormal", rdna4::kSMinNumF16Sop2, 0x5a5a8001u, 0x5a5a0000u, 0xc0u,
     0x00008001u},
    {"MaxNumF16PositiveZero", rdna4::kSMaxNumF16Sop2, 0x5a5a8000u, 0x5a5a0000u, 0xf0u, 0x00000000u},
    {"MaxNumF16BothQuietNans", rdna4::kSMaxNumF16Sop2, 0xa5a57e00u, 0x0000fe34u, 0xf0u,
     0x00007e00u},
    {"MinimumF16FirstNanWins", rdna4::kSMinimumF16Sop2, 0xa5a57e00u, 0x00007c01u, 0xf0u,
     0x00007e00u},
    {"MinimumF16KeepsNanPayload", rdna4::kSMinimumF16Sop2, 0x00000000u, 0x0000fe34u, 0xf0u,
     0x0000fe34u},
    {"MaximumF16QuietsSignalingNan", rdna4::kSMaximumF16Sop2, 0x00007c01u, 0x00007e00u, 0xf0u,
     0x00007e01u},
    {"MaximumF16FlushesDenormal", rdna4::kSMaximumF16Sop2, 0x000003ffu, 0x00000000u, 0x30u,
     0x00000000u},
};

class SaluMinmaxTest : public testing::TestWithParam<SaluMinmaxCase> {};

TEST_P(SaluMinmaxTest, MatchesGfx1201) {
  const SaluMinmaxCase &test = GetParam();
  amdgpu::GpuMemory memory("salu_minmax_memory");
  amdgpu::L2Cache cache("salu_minmax_cache");
  cache.set_backing_memory(&memory);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = ROCJITSU_CODE_ARCH_RDNA4;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu =
      amdgpu::ComputeUnitCore::create("salu_minmax", config, &memory, &cache);
  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  amdgpu::Wavefront *wave = cu->dispatch_wf(0, 0, 106, 256);
  ASSERT_NE(wave, nullptr);
  // s_<op> s6, s0, s1
  const auto word = rdna4::build_sop2(test.op, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 6});
  const std::array<uint32_t, 3> words = {word[0], 0u, 0u};
  DecodeResult decoded = decoder->decode(words.data());
  ASSERT_FALSE(decoded.failed());
  std::unique_ptr<Instruction> instruction = std::move(decoded).value();
  const uint32_t sgpr = wave->sgpr_alloc().base;
  wave->set_mode_raw(test.mode);
  cu->write_sgpr(sgpr + 0, test.a);
  cu->write_sgpr(sgpr + 1, test.b);
  cu->write_sgpr(sgpr + 6, 0xdeadbeefu);
  ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
  EXPECT_EQ(cu->read_sgpr(sgpr + 6), test.expected);
  wave->halt();
}

INSTANTIATE_TEST_SUITE_P(SaluMinmax, SaluMinmaxTest, testing::ValuesIn(kSaluMinmaxCases),
                         [](const testing::TestParamInfo<SaluMinmaxCase> &info) {
                           return info.param.name;
                         });

} // namespace
