// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

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
    {"Gfx950MulF16ScaleBeforeNarrow",
     ROCJITSU_CODE_ARCH_CDNA4,
     {0xd1220006u, 0x18020300u},
     {{0, 0x7bffu}, {1, 0x4000u}},
     {{6, 0x7bffu}},
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
    // Output scaling must precede narrowing to half, including overflow rescue.
    if (encoding.words[1] != 0) {
      add("OmodOverflowRescue", 0x7bffu, 1, 0x7bffu, 0);
      cases.back().words[1] |= 3u << 27; // div:2
    }
  }
  return cases;
}

std::vector<ArithmeticCase> dx9_fma_cases() {
  std::vector<ArithmeticCase> cases;
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (uint32_t denorm = 0; denorm < 4; ++denorm) {
      const std::string prefix = "Arch" + std::to_string(arch) + "Denorm" + std::to_string(denorm);
      const auto add = [&](const char *name, uint32_t a, uint32_t b, uint32_t c,
                           uint32_t expected) {
        // v_fma_dx9_zero_f32 v6, v0, v1, v2, assembled with llvm-mc.
        cases.push_back({prefix + name,
                         arch,
                         {0xd6090006u, 0x040a0300u},
                         {{0, a}, {1, b}, {2, c}},
                         {{6, expected}},
                         denorm << 4,
                         FE_TONEAREST});
      };
      add("ZeroInf", 0u, 0x7f800000u, 0x3f800000u, 0x3f800000u);
      add("NanZero", 0x7fc01234u, 0x80000000u, 0x3f800000u, 0x3f800000u);
      add("InputFlush", 1u, 0x4b000000u, 0u, 0u);
      add("OutputFlush", 0x00800000u, 0x3f000000u, 0u, 0u);
      add("AccumulatorFlush", 0u, 0x7f800000u, 0x80000001u, 0x80000000u);
      add("NegativeZeroAccumulator", 0u, 0x7f800000u, 0x80000000u, 0x80000000u);
      add("NormalControl", 0x3f800000u, 0x3f800000u, 0x3f800000u, 0x40000000u);
    }
  }
  return cases;
}

std::vector<ArithmeticCase> host_mxcsr_cases() {
  std::vector<ArithmeticCase> cases;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  constexpr uint32_t kDazMask = 1u << 6;
  constexpr uint32_t kFtzMask = 1u << 15;
  constexpr uint32_t kControlMask = _MM_ROUND_MASK | kDazMask | kFtzMask;
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

class ValuFpModeTest : public testing::TestWithParam<ArithmeticCase> {};

TEST_P(ValuFpModeTest, HonorsModeAndPreservesInactiveLanes) {
  const ArithmeticCase &test = GetParam();
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

INSTANTIATE_TEST_SUITE_P(AllTargets, ValuFpModeTest, testing::ValuesIn(kCases),
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

INSTANTIATE_TEST_SUITE_P(HostMxcsr, ValuFpModeTest, testing::ValuesIn(host_mxcsr_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(ModifierEnvironment, ValuFpModeTest,
                         testing::ValuesIn(modifier_environment_cases()),
                         [](const testing::TestParamInfo<ArithmeticCase> &info) {
                           return info.param.name;
                         });

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

} // namespace
