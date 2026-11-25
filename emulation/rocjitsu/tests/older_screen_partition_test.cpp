// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older CDNA rectangle-coverage routing, including SDWA placement, overlap and masks.
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/execution_backend.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/arithmetic_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <bit>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>

namespace rocjitsu::amdgpu {
namespace {
// Independently enumerate the shader engines touched by the wrapped rectangle.
uint32_t coverage(uint32_t selector) {
  const int xmin = (selector >> 6) & 3, ymin = (selector >> 4) & 3;
  const int dx = (((selector >> 2) & 3) - xmin) & 3, dy = ((selector & 3) - ymin) & 3;
  uint32_t result = 0;
  for (int x = xmin; x <= xmin + dx; ++x) {
    for (int y = ymin; y <= ymin + dy; ++y) {
      const int engine = (((x ^ (y >> 1)) & 1) << 1) | ((y ^ (x >> 1)) & 1);
      result |= 1U << engine;
    }
  }
  return result;
}

// Both profiles share these encodings; the test still decodes with each target's decoder.
std::array<uint32_t, 2> encode(int kind, uint8_t destination) {
  static_assert(cdna1::kVScreenPartition4seB32Vop1 == cdna2::kVScreenPartition4seB32Vop1);
  static_assert(cdna1::kVScreenPartition4seB32Vop3 == cdna2::kVScreenPartition4seB32Vop3);
  if (kind == 1) {
    return cdna2::build_vop3(cdna2::kVScreenPartition4seB32Vop3,
                             {.vdst = destination, .src0 = 256});
  }
  auto base =
      cdna2::build_vop1(cdna2::kVScreenPartition4seB32Vop1, {.src0 = 256, .vdst = destination});
  if (kind == 0) {
    return {base[0], 0};
  }
  cdna2::Vop1VopSdwaMachineInst raw{};
  raw.encoding = base[0] >> 25;
  raw.op = cdna2::kVScreenPartition4seB32Vop1;
  raw.vdst = destination;
  raw.src0 = SRC_SDWA;
  raw.vsrc0 = 0;
  raw.src0_sel = sdwa::BYTE_1;
  raw.dst_sel = sdwa::BYTE_2;
  raw.dst_unused = sdwa::UNUSED_PRESERVE;
  return std::bit_cast<std::array<uint32_t, 2>>(raw);
}

// Byte placement is separate from the coverage model and preserves the other bytes.
TEST(OlderScreenPartition, CoverageAndPlacement) {
  // The named engine and default seed give a sequence independent of the C++ library.
  std::minstd_rand random;
  uint32_t input[2][64]; // One register file permits D == A without changing the reference.
  for (auto &reg : input) {
    for (auto &word : reg) {
      const auto first_draw = random();
      word = (uint32_t(first_draw >> 15) << 16) | uint32_t(random() >> 15);
    }
  }
  for (auto arch : {rocjitsu::IsaFamily::Cdna1, rocjitsu::IsaFamily::Cdna2}) {
    ScopedIsaExecutionBackend backend(arch == rocjitsu::IsaFamily::Cdna1
                                          ? &cdna1::execution_backend()
                                          : &cdna2::execution_backend());
    for (auto semantics : {ArithmeticSemantics::Loose, ArithmeticSemantics::Compatibility}) {
      ComputeUnitCore::Config config{};
      config.arch = arch;
      config.arithmetic = {semantics};
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 104;
      config.vgprs_per_wf = 8;
      config.lds_size_kb = 64;
      GpuMemory memory("screen_partition_memory");
      L2Cache l2("screen_partition_l2");
      auto cu = ComputeUnitCore::create("screen_partition", config, &memory, &l2);
      auto *wave = cu->dispatch_wf(0, 0, 104, 8);
      ASSERT_NE(wave, nullptr);
      auto decoder = Decoder::create(arch);
      const auto base = wave->vgpr_alloc().base;
      for (int kind : {0, 1, 2}) {
        for (uint8_t aliasing_offset : {0, 1}) {
          for (uint64_t exec_mask : {~0ULL, 0x800000010000000bULL, 0ULL}) {
            wave->set_exec(exec_mask);
            wave->set_vgpr_write_mask(~2ULL);
            wave->set_vcc_raw(0x123456789abcdef0ULL);
            wave->write_scc(true);
            for (unsigned reg = 0; reg < 2; ++reg) {
              for (unsigned lane = 0; lane < 64; ++lane) {
                cu->write_vgpr(base + reg, lane, input[reg][lane]);
              }
            }
            auto decoded = decoder->decode_window(encode(kind, aliasing_offset));
            ASSERT_TRUE(decoded.succeeded());
            const auto diagnostics = cu->isa_diagnostic_count();
            ASSERT_TRUE(cu->execute_instruction(decoded.value().get(), *wave).succeeded());
            EXPECT_EQ(cu->isa_diagnostic_count(), diagnostics);
            for (unsigned reg = 0; reg < 2; ++reg) {
              for (unsigned lane = 0; lane < 64; ++lane) {
                uint32_t expected = input[reg][lane];
                if (reg == aliasing_offset && (((exec_mask & ~2ULL) >> lane) & 1)) {
                  const auto value = coverage(input[0][lane] >> (kind == 2 ? 8 : 0));
                  expected = kind == 2 ? (expected & ~0x00ff0000U) | (value << 16) : value;
                }
                EXPECT_EQ(cu->read_vgpr(base + reg, lane), expected)
                    << arch << ',' << kind << ',' << unsigned(aliasing_offset) << ',' << lane;
              }
            }
            EXPECT_EQ(wave->exec(), exec_mask);
            EXPECT_EQ(wave->vcc(), 0x123456789abcdef0ULL);
            EXPECT_TRUE(wave->read_scc());
          }
        }
      }
    }
  }
}
} // namespace
} // namespace rocjitsu::amdgpu
