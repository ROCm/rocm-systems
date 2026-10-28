// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

struct PcArchitecture {
  rj_code_arch_t arch;
  const char *name;
};

struct PcExportCase {
  uint64_t pc;
  uint64_t signed48;
  uint64_t zero48;
  uint64_t signed57;
};

// Expected values describe the next instruction's address, independently of the
// representation used for the emulator's internal PC.
constexpr std::array kExportCases{
    PcExportCase{0x0000000012345000ULL, 0x0000000012345004ULL, 0x0000000012345004ULL,
                 0x0000000012345004ULL},
    PcExportCase{0x0000800012345000ULL, 0xFFFF800012345004ULL, 0x0000800012345004ULL,
                 0x0000800012345004ULL},
    PcExportCase{0xFFFF800012345000ULL, 0xFFFF800012345004ULL, 0x0000800012345004ULL,
                 0xFFFF800012345004ULL},
    PcExportCase{0x0001800012345000ULL, 0xFFFF800012345004ULL, 0x0000800012345004ULL,
                 0x0001800012345004ULL},
    PcExportCase{0x0101000012345000ULL, 0x0000000012345004ULL, 0x0000000012345004ULL,
                 0xFF01000012345004ULL},
    PcExportCase{0xFF01000012345000ULL, 0x0000000012345004ULL, 0x0000000012345004ULL,
                 0xFF01000012345004ULL},
    PcExportCase{0x00007FFFFFFFFFFCULL, 0xFFFF800000000000ULL, 0x0000800000000000ULL,
                 0x0000800000000000ULL},
    PcExportCase{0x00FFFFFFFFFFFFFCULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
                 0xFF00000000000000ULL},
};

// RDNA3's overview and LLVM specify signed48 export; its branch table says
// zero extension. These high-address expectations follow LLVM and have no
// physical gfx1100 qualification.

struct PcTargetCase {
  uint64_t encoded;
  uint64_t target48;
  uint64_t target57;
};

constexpr std::array kTargetCases{
    PcTargetCase{0x0000000012345000ULL, 0x0000000012345000ULL, 0x0000000012345000ULL},
    PcTargetCase{0x0001000012345000ULL, 0x0000000012345000ULL, 0x0001000012345000ULL},
    PcTargetCase{0x0001800012345000ULL, 0x0000800012345000ULL, 0x0001800012345000ULL},
    PcTargetCase{0x0101000012345000ULL, 0x0000000012345000ULL, 0x0101000012345000ULL},
    PcTargetCase{0xFF01000012345000ULL, 0x0000000012345000ULL, 0x0101000012345000ULL},
};

class PcExtensionTest : public ::testing::TestWithParam<PcArchitecture> {
protected:
  void SetUp() override {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam().arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 102;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("pc_extension", config, &memory, &cache);
    decoder = Decoder::create(GetParam().arch);
    ASSERT_NE(cu, nullptr);
    ASSERT_NE(decoder, nullptr);
    wave = cu->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(0); // Scalar control flow executes even without active lanes.
    ASSERT_EQ(wave->code_load_bias(), 0u);
  }

  void TearDown() override {
    if (wave)
      wave->halt();
  }

  void execute(uint32_t word) {
    const std::array<uint32_t, 2> words{word, 0};
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
    ASSERT_NE(instruction, nullptr);
    ASSERT_EQ(instruction->size(), 4u);
    ASSERT_TRUE(cu->execute_instruction(instruction.get(), *wave).succeeded());
    // Match the CU's normal post-execution increment without fetching code at
    // the deliberately high addresses exercised by these tests.
    wave->pc += instruction->size();
  }

  uint64_t expected_export(const PcExportCase &test) const {
    if (GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5)
      return test.signed57;
    return GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4 ? test.zero48 : test.signed48;
  }

  uint64_t expected_target(const PcTargetCase &test) const {
    return GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5 ? test.target57 : test.target48;
  }

  void write_pair(uint32_t reg, uint64_t value) {
    const uint32_t base = wave->sgpr_alloc().base;
    cu->write_sgpr(base + reg, static_cast<uint32_t>(value));
    cu->write_sgpr(base + reg + 1, static_cast<uint32_t>(value >> 32));
  }

  uint64_t read_pair(uint32_t reg) const {
    const uint32_t base = wave->sgpr_alloc().base;
    return uint64_t{cu->read_sgpr(base + reg)} | (uint64_t{cu->read_sgpr(base + reg + 1)} << 32);
  }

  amdgpu::GpuMemory memory{"pc_extension_memory"};
  amdgpu::L2Cache cache{"pc_extension_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;
};

INSTANTIATE_TEST_SUITE_P(Architectures, PcExtensionTest,
                         ::testing::Values(PcArchitecture{ROCJITSU_CODE_ARCH_RDNA3, "Gfx1100"},
                                           PcArchitecture{ROCJITSU_CODE_ARCH_RDNA4, "Gfx1201"},
                                           PcArchitecture{ROCJITSU_CODE_ARCH_CDNA4, "Gfx950"},
                                           PcArchitecture{ROCJITSU_CODE_ARCH_CDNA5, "Gfx1250"}),
                         [](const ::testing::TestParamInfo<PcArchitecture> &info) {
                           return std::string(info.param.name);
                         });

TEST_P(PcExtensionTest, GetpcExportsTheTargetAddressWidth) {
  for (const PcExportCase &test : kExportCases) {
    SCOPED_TRACE(::testing::Message() << "pc=" << std::hex << test.pc);
    wave->pc = test.pc;
    write_pair(4, 0xDEADBEEFDEADBEEFULL);
    ASSERT_NO_FATAL_FAILURE(execute(build_s_getpc_b64(4, GetParam().arch)));
    EXPECT_EQ(read_pair(4), expected_export(test));
    EXPECT_EQ(wave->pc, test.pc + 4);
  }
}

TEST_P(PcExtensionTest, CallExportsTheTargetAddressWidth) {
  // s_call_b64 / s_call_i64 s[4:5], 1: export the return address while
  // branching one DWORD past the next instruction.
  const uint32_t word = build_s_call_b64(4, 1, GetParam().arch);
  for (const PcExportCase &test : kExportCases) {
    SCOPED_TRACE(::testing::Message() << "pc=" << std::hex << test.pc);
    wave->pc = test.pc;
    write_pair(4, 0xDEADBEEFDEADBEEFULL);
    ASSERT_NO_FATAL_FAILURE(execute(word));
    EXPECT_EQ(read_pair(4), expected_export(test));
    EXPECT_EQ(wave->pc, test.pc + 8);
  }
}

TEST_P(PcExtensionTest, SetpcPreservesTheTargetAddressWidth) {
  for (const PcTargetCase &test : kTargetCases) {
    SCOPED_TRACE(::testing::Message() << "target=" << std::hex << test.encoded);
    wave->pc = 0x1000;
    write_pair(4, test.encoded);
    ASSERT_NO_FATAL_FAILURE(execute(build_s_setpc_b64(4, GetParam().arch)));
    EXPECT_EQ(wave->pc, expected_target(test));
    EXPECT_EQ(read_pair(4), test.encoded);
  }
}

TEST_P(PcExtensionTest, SwappcPreservesTheTargetAndExportsTheReturnAddress) {
  for (bool alias : {false, true}) {
    SCOPED_TRACE(alias);
    for (const PcExportCase &start : kExportCases) {
      SCOPED_TRACE(::testing::Message() << "pc=" << std::hex << start.pc);
      for (const PcTargetCase &test : kTargetCases) {
        SCOPED_TRACE(::testing::Message() << "target=" << std::hex << test.encoded);
        const uint32_t source = alias ? 4 : 0;
        wave->pc = start.pc;
        write_pair(4, 0xDEADBEEFDEADBEEFULL);
        write_pair(source, test.encoded);
        ASSERT_NO_FATAL_FAILURE(execute(build_s_swappc_b64(4, source, GetParam().arch)));
        EXPECT_EQ(wave->pc, expected_target(test));
        EXPECT_EQ(read_pair(4), expected_export(start));
      }
    }
  }
}

TEST_P(PcExtensionTest, GetpcRoundTripsThroughIndirectBranchesWithLoadBias) {
  constexpr uint64_t kLoadBias = 0x00007FFF00000000ULL;
  const std::array starts{
      PcExportCase{0x0000FFFF12345000ULL, 0xFFFFFFFF12345004ULL, 0x0000FFFF12345004ULL,
                   0x0000FFFF12345004ULL},
      PcExportCase{0x0100000012345000ULL, 0, 0, 0xFF00000012345004ULL},
      PcExportCase{0x01FFFFFF12345000ULL, 0, 0, 0xFFFFFFFF12345004ULL},
  };
  wave->set_code_load_bias(kLoadBias);
  for (const PcExportCase &start : starts) {
    if (start.pc > 0x0000FFFFFFFFFFFFULL && GetParam().arch != ROCJITSU_CODE_ARCH_CDNA5)
      continue;
    SCOPED_TRACE(::testing::Message() << "pc=" << std::hex << start.pc);
    for (bool swap : {false, true}) {
      SCOPED_TRACE(swap);
      for (bool alias : {false, true}) {
        if (!swap && alias)
          continue;
        SCOPED_TRACE(alias);
        wave->pc = start.pc;
        write_pair(4, 0xDEADBEEFDEADBEEFULL);
        write_pair(6, 0xDEADBEEFDEADBEEFULL);
        ASSERT_NO_FATAL_FAILURE(execute(build_s_getpc_b64(4, GetParam().arch)));
        ASSERT_EQ(read_pair(4), expected_export(start));
        ASSERT_EQ(wave->pc, start.pc + 4);

        const uint32_t destination = alias ? 4 : 6;
        ASSERT_NO_FATAL_FAILURE(execute(swap ? build_s_swappc_b64(destination, 4, GetParam().arch)
                                             : build_s_setpc_b64(4, GetParam().arch)));
        EXPECT_EQ(wave->pc, start.pc + 4);
        if (swap) {
          EXPECT_EQ(read_pair(destination), expected_export(start) + 4);
        }
        if (!swap || !alias) {
          EXPECT_EQ(read_pair(4), expected_export(start));
        }
      }
    }
  }
}

TEST_P(PcExtensionTest, LoadBiasPreservesAbsoluteTargetsAndLinkedRel32) {
  constexpr uint64_t kLoadBias = 0x00007FFF00000000ULL;
  constexpr uint64_t kPositiveTag = 0x0001FFFF00001000ULL;
  constexpr uint64_t kNegativeTag = 0xFFFFFFFFFFFFF000ULL;
  wave->set_code_load_bias(kLoadBias);
  for (bool swap : {false, true}) {
    SCOPED_TRACE(swap);
    wave->pc = 0x2000;
    write_pair(0, kPositiveTag);
    ASSERT_NO_FATAL_FAILURE(execute(swap ? build_s_swappc_b64(4, 0, GetParam().arch)
                                         : build_s_setpc_b64(0, GetParam().arch)));
    const uint64_t expected = GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5 ? kPositiveTag
                              : swap                                      ? 0x2FF0ULL
                                                                          : kLoadBias + 0x1000;
    EXPECT_EQ(wave->pc, expected);

    wave->pc = 0x2000;
    write_pair(0, kNegativeTag);
    ASSERT_NO_FATAL_FAILURE(execute(swap ? build_s_swappc_b64(4, 0, GetParam().arch)
                                         : build_s_setpc_b64(0, GetParam().arch)));
    EXPECT_EQ(wave->pc, GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5 ? 0x01FFFFFFFFFFF000ULL
                                                                    : 0x0000FFFFFFFFF000ULL);
  }
}

} // namespace
