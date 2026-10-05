// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DbiSim's guest-memory access, and the global load/store encoders in
// global_memory_builders.h.
//
// Expected encodings are ground truth captured from
// `llvm-mc -triple=amdgcn -mcpu=<gfx942|gfx950|gfx1200> -show-encoding`.

#include "dbi_sim.h"
#include "global_memory_builders.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace test {
namespace {

// global_load_dword v1, v0, s[2:3] offset:64 and
// global_store_dword v0, v1, s[2:3] offset:64, identical on gfx942 and gfx950.
const std::vector<uint32_t> kCdnaLoad{0xdc508040, 0x01020000};
const std::vector<uint32_t> kCdnaStore{0xdc708040, 0x00020100};
// global_load_b32 v1, v0, s[2:3] offset:64 and
// global_store_b32 v0, v1, s[2:3] offset:64 on gfx1200.
const std::vector<uint32_t> kRdna4Load{0xee050002, 0x00000001, 0x00004000};
const std::vector<uint32_t> kRdna4Store{0xee068002, 0x00800000, 0x00004000};

TEST(GlobalMemoryBuilders, MatchLlvmOnCdna3) {
  EXPECT_EQ(build_global_load_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_CDNA3), kCdnaLoad);
  EXPECT_EQ(build_global_store_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_CDNA3), kCdnaStore);
}

TEST(GlobalMemoryBuilders, MatchLlvmOnCdna4) {
  EXPECT_EQ(build_global_load_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_CDNA4), kCdnaLoad);
  EXPECT_EQ(build_global_store_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_CDNA4), kCdnaStore);
}

TEST(GlobalMemoryBuilders, MatchLlvmOnRdna4) {
  EXPECT_EQ(build_global_load_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_RDNA4), kRdna4Load);
  EXPECT_EQ(build_global_store_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_RDNA4), kRdna4Store);
}

// The 64-, 96- and 128-bit forms. gfx942 and gfx950 accept a multi-dword VGPR
// operand only at an even base, so every tuple here starts on one:
//   global_load_dwordx2  v[6:7], v5, s[4:5] offset:64
//   global_store_dwordx2 v5, v[28:29], s[4:5] offset:64
//   global_store_dwordx3 v8, v[10:12], s[4:5] offset:4
//   global_store_dwordx4 v8, v[14:17], s[4:5] offset:16
// (b64/b96/b128 on gfx1200), identical on gfx942 and gfx950.
struct WideGlobalWords {
  std::vector<uint32_t> load_x2;
  std::vector<uint32_t> store_x2;
  std::vector<uint32_t> store_x3;
  std::vector<uint32_t> store_x4;
};

const WideGlobalWords kCdnaWide{{0xdc548040, 0x06040005},
                                {0xdc748040, 0x00041c05},
                                {0xdc788004, 0x00040a08},
                                {0xdc7c8010, 0x00040e08}};
const WideGlobalWords kRdna4Wide{{0xee054004, 0x00000006, 0x00004005},
                                 {0xee06c004, 0x0e000000, 0x00004005},
                                 {0xee070004, 0x05000000, 0x00000408},
                                 {0xee074004, 0x07000000, 0x00001008}};

void expect_wide_encodings(rj_code_arch_t arch, const WideGlobalWords &expected) {
  EXPECT_EQ(build_global_load_dwordx2(6, 5, 4, 64, arch), expected.load_x2);
  EXPECT_EQ(build_global_store_dwordx2(28, 5, 4, 64, arch), expected.store_x2);
  EXPECT_EQ(build_global_store_dwordx3(10, 8, 4, 4, arch), expected.store_x3);
  EXPECT_EQ(build_global_store_dwordx4(14, 8, 4, 16, arch), expected.store_x4);
}

TEST(GlobalMemoryBuilders, WideFormsMatchLlvmOnCdna3) {
  expect_wide_encodings(ROCJITSU_CODE_ARCH_CDNA3, kCdnaWide);
}
TEST(GlobalMemoryBuilders, WideFormsMatchLlvmOnCdna4) {
  expect_wide_encodings(ROCJITSU_CODE_ARCH_CDNA4, kCdnaWide);
}
TEST(GlobalMemoryBuilders, WideFormsMatchLlvmOnRdna4) {
  expect_wide_encodings(ROCJITSU_CODE_ARCH_RDNA4, kRdna4Wide);
}

// llvm-mc rejects an odd-based tuple on gfx942 and gfx950 ("vgpr tuples must be
// 64 bit aligned"); the simulator would not, so the encoder does.
TEST(GlobalMemoryBuilders, CdnaRefusesAnOddBasedTuple) {
  EXPECT_THROW((void)build_global_store_dwordx2(29, 5, 4, 0, ROCJITSU_CODE_ARCH_CDNA3),
               std::invalid_argument);
  EXPECT_THROW((void)build_global_load_dwordx2(7, 5, 4, 0, ROCJITSU_CODE_ARCH_CDNA4),
               std::invalid_argument);
  EXPECT_NO_THROW((void)build_global_store_dword(29, 5, 4, 0, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_NO_THROW((void)build_global_store_dwordx2(29, 5, 4, 0, ROCJITSU_CODE_ARCH_RDNA4));
}

// llvm-mc rejects an odd SGPR base on gfx942, gfx950 and gfx1200 ("invalid
// register alignment"); the decoder does not.
TEST(GlobalMemoryBuilders, RefusesAnOddScalarBase) {
  EXPECT_THROW((void)build_global_load_dword(1, 0, 3, 0, ROCJITSU_CODE_ARCH_CDNA3),
               std::invalid_argument);
  EXPECT_THROW((void)build_global_store_dwordx2(28, 5, 5, 0, ROCJITSU_CODE_ARCH_CDNA4),
               std::invalid_argument);
  EXPECT_THROW((void)build_global_store_dword(1, 0, 3, 0, ROCJITSU_CODE_ARCH_RDNA4),
               std::invalid_argument);
}

// The generated CDNA builder carries 12 offset bits; a larger offset is refused
// rather than truncated into a different address.
TEST(GlobalMemoryBuilders, RejectsAnOffsetPastTheCdnaField) {
  EXPECT_NO_THROW((void)build_global_store_dword(1, 0, 2, 0xFFF, ROCJITSU_CODE_ARCH_CDNA3));
  EXPECT_THROW((void)build_global_store_dword(1, 0, 2, 0x1000, ROCJITSU_CODE_ARCH_CDNA3),
               std::out_of_range);
  EXPECT_NO_THROW((void)build_global_store_dword(1, 0, 2, 0x1000, ROCJITSU_CODE_ARCH_RDNA4));
}

struct SimArch {
  std::string_view name;
  rj_code_arch_t arch;
  uint32_t wave_size;
};

constexpr SimArch kCdna3{"cdna3", ROCJITSU_CODE_ARCH_CDNA3, 64};
constexpr SimArch kCdna4{"cdna4", ROCJITSU_CODE_ARCH_CDNA4, 64};
constexpr SimArch kRdna4{"rdna4", ROCJITSU_CODE_ARCH_RDNA4, 32};

// Clear of the kernel, the kernargs and the queue rings DbiSim places.
constexpr uint64_t kSource = 0x80000;
constexpr uint32_t kDestinationDelta = 64;
constexpr uint32_t kSeed = 0xC0FFEE11;

constexpr uint16_t kScalarSrcLiteral = rdna4::OPR_SSRC_SRC_LITERAL;
static_assert(cdna3::OPR_SSRC_SRC_LITERAL == kScalarSrcLiteral &&
              cdna4::OPR_SSRC_SRC_LITERAL == kScalarSrcLiteral);
constexpr uint16_t kInlineZero = 128;

// Copies the dword at kSource to kSource + kDestinationDelta, addressing both
// through s[2:3] and a zero per-lane offset in v0. Without @p with_store the
// store is replaced by s_nops of the same length.
std::vector<uint32_t> make_copy_kernel(rj_code_arch_t arch, bool with_store) {
  std::vector<uint32_t> code{build_s_mov_b32(2, kScalarSrcLiteral, arch),
                             static_cast<uint32_t>(kSource), build_s_mov_b32(3, kInlineZero, arch),
                             build_v_mov_b32_src(0, kInlineZero, arch)};
  const std::vector<uint32_t> load = build_global_load_dword(1, 0, 2, 0, arch);
  code.insert(code.end(), load.begin(), load.end());
  code.push_back(build_wait_loads_complete(arch));
  const std::vector<uint32_t> store = build_global_store_dword(1, 0, 2, kDestinationDelta, arch);
  if (with_store)
    code.insert(code.end(), store.begin(), store.end());
  else
    code.insert(code.end(), store.size(), build_s_nop(0, arch));
  code.push_back(build_s_endpgm(arch));
  return code;
}

// Seeds kSource, runs the copy kernel, and returns the destination dword, or
// std::nullopt when no wave halted.
std::optional<uint32_t> run_copy(const SimArch &a, bool with_store) {
  DbiSim sim(a.name, a.wave_size);
  std::array<uint8_t, sizeof(kSeed)> seed{};
  std::memcpy(seed.data(), &kSeed, sizeof(kSeed));
  sim.write_memory(kSource, seed);
  const std::optional<std::vector<uint8_t>> bytes =
      sim.run_and_read_memory(make_copy_kernel(a.arch, with_store), /*private_bytes=*/0,
                              kSource + kDestinationDelta, sizeof(uint32_t));
  if (!bytes)
    return std::nullopt;
  uint32_t value = 0;
  std::memcpy(&value, bytes->data(), sizeof(value));
  return value;
}

void expect_seed_is_copied(const SimArch &a) {
  const std::optional<uint32_t> value = run_copy(a, /*with_store=*/true);
  ASSERT_TRUE(value.has_value()) << "no wave halted";
  EXPECT_EQ(*value, kSeed);
}

// Control: the destination holds the seed only because the kernel stored it
// there, not because the read-back returned the seeded source.
void expect_destination_untouched_without_the_store(const SimArch &a) {
  const std::optional<uint32_t> value = run_copy(a, /*with_store=*/false);
  ASSERT_TRUE(value.has_value()) << "no wave halted";
  EXPECT_EQ(*value, 0u);
}

TEST(DbiSimMemory, Cdna3KernelCopiesASeededWord) { expect_seed_is_copied(kCdna3); }
TEST(DbiSimMemory, Cdna4KernelCopiesASeededWord) { expect_seed_is_copied(kCdna4); }
TEST(DbiSimMemory, Rdna4KernelCopiesASeededWord) { expect_seed_is_copied(kRdna4); }

TEST(DbiSimMemory, Cdna3WithoutTheStoreTheDestinationIsUntouched) {
  expect_destination_untouched_without_the_store(kCdna3);
}
TEST(DbiSimMemory, Cdna4WithoutTheStoreTheDestinationIsUntouched) {
  expect_destination_untouched_without_the_store(kCdna4);
}
TEST(DbiSimMemory, Rdna4WithoutTheStoreTheDestinationIsUntouched) {
  expect_destination_untouched_without_the_store(kRdna4);
}

} // namespace
} // namespace test
} // namespace rocjitsu
