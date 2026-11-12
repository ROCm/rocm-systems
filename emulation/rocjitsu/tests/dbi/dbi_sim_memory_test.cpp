// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// DbiSim's guest-memory access, and the global load/store encoders in
// global_memory_builders.h.
//
// Expected encodings are ground truth captured from
// `llvm-mc -triple=amdgcn -mcpu=<gfx942|gfx950|gfx1200> -show-encoding`.

#include "dbi_sim.h"
#include "global_memory_builders.h"
#include "scalar_builders.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
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
  util::FailureOr<std::vector<uint32_t>> maybe_load =
      build_global_load_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_TRUE(maybe_load.succeeded());
  EXPECT_EQ(maybe_load.value(), kCdnaLoad);
  util::FailureOr<std::vector<uint32_t>> maybe_store =
      build_global_store_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_TRUE(maybe_store.succeeded());
  EXPECT_EQ(maybe_store.value(), kCdnaStore);
}

TEST(GlobalMemoryBuilders, MatchLlvmOnCdna4) {
  util::FailureOr<std::vector<uint32_t>> maybe_load =
      build_global_load_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_load.succeeded());
  EXPECT_EQ(maybe_load.value(), kCdnaLoad);
  util::FailureOr<std::vector<uint32_t>> maybe_store =
      build_global_store_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(maybe_store.succeeded());
  EXPECT_EQ(maybe_store.value(), kCdnaStore);
}

TEST(GlobalMemoryBuilders, MatchLlvmOnRdna4) {
  util::FailureOr<std::vector<uint32_t>> maybe_load =
      build_global_load_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_load.succeeded());
  EXPECT_EQ(maybe_load.value(), kRdna4Load);
  util::FailureOr<std::vector<uint32_t>> maybe_store =
      build_global_store_dword(1, 0, 2, 64, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_store.succeeded());
  EXPECT_EQ(maybe_store.value(), kRdna4Store);
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
  util::FailureOr<std::vector<uint32_t>> maybe_load_x2 =
      build_global_load_dwordx2(6, 5, 4, 64, arch);
  ASSERT_TRUE(maybe_load_x2.succeeded());
  EXPECT_EQ(maybe_load_x2.value(), expected.load_x2);
  util::FailureOr<std::vector<uint32_t>> maybe_store_x2 =
      build_global_store_dwordx2(28, 5, 4, 64, arch);
  ASSERT_TRUE(maybe_store_x2.succeeded());
  EXPECT_EQ(maybe_store_x2.value(), expected.store_x2);
  util::FailureOr<std::vector<uint32_t>> maybe_store_x3 =
      build_global_store_dwordx3(10, 8, 4, 4, arch);
  ASSERT_TRUE(maybe_store_x3.succeeded());
  EXPECT_EQ(maybe_store_x3.value(), expected.store_x3);
  util::FailureOr<std::vector<uint32_t>> maybe_store_x4 =
      build_global_store_dwordx4(14, 8, 4, 16, arch);
  ASSERT_TRUE(maybe_store_x4.succeeded());
  EXPECT_EQ(maybe_store_x4.value(), expected.store_x4);
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
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(
      build_global_store_dwordx2(29, 5, 4, 0, ROCJITSU_CODE_ARCH_CDNA3, diagnostic.emitter())
          .failed());
  EXPECT_EQ(diagnostic.message(), "multi-dword VGPR operand must start on an even register");
  EXPECT_TRUE(build_global_load_dwordx2(7, 5, 4, 0, ROCJITSU_CODE_ARCH_CDNA4).failed());
  EXPECT_TRUE(build_global_store_dword(29, 5, 4, 0, ROCJITSU_CODE_ARCH_CDNA3).succeeded());
  EXPECT_TRUE(build_global_store_dwordx2(29, 5, 4, 0, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
}

// llvm-mc rejects an odd SGPR base on gfx942, gfx950 and gfx1200 ("invalid
// register alignment"); the decoder does not.
TEST(GlobalMemoryBuilders, RefusesAnOddScalarBase) {
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(
      build_global_load_dword(1, 0, 3, 0, ROCJITSU_CODE_ARCH_CDNA3, diagnostic.emitter()).failed());
  EXPECT_EQ(diagnostic.message(), "SGPR base pair must start on an even register");
  EXPECT_TRUE(build_global_store_dwordx2(28, 5, 5, 0, ROCJITSU_CODE_ARCH_CDNA4).failed());
  EXPECT_TRUE(build_global_store_dword(1, 0, 3, 0, ROCJITSU_CODE_ARCH_RDNA4).failed());
}

// Past the ordinary SGPRs the field selects something other than an SGPR pair.
// llvm-mc reads gfx1200's 124 as `off` (the address then comes from v[0:1]) and
// its 106 as vcc, gfx942's and gfx950's 102 as flat_scratch and 106 as vcc, and
// the 7-bit field wraps 128 to s0 on every target.
TEST(GlobalMemoryBuilders, RefusesAScalarBasePastTheOrdinarySgprs) {
  for (const rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4})
    EXPECT_TRUE(build_global_store_dword(1, 0, 128, 0, arch).failed()) << "arch " << arch;
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(build_global_store_dword(1, 0, 124, 0, ROCJITSU_CODE_ARCH_RDNA4, diagnostic.emitter())
                  .failed());
  EXPECT_EQ(diagnostic.message(), "SGPR base pair must lie within the ordinary SGPRs");
  EXPECT_TRUE(build_global_load_dword(1, 0, 106, 0, ROCJITSU_CODE_ARCH_RDNA4).failed());
  EXPECT_TRUE(build_global_store_dword(1, 0, 102, 0, ROCJITSU_CODE_ARCH_CDNA3).failed());
  EXPECT_TRUE(build_global_load_dword(1, 0, 106, 0, ROCJITSU_CODE_ARCH_CDNA4).failed());
  // The last ordinary pair on each family.
  EXPECT_TRUE(build_global_store_dword(1, 0, 100, 0, ROCJITSU_CODE_ARCH_CDNA3).succeeded());
  EXPECT_TRUE(build_global_store_dword(1, 0, 104, 0, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
}

// The generated CDNA builder carries 12 offset bits; a larger offset is refused
// rather than truncated into a different address.
TEST(GlobalMemoryBuilders, RejectsAnOffsetPastTheCdnaField) {
  EXPECT_TRUE(build_global_store_dword(1, 0, 2, 0xFFF, ROCJITSU_CODE_ARCH_CDNA3).succeeded());
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(
      build_global_store_dword(1, 0, 2, 0x1000, ROCJITSU_CODE_ARCH_CDNA3, diagnostic.emitter())
          .failed());
  EXPECT_EQ(diagnostic.message(), "global memory offset does not fit the immediate field");
  EXPECT_TRUE(build_global_store_dword(1, 0, 2, 0x1000, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
}

TEST(GlobalMemoryBuilders, RejectUnsupportedArchitectures) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_INVALID, ROCJITSU_CODE_ARCH_CDNA1}) {
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(build_global_load_dword(1, 0, 2, 0, arch, diagnostic.emitter()).failed());
    EXPECT_EQ(diagnostic.message(),
              "Unimplemented instruction: global load for target architecture");
    EXPECT_TRUE(build_global_store_dword(1, 0, 2, 0, arch, diagnostic.emitter()).failed());
    EXPECT_EQ(diagnostic.message(),
              "Unimplemented instruction: global store for target architecture");
  }
}

TEST(GlobalMemoryBuilders, RejectsAnRdna4OffsetThatSetsTheSignBit) {
  EXPECT_TRUE(build_global_load_dword(1, 0, 2, 0x7fffff, ROCJITSU_CODE_ARCH_RDNA4).succeeded());
  EXPECT_TRUE(build_global_load_dword(1, 0, 2, 0x800000, ROCJITSU_CODE_ARCH_RDNA4).failed());
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

constexpr uint16_t kInlineZero = scalar_positive_inline_u32(0);

// Copies the dword at kSource to kSource + kDestinationDelta, addressing both
// through s[2:3] and a zero per-lane offset in v0. Without @p with_store the
// store is replaced by s_nops of the same length. Replaces @p code with the
// kernel words.
void make_copy_kernel(rj_code_arch_t arch, bool with_store, std::vector<uint32_t> &code) {
  const auto zero = build_v_mov_b32_src(0, kInlineZero, arch);
  ASSERT_TRUE(zero.succeeded());
  util::FailureOr<uint32_t> maybe_s_mov_b32_5 = build_s_mov_b32(3, kInlineZero, arch);
  ASSERT_TRUE(maybe_s_mov_b32_5.succeeded());
  util::FailureOr<uint32_t> maybe_s_mov_b32_6 = build_s_mov_b32(2, kScalarSrcLiteral, arch);
  ASSERT_TRUE(maybe_s_mov_b32_6.succeeded());
  code = {maybe_s_mov_b32_6.value(), static_cast<uint32_t>(kSource), maybe_s_mov_b32_5.value(),
          zero.value()};
  const util::FailureOr<std::vector<uint32_t>> maybe_load =
      build_global_load_dword(1, 0, 2, 0, arch);
  ASSERT_TRUE(maybe_load.succeeded());
  const std::vector<uint32_t> &load = maybe_load.value();
  code.insert(code.end(), load.begin(), load.end());
  const util::FailureOr<uint32_t> maybe_wait_loads_complete = build_wait_loads_complete(arch);
  ASSERT_TRUE(maybe_wait_loads_complete.succeeded());
  code.push_back(maybe_wait_loads_complete.value());
  const util::FailureOr<std::vector<uint32_t>> maybe_store =
      build_global_store_dword(1, 0, 2, kDestinationDelta, arch);
  ASSERT_TRUE(maybe_store.succeeded());
  const std::vector<uint32_t> &store = maybe_store.value();
  if (with_store)
    code.insert(code.end(), store.begin(), store.end());
  else {
    util::FailureOr<uint32_t> maybe_s_nop_4 = build_s_nop(0, arch);
    ASSERT_TRUE(maybe_s_nop_4.succeeded());
    code.insert(code.end(), store.size(), maybe_s_nop_4.value());
  }
  util::FailureOr<uint32_t> maybe_s_endpgm_3 = build_s_endpgm(arch);
  ASSERT_TRUE(maybe_s_endpgm_3.succeeded());
  code.push_back(maybe_s_endpgm_3.value());
}

// Seeds kSource, runs the copy kernel, and returns the destination dword, or
// std::nullopt when no wave halted.
std::optional<uint32_t> run_copy(const SimArch &target, bool with_store) {
  DbiSim sim;
  sim.initialize(target.name, target.wave_size);
  if (testing::Test::HasFatalFailure())
    return std::nullopt;
  std::array<uint8_t, sizeof(kSeed)> seed{};
  std::memcpy(seed.data(), &kSeed, sizeof(kSeed));
  sim.write_memory(kSource, seed);
  std::vector<uint32_t> code;
  make_copy_kernel(target.arch, with_store, code);
  if (testing::Test::HasFatalFailure())
    return std::nullopt;
  const std::optional<std::vector<uint8_t>> bytes = sim.run_and_read_memory(
      code, /*private_bytes=*/0, kSource + kDestinationDelta, sizeof(uint32_t));
  if (!bytes)
    return std::nullopt;
  uint32_t value = 0;
  std::memcpy(&value, bytes->data(), sizeof(value));
  return value;
}

void expect_seed_is_copied(const SimArch &target) {
  const std::optional<uint32_t> value = run_copy(target, /*with_store=*/true);
  ASSERT_TRUE(value.has_value()) << "no wave halted";
  EXPECT_EQ(*value, kSeed);
}

// Control: the destination holds the seed only because the kernel stored it
// there, not because the read-back returned the seeded source.
void expect_destination_untouched_without_the_store(const SimArch &target) {
  const std::optional<uint32_t> value = run_copy(target, /*with_store=*/false);
  ASSERT_TRUE(value.has_value()) << "no wave halted";
  EXPECT_EQ(*value, 0u);
}

// A DbiSim runs only its first dispatch, so a second one reads back nothing even
// though its kernel would halt, rather than the first dispatch's memory.
void expect_reuse_is_rejected(const SimArch &target) {
  DbiSim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize(target.name, target.wave_size));
  util::FailureOr<uint32_t> maybe_s_endpgm_2 = build_s_endpgm(target.arch);
  ASSERT_TRUE(maybe_s_endpgm_2.succeeded());
  const std::vector<uint32_t> halt{maybe_s_endpgm_2.value()};
  ASSERT_TRUE(
      sim.run_and_read_memory(halt, /*private_bytes=*/0, kSource, sizeof(uint32_t)).has_value())
      << "the first dispatch did not halt";
  EXPECT_FALSE(
      sim.run_and_read_memory(halt, /*private_bytes=*/0, kSource, sizeof(uint32_t)).has_value());
}

// A kernel that never halts reads back nothing, so its untouched memory cannot
// pass a "stays zero" control.
void expect_no_read_back_without_a_halt(const SimArch &target) {
  DbiSim sim;
  ASSERT_NO_FATAL_FAILURE(sim.initialize(target.name, target.wave_size));
  // s_branch to itself.
  util::FailureOr<uint32_t> maybe_s_branch_1 = build_s_branch(-1, target.arch);
  ASSERT_TRUE(maybe_s_branch_1.succeeded());
  EXPECT_FALSE(sim.run_and_read_memory({maybe_s_branch_1.value()},
                                       /*private_bytes=*/0, kSource, sizeof(uint32_t))
                   .has_value());
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

TEST(DbiSimMemory, Cdna3ReuseIsRejected) { expect_reuse_is_rejected(kCdna3); }
TEST(DbiSimMemory, Cdna4ReuseIsRejected) { expect_reuse_is_rejected(kCdna4); }
TEST(DbiSimMemory, Rdna4ReuseIsRejected) { expect_reuse_is_rejected(kRdna4); }

TEST(DbiSimMemory, Cdna3KernelThatNeverHaltsReadsNothingBack) {
  expect_no_read_back_without_a_halt(kCdna3);
}
TEST(DbiSimMemory, Cdna4KernelThatNeverHaltsReadsNothingBack) {
  expect_no_read_back_without_a_halt(kCdna4);
}
TEST(DbiSimMemory, Rdna4KernelThatNeverHaltsReadsNothingBack) {
  expect_no_read_back_without_a_halt(kRdna4);
}

} // namespace
} // namespace test
} // namespace rocjitsu
