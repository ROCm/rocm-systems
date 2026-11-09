// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "execution_fuzz_core.h"
#include "generated_kernel.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace rocjitsu::execution_fuzz {
namespace {

constexpr size_t kMinKernelInputSize = kHeaderSize + kMinGeneratedKernelBytes;

util::FailureOr<std::optional<uint64_t>> find_generated_kernel_seed(FuzzTarget target,
                                                                    uint64_t start_seed = 0) {
  // The random mix sets the wait-checking bit only for generated kernels.
  constexpr uint64_t kSeedSearchLimit = 256;
  for (uint64_t seed = start_seed; seed < start_seed + kSeedSearchLimit; ++seed) {
    const util::FailureOr<std::vector<uint8_t>> input = random_input(seed, kMaxInputSize, target);
    if (input.failed())
      return util::Result::failure();
    if (input.value()[1] & kWaitCheckingBit)
      return std::optional<uint64_t>{seed};
  }
  return std::optional<uint64_t>{};
}

TEST(ExecutionFuzzTargetTest, NamesCoverEveryExecutionTarget) {
  constexpr std::array<std::string_view, static_cast<size_t>(FuzzTarget::Count)> expected = {
      "gfx908",  "gfx90a",  "gfx942",  "gfx950",  "gfx1010", "gfx1030",
      "gfx1100", "gfx1151", "gfx1200", "gfx1201", "gfx1250", "gfx1251"};
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    const FuzzTarget target = *indexed_target(index);
    EXPECT_EQ(target_name(target), expected[index]);
    EXPECT_EQ(parse_target(expected[index]), target);
  }
  EXPECT_FALSE(parse_target("unknown"));
  EXPECT_TRUE(targets_cover_amdgpu_execution_registry());
}

TEST(ExecutionFuzzTargetTest, InputSelectorExecutesOnEveryTarget) {
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    SCOPED_TRACE(target_name(*indexed_target(index)));
    const std::array<uint8_t, 1> input = {index};
    const util::FailureOr<RunStats> maybe_stats = run_input(input);
    ASSERT_TRUE(maybe_stats.succeeded());
    const RunStats stats = maybe_stats.value();
    EXPECT_EQ(stats.executed, 1u);
    EXPECT_EQ(stats.waves_halted, 1u);
    EXPECT_EQ(stats.inputs_executed, 1u);
  }
}

TEST(ExecutionFuzzTargetTest, UnassignedSelectorsDoNotWrapToAnotherTarget) {
  for (unsigned selector = static_cast<uint8_t>(FuzzTarget::Count); selector < 256; ++selector) {
    const auto byte = static_cast<uint8_t>(selector);
    EXPECT_FALSE(indexed_target(byte));
    const util::FailureOr<RunStats> maybe_stats = run_input(std::array<uint8_t, 1>{byte});
    ASSERT_TRUE(maybe_stats.succeeded());
    const RunStats stats = maybe_stats.value();
    EXPECT_EQ(stats.inputs, 1u);
    EXPECT_EQ(stats.executed, 0u);
    EXPECT_EQ(stats.inputs_executed, 0u);
  }
  const util::FailureOr<RunStats> maybe_overridden =
      run_input(std::array<uint8_t, 1>{255}, FuzzTarget::Gfx1250);
  ASSERT_TRUE(maybe_overridden.succeeded());
  const RunStats overridden = maybe_overridden.value();
  EXPECT_EQ(overridden.waves_halted, 1u);
}

TEST(ExecutionFuzzTargetTest, GeneratedEncodingsReachExecutionAcrossTargets) {
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    SCOPED_TRACE(target_name(*indexed_target(index)));
    const FuzzTarget target = *indexed_target(index);
    const util::FailureOr<std::vector<uint8_t>> maybe_input =
        random_input(17 + index * 4, 64, target);
    ASSERT_TRUE(maybe_input.succeeded());
    const util::FailureOr<RunStats> maybe_stats = run_input(maybe_input.value(), target);
    ASSERT_TRUE(maybe_stats.succeeded());
    const RunStats stats = maybe_stats.value();
    EXPECT_EQ(stats.inputs, 1u);
    EXPECT_GT(stats.executed, 0u);
    EXPECT_EQ(stats.inputs_executed, 1u);
  }
}

TEST(ExecutionFuzzTargetTest, RandomMixRunsGeneratedKernelsAcrossTargets) {
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    const FuzzTarget target = *indexed_target(index);
    SCOPED_TRACE(target_name(target));
    for (uint64_t start_seed : {0u, 64u, 128u, 192u}) {
      const util::FailureOr<std::optional<uint64_t>> maybe_seed =
          find_generated_kernel_seed(target, start_seed);
      ASSERT_TRUE(maybe_seed.succeeded());
      const std::optional<uint64_t> seed = maybe_seed.value();
      ASSERT_TRUE(seed);
      SCOPED_TRACE(*seed);
      for (size_t size : {kMinKernelInputSize, kMinKernelInputSize + 1, kMinKernelInputSize + 2,
                          kMinKernelInputSize + 3, kMaxInputSize}) {
        SCOPED_TRACE(size);
        const util::FailureOr<std::vector<uint8_t>> maybe_input = random_input(*seed, size, target);
        ASSERT_TRUE(maybe_input.succeeded());
        std::vector<uint8_t> input = maybe_input.value();
        const util::FailureOr<std::vector<uint8_t>> maybe_kernel =
            generated_kernel_input(*seed, size, target);
        ASSERT_TRUE(maybe_kernel.succeeded());
        ASSERT_EQ(input, maybe_kernel.value());
        for (uint8_t memory_mode : {uint8_t{0}, kCacheMemoryBit}) {
          SCOPED_TRACE(memory_mode != 0);
          input[1] = (input[1] & ~kCacheMemoryBit) | memory_mode;
          testing::internal::CaptureStderr();
          const util::FailureOr<RunStats> maybe_stats =
              run_input(input, std::nullopt, nullptr, AsyncMode::Disabled, 1);
          const std::string diagnostics = testing::internal::GetCapturedStderr();
          ASSERT_TRUE(maybe_stats.succeeded());
          const RunStats stats = maybe_stats.value();
          EXPECT_EQ(diagnostics.find("memory-wait:"), std::string::npos) << diagnostics;
          EXPECT_EQ(diagnostics.find("xcnt-wait:"), std::string::npos) << diagnostics;
          EXPECT_GT(stats.executed, size == kMaxInputSize ? 10u : 2u);
          EXPECT_EQ(stats.inputs_halted, 1u);
          EXPECT_EQ(stats.inputs_stopped, 0u);
          EXPECT_EQ(stats.budget_exhausted, 0u);
        }
      }
    }
  }
}

TEST(ExecutionFuzzInputTest, TrailingLiteralLeavesEnoughBytesToFetchLastTerminator) {
  for (const FuzzTarget target : {FuzzTarget::Gfx908, FuzzTarget::Gfx90a, FuzzTarget::Gfx942}) {
    SCOPED_TRACE(target_name(target));
    for (const uint8_t memory_mode : {uint8_t{0}, kCacheMemoryBit}) {
      SCOPED_TRACE(memory_mode);
      // s_mov_b32 s0, literal. It consumes the first appended terminator as its literal.
      const std::array<uint8_t, 6> input = {
          static_cast<uint8_t>(target), memory_mode, 0xff, 0x00, 0x80, 0xbe};
      const util::FailureOr<RunStats> maybe_stats =
          run_input(input, std::nullopt, nullptr, AsyncMode::Disabled, 1);
      ASSERT_TRUE(maybe_stats.succeeded());
      const RunStats stats = maybe_stats.value();
      EXPECT_EQ(stats.waves_halted, 1u);
    }
  }
}

TEST(ExecutionFuzzInputTest, GeneratedKernelBranchTargetsAndMemoryWaits) {
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    const FuzzTarget target = *indexed_target(index);
    SCOPED_TRACE(target_name(target));
    const rj_code_arch_t arch = target_arch(target);
    const amdgpu::WaitcntModel wait_model = amdgpu::waitcnt_model(arch).value();
    std::unique_ptr<Decoder> decoder = Decoder::create(arch);
    ASSERT_TRUE(decoder);
    bool forward = false;
    bool backward = false;
    bool scalar_memory = false;
    bool vector_load = false;
    bool vector_store = false;
    for (uint64_t seed = 0; seed != 12; ++seed) {
      SCOPED_TRACE(seed);
      const util::FailureOr<std::vector<uint8_t>> maybe_input =
          generated_kernel_input(seed, kMaxInputSize, target);
      ASSERT_TRUE(maybe_input.succeeded());
      const std::vector<uint8_t> input = maybe_input.value();
      std::vector<uint32_t> words((input.size() - kHeaderSize) / sizeof(uint32_t));
      for (size_t word_index = 0; word_index < words.size(); ++word_index)
        for (unsigned byte = 0; byte < sizeof(uint32_t); ++byte)
          words[word_index] |= uint32_t{input[kHeaderSize + word_index * sizeof(uint32_t) + byte]}
                               << (8 * byte);
      std::vector<int64_t> branchable_word_offsets;
      std::vector<int64_t> branch_target_words;
      std::bitset<amdgpu::kWaitCounterCount> pending_counters;
      bool terminated = false;
      for (size_t word_index = 0; word_index < words.size();) {
        const DecodeResult result = decoder->decode_window(std::span(words).subspan(word_index));
        ASSERT_TRUE(result.succeeded()) << word_index;
        const Instruction &instruction = *result.value();
        if (pending_counters.any()) {
          ASSERT_TRUE(instruction.is_waitcnt()) << word_index;
          const auto wait = amdgpu::WaitCounterPolicy::explicit_wait_fields(instruction, arch);
          ASSERT_TRUE(wait.succeeded() && wait.value().has_value()) << word_index;
          const amdgpu::WaitFields &fields = *wait.value();
          for (size_t counter_index = 0; counter_index < fields.size(); ++counter_index)
            if (fields[counter_index] == 0)
              pending_counters.reset(counter_index);
        }
        if (!instruction.is_waitcnt())
          branchable_word_offsets.push_back(word_index);
        if (instruction.flags() & PROGRAM_TERMINATOR) {
          terminated = true;
          break;
        }
        if (instruction.is_branch()) {
          const int16_t displacement =
              std::bit_cast<int16_t>(static_cast<uint16_t>(words[word_index]));
          const int64_t target_word = static_cast<int64_t>(word_index) + 1 + displacement;
          forward |= target_word > static_cast<int64_t>(word_index);
          backward |= target_word < static_cast<int64_t>(word_index);
          branch_target_words.push_back(target_word);
        }
        if (instruction.is_memory_op()) {
          const amdgpu::MemoryIssueInfo *memory = instruction.amdgpu_memory_issue_info();
          ASSERT_NE(memory, nullptr) << instruction.mnemonic();
          scalar_memory |= !memory->exec_masked;
          const Operand *destination = instruction.dst_operand(0);
          ASSERT_NE(destination, nullptr) << instruction.mnemonic();
          const bool writes_register = destination->to_register_ref().has_value();
          vector_load |= memory->exec_masked && writes_register;
          vector_store |= memory->exec_masked && !writes_register;
          // Check every possible domain before runtime routing filters the FLAT domains.
          for (amdgpu::MemoryCounterObligation obligation : memory->counter_obligations()) {
            const amdgpu::WaitCounterKind counter =
                amdgpu::canonical_wait_counter(obligation.wait_counter_type(), wait_model);
            ASSERT_NE(counter, amdgpu::WaitCounterKind::Count);
            pending_counters.set(static_cast<size_t>(counter));
          }
        }
        word_index += instruction.size() / sizeof(uint32_t);
      }
      EXPECT_TRUE(terminated);
      EXPECT_TRUE(pending_counters.none());
      for (int64_t target_word : branch_target_words)
        EXPECT_NE(std::ranges::find(branchable_word_offsets, target_word),
                  branchable_word_offsets.end());
    }
    EXPECT_TRUE(forward);
    EXPECT_TRUE(backward);
    EXPECT_TRUE(scalar_memory);
    EXPECT_TRUE(vector_load);
    EXPECT_TRUE(vector_store);
  }
}

TEST(ExecutionFuzzInputTest, GeneratedInputsRespectEverySizeAndTarget) {
  for (uint8_t index = 0; index < static_cast<uint8_t>(FuzzTarget::Count); ++index) {
    const FuzzTarget target = *indexed_target(index);
    SCOPED_TRACE(target_name(target));
    const util::FailureOr<std::optional<uint64_t>> maybe_kernel_seed =
        find_generated_kernel_seed(target);
    ASSERT_TRUE(maybe_kernel_seed.succeeded());
    const std::optional<uint64_t> kernel_seed = maybe_kernel_seed.value();
    ASSERT_TRUE(kernel_seed);
    for (size_t size = 1; size <= kMaxInputSize; ++size) {
      SCOPED_TRACE(size);
      for (uint64_t seed : {uint64_t{0}, uint64_t{1}, uint64_t{17}, *kernel_seed,
                            std::numeric_limits<uint64_t>::max()}) {
        SCOPED_TRACE(seed);
        const util::FailureOr<std::vector<uint8_t>> maybe_input = random_input(seed, size, target);
        ASSERT_TRUE(maybe_input.succeeded());
        const std::vector<uint8_t> input = maybe_input.value();
        ASSERT_EQ(input.size(), size);
        EXPECT_EQ(input.front(), static_cast<uint8_t>(target));
        const util::FailureOr<std::vector<uint8_t>> maybe_input_2 =
            random_input(seed, size, target);
        ASSERT_TRUE(maybe_input_2.succeeded());
        EXPECT_EQ(input, maybe_input_2.value());
        if (size >= kHeaderSize) {
          EXPECT_EQ(input[1] & ~(kAsyncSelectBit | kConcurrentIssuerMask | kFullExecMaskBit |
                                 kWaitCheckingBit | kCacheMemoryBit),
                    0);
          if (seed == *kernel_seed) {
            EXPECT_EQ((input[1] & kWaitCheckingBit) != 0, size >= kMinKernelInputSize);
          }
        }
      }
    }
  }
}

TEST(ExecutionFuzzInputTest, GeneratedInputsRejectInvalidSizes) {
  for (size_t size : {size_t{0}, kMaxInputSize + 1, std::numeric_limits<size_t>::max()}) {
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(random_input(0, size, std::nullopt, diagnostic.emitter()).failed());
    EXPECT_FALSE(diagnostic.message().empty());
  }
  for (size_t size : {size_t{0}, kHeaderSize, kMinKernelInputSize - 1, kMaxInputSize + 1,
                      std::numeric_limits<size_t>::max()}) {
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(
        generated_kernel_input(0, size, FuzzTarget::Gfx1250, diagnostic.emitter()).failed());
    EXPECT_FALSE(diagnostic.message().empty());
  }
}

TEST(ExecutionFuzzInputTest, InvalidHarnessArgumentsReturnDiagnostics) {
  const std::array<uint8_t, 1> input = {0};
  for (const unsigned issuers : {0u, 9u}) {
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(
        run_input(input, std::nullopt, nullptr, AsyncMode::Disabled, issuers, diagnostic.emitter())
            .failed());
    EXPECT_EQ(diagnostic.message(), "execution fuzz issuers must be 1..8");
  }
  for (const size_t size : {size_t{0}, kMaxInputSize + 1}) {
    const std::vector<uint8_t> bytes(size);
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(
        run_input(bytes, std::nullopt, nullptr, AsyncMode::Disabled, 1, diagnostic.emitter())
            .failed());
    EXPECT_EQ(diagnostic.message(), "execution fuzz input must contain 1..1024 bytes");
  }
  util::StringDiagnostic async_diagnostic;
  EXPECT_TRUE(
      run_input(input, std::nullopt, nullptr, AsyncMode::Enabled, 1, async_diagnostic.emitter())
          .failed());
  EXPECT_EQ(async_diagnostic.message(), "enabled asynchronous execution requires helper resources");
  for (const unsigned helpers : {0u, 129u}) {
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(make_async_execution_resources(helpers, diagnostic.emitter()).failed());
    EXPECT_EQ(diagnostic.message(), "async helper count must be 1..128");
  }
}

TEST(ExecutionFuzzInputTest, InvalidTargetReturnsFailureBeforeMetadataLookup) {
  const FuzzTarget invalid = FuzzTarget::Count;
  const std::array<uint8_t, 1> input = {0};
  util::StringDiagnostic execution;
  EXPECT_TRUE(
      run_input(input, invalid, nullptr, AsyncMode::Disabled, 1, execution.emitter()).failed());
  EXPECT_EQ(execution.message(), "invalid execution fuzz target");
  util::StringDiagnostic random;
  EXPECT_TRUE(random_input(0, 32, invalid, random.emitter()).failed());
  EXPECT_EQ(random.message(), "invalid execution fuzz target");
  util::StringDiagnostic kernel;
  EXPECT_TRUE(generated_kernel_input(0, kMaxInputSize, invalid, kernel.emitter()).failed());
  EXPECT_EQ(kernel.message(), "invalid execution fuzz target");
}

TEST(ExecutionFuzzInputTest, GeneratedKernelRejectsArgumentsWithoutChangingPayload) {
  for (const bool invalid_arch : {false, true}) {
    std::vector<uint8_t> payload(
        invalid_arch ? kMinGeneratedKernelBytes : kMinGeneratedKernelBytes - 1, 0x5a);
    const std::vector<uint8_t> original = payload;
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(write_generated_kernel(
                    payload, invalid_arch ? ROCJITSU_CODE_ARCH_RV64I : ROCJITSU_CODE_ARCH_CDNA5, 0,
                    diagnostic.emitter())
                    .failed());
    EXPECT_FALSE(diagnostic.message().empty());
    EXPECT_EQ(payload, original);
  }
}

TEST(ExecutionFuzzInputTest, ArbitraryInputsPreserveByteStreamAcrossSizes) {
  const std::vector<uint8_t> expected = {0x02, 0, 0xaf, 0xf4, 0x4f, 0xec, 0x9b, 0xea};
  const util::FailureOr<std::vector<uint8_t>> maybe_input =
      random_input(0, expected.size(), FuzzTarget::Gfx942);
  ASSERT_TRUE(maybe_input.succeeded());
  std::vector<uint8_t> input = maybe_input.value();
  input[1] &= ~kCacheMemoryBit;
  EXPECT_EQ(input, expected);

  const util::FailureOr<std::vector<uint8_t>> maybe_full_input =
      random_input(0, kMaxInputSize, FuzzTarget::Gfx942);
  ASSERT_TRUE(maybe_full_input.succeeded());
  const std::vector<uint8_t> full_input = maybe_full_input.value();
  ASSERT_EQ(full_input.size(), kMaxInputSize);
  for (size_t size = 1; size < kMaxInputSize; ++size) {
    SCOPED_TRACE(size);
    const util::FailureOr<std::vector<uint8_t>> maybe_input_2 =
        random_input(0, size, FuzzTarget::Gfx942);
    ASSERT_TRUE(maybe_input_2.succeeded());
    const std::vector<uint8_t> input = maybe_input_2.value();
    EXPECT_EQ(input, (std::vector<uint8_t>{full_input.begin(), full_input.begin() + size}));
  }
}

TEST(ExecutionFuzzInputTest, ArbitraryInputsSampleBothCacheModesForEitherFirstByteBit) {
  constexpr uint64_t kRawSeedStride = 8;
  std::bitset<4> observed;
  for (uint64_t seed = 0; seed < 64 * kRawSeedStride; seed += kRawSeedStride) {
    const util::FailureOr<std::vector<uint8_t>> maybe_input =
        random_input(seed, 64, FuzzTarget::Gfx942);
    ASSERT_TRUE(maybe_input.succeeded());
    const std::vector<uint8_t> input = maybe_input.value();
    const bool cache_enabled = (input[1] & kCacheMemoryBit) != 0;
    const bool first_byte_bit = (input[kHeaderSize] & 8) != 0;
    observed.set(2 * cache_enabled + first_byte_bit);
  }
  EXPECT_TRUE(observed.all()) << observed;
}

TEST(ExecutionFuzzInputTest, GeneratedInstructionsPreservePrefixesAcrossSizes) {
  constexpr uint64_t seed = 2;
  for (FuzzTarget target : {FuzzTarget::Gfx942, FuzzTarget::Gfx1250}) {
    SCOPED_TRACE(target_name(target));
    const util::FailureOr<std::vector<uint8_t>> maybe_full_input =
        random_input(seed, kMaxInputSize, target);
    ASSERT_TRUE(maybe_full_input.succeeded());
    const std::vector<uint8_t> full_input = maybe_full_input.value();
    ASSERT_EQ(full_input.size(), kMaxInputSize);
    for (size_t size = 1; size < kMaxInputSize; ++size) {
      SCOPED_TRACE(size);
      const util::FailureOr<std::vector<uint8_t>> maybe_input = random_input(seed, size, target);
      ASSERT_TRUE(maybe_input.succeeded());
      const std::vector<uint8_t> input = maybe_input.value();
      ASSERT_EQ(input.size(), size);
      // Each input has one sparse bit mutation, so their common prefix can
      // differ in at most two bits when those mutations land differently.
      unsigned differing_bits = 0;
      for (size_t byte = 0; byte < size; ++byte)
        differing_bits += std::popcount(static_cast<unsigned>(input[byte] ^ full_input[byte]));
      EXPECT_LE(differing_bits, 2u);
    }
  }
}

TEST(ExecutionFuzzInputTest, EntryMutationsStayWithinThePrefix) {
  constexpr size_t short_size = 64;
  constexpr size_t mutation_end = kHeaderSize + 32;
  bool saw_mutation = false;
  for (uint64_t seed : {258u, 259u, 261u, 262u, 263u}) {
    SCOPED_TRACE(seed);
    const util::FailureOr<std::vector<uint8_t>> maybe_small =
        random_input(seed, short_size, FuzzTarget::Gfx942);
    ASSERT_TRUE(maybe_small.succeeded());
    const std::vector<uint8_t> small = maybe_small.value();
    const util::FailureOr<std::vector<uint8_t>> maybe_large =
        random_input(seed, kMaxInputSize, FuzzTarget::Gfx942);
    ASSERT_TRUE(maybe_large.succeeded());
    const std::vector<uint8_t> large = maybe_large.value();
    saw_mutation |= !std::equal(small.begin() + kHeaderSize, small.begin() + mutation_end,
                                large.begin() + kHeaderSize);
    EXPECT_TRUE(
        std::equal(small.begin() + mutation_end, small.end(), large.begin() + mutation_end));
  }
  EXPECT_TRUE(saw_mutation);
}

TEST(ExecutionFuzzInputTest, WaitCheckingBitEnablesDiagnosticsOnReplay) {
  std::vector<uint8_t> input{static_cast<uint8_t>(FuzzTarget::Gfx1250), 0};
  const auto append = [&](auto words) {
    for (const uint32_t word : words)
      for (unsigned byte = 0; byte < sizeof(word); ++byte)
        input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
  };
  append(cdna5::build_smem(cdna5::kSLoadB32Smem, {.sbase = 0, .sdata = 20, .soffset = 124}));
  append(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 20, .sdst = 21}));
  append(cdna5::build_sop1(cdna5::kSMovB32Sop1, {.ssrc0 = 128, .sdst = 0}));

  for (bool check_waits : {false, true}) {
    input[1] = check_waits ? kWaitCheckingBit : 0;
    testing::internal::CaptureStderr();
    const util::FailureOr<RunStats> maybe_stats =
        run_input(input, std::nullopt, nullptr, AsyncMode::Disabled, 1);
    const std::string diagnostics = testing::internal::GetCapturedStderr();
    ASSERT_TRUE(maybe_stats.succeeded());
    const RunStats stats = maybe_stats.value();
    EXPECT_EQ(diagnostics.find("memory-wait:") != std::string::npos, check_waits) << diagnostics;
    EXPECT_EQ(diagnostics.find("xcnt-wait:") != std::string::npos, check_waits) << diagnostics;
    EXPECT_EQ(stats.executed, 4u);
    EXPECT_EQ(stats.inputs_halted, 1u);
  }
}

TEST(ExecutionFuzzInputTest, WmmaPrefixStartsAtExactCapacity) {
  constexpr std::array first_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
                         {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 320, .opsel_hi = 3});
  constexpr std::array second_wmma =
      cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
                         {.vdst = 96, .src0 = 256, .src1 = 288, .src2 = 352, .opsel_hi = 3});
  constexpr size_t input_size =
      kHeaderSize + (first_wmma.size() + second_wmma.size()) * sizeof(uint32_t);
  struct Case {
    uint64_t seed;
    uint8_t short_mode;
    uint8_t prefix_mode;
  };
  constexpr std::array cases = {
      Case{1, 0, kFullExecMaskBit},
      Case{17, kAsyncSelectBit, kAsyncSelectBit | kFullExecMaskBit},
  };
  for (FuzzTarget target : {FuzzTarget::Gfx1250, FuzzTarget::Gfx1251}) {
    SCOPED_TRACE(target_name(target));
    for (const Case &test_case : cases) {
      SCOPED_TRACE(test_case.seed);
      const util::FailureOr<std::vector<uint8_t>> maybe_short_input =
          random_input(test_case.seed, input_size - 1, target);
      ASSERT_TRUE(maybe_short_input.succeeded());
      const std::vector<uint8_t> short_input = maybe_short_input.value();
      ASSERT_EQ(short_input.size(), input_size - 1);
      EXPECT_EQ(short_input[1] & ~kCacheMemoryBit, test_case.short_mode);

      const uint8_t cache_mode = short_input[1] & kCacheMemoryBit;
      std::vector<uint8_t> expected{static_cast<uint8_t>(target),
                                    static_cast<uint8_t>(test_case.prefix_mode | cache_mode)};
      const auto append = [&](const auto &words) {
        for (const uint32_t word : words)
          for (unsigned byte = 0; byte < sizeof(word); ++byte)
            expected.push_back(static_cast<uint8_t>(word >> (8 * byte)));
      };
      auto first = first_wmma, second = second_wmma;
      if (test_case.seed & 16) {
        first[0] = (first[0] & 0xff00ffffU) | (cdna5::kVWmmaF3232x16x128F4Vop3p << 16);
        second[0] = (second[0] & 0xff00ffffU) | (cdna5::kVWmmaF3232x16x128F4Vop3p << 16);
      }
      append(first);
      append(second);
      ASSERT_EQ(expected.size(), input_size);
      const util::FailureOr<std::vector<uint8_t>> maybe_input =
          random_input(test_case.seed, input_size, target);
      ASSERT_TRUE(maybe_input.succeeded());
      EXPECT_EQ(maybe_input.value(), expected);
    }
  }
}

TEST(ExecutionFuzzInputTest, InitialAddressPairsReachMappedMemory) {
  for (const bool scalar : {false, true}) {
    for (const uint8_t reg : {0, 6, 14}) {
      SCOPED_TRACE(scalar);
      SCOPED_TRACE(reg);
      const FuzzTarget target = scalar ? FuzzTarget::Gfx1250 : FuzzTarget::Gfx942;
      std::vector<uint8_t> input{static_cast<uint8_t>(target), 0};
      const auto append = [&](auto words) {
        for (const uint32_t word : words)
          for (unsigned byte = 0; byte < sizeof(word); ++byte)
            input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
      };
      if (scalar) {
        append(cdna5::build_smem(
            cdna5::kSLoadB32Smem,
            {.sbase = static_cast<uint8_t>(reg / 2), .sdata = 20, .soffset = 124}));
        append(cdna5::build_sopp(cdna5::kSWaitKmcntSopp));
        append(cdna5::build_sopp(cdna5::kSNopSopp));
      } else {
        append(cdna3::build_flat(cdna3::kFlatLoadDwordFlat, {.addr = reg, .vdst = 20}));
        append(cdna3::build_sopp(cdna3::kSWaitcntSopp));
        append(cdna3::build_sopp(cdna3::kSNopSopp));
      }
      for (uint8_t mode : {uint8_t{0}, kCacheMemoryBit}) {
        SCOPED_TRACE(mode);
        input[1] = mode;
        const util::FailureOr<RunStats> maybe_stats =
            run_input(input, target, nullptr, AsyncMode::Disabled, 1);
        ASSERT_TRUE(maybe_stats.succeeded());
        const RunStats stats = maybe_stats.value();
        EXPECT_EQ(stats.executed, 4u);
        EXPECT_EQ(stats.inputs_halted, 1u);
        EXPECT_EQ(stats.inputs_stopped, 0u);
        EXPECT_EQ(stats.budget_exhausted, 0u);
        EXPECT_EQ(stats.l2_backing_reads != 0, mode == kCacheMemoryBit);
      }
    }
  }
}

TEST(ExecutionFuzzInputTest, GwsBarrierWithoutInitExecutesFromHarnessState) {
  for (const uint8_t rid : {0, 2, 34, 59, 63}) {
    SCOPED_TRACE(rid);
    std::vector<uint8_t> input{static_cast<uint8_t>(FuzzTarget::Gfx942), 0};
    const auto append = [&](auto words) {
      for (const uint32_t word : words)
        for (unsigned byte = 0; byte < sizeof(word); ++byte)
          input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
    };
    append(cdna3::build_ds(cdna3::kDsGwsBarrierDs, {.offset0 = rid, .gds = 1, .addr = 0}));
    const util::FailureOr<RunStats> maybe_stats =
        run_input(input, FuzzTarget::Gfx942, nullptr, AsyncMode::Disabled, 1);
    ASSERT_TRUE(maybe_stats.succeeded());
    const RunStats stats = maybe_stats.value();
    EXPECT_EQ(stats.inputs, 1u);
    EXPECT_EQ(stats.executed, 2u);
    EXPECT_EQ(stats.inputs_halted, 1u);
    EXPECT_EQ(stats.budget_exhausted, 0u);
  }
}

TEST(ExecutionFuzzInputTest, GeneratedInputsSelectSyncAsyncAndConcurrentExecution) {
  const util::FailureOr<std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>>
      maybe_resources = make_async_execution_resources(2);
  ASSERT_TRUE(maybe_resources.succeeded());
  auto resources = maybe_resources.value();
  struct Case {
    uint64_t seed;
    bool async;
    bool concurrent;
  };
  constexpr std::array cases = {
      Case{1, false, false},
      Case{17, true, false},
      Case{97, false, true},
      Case{113, true, true},
  };
  for (const auto &[seed, async, concurrent] : cases) {
    SCOPED_TRACE(seed);
    const util::FailureOr<std::vector<uint8_t>> maybe_input =
        random_input(seed, 24, FuzzTarget::Gfx1250);
    ASSERT_TRUE(maybe_input.succeeded());
    const util::FailureOr<RunStats> maybe_stats =
        run_input(maybe_input.value(), FuzzTarget::Gfx1250, resources);
    ASSERT_TRUE(maybe_stats.succeeded());
    const RunStats stats = maybe_stats.value();
    EXPECT_EQ(stats.async_capable_inputs, 1u);
    EXPECT_EQ(stats.async_enabled_inputs, async);
    EXPECT_EQ(stats.concurrent_inputs, concurrent);
    EXPECT_EQ(stats.issuer_executions, concurrent ? 2u : 1u);
    // Odd rounds use the FP4 prefix; even rounds keep migrated FP8 on the issuer.
    if (async) {
      EXPECT_GT(stats.async_executed, 0u);
    } else {
      EXPECT_EQ(stats.async_executed, 0u);
    }
  }
}

TEST(ExecutionFuzzInputTest, ModeByteDoesNotRewriteAsyncWmmaPrefix) {
  for (uint64_t seed : {1u, 17u, 97u, 113u}) {
    const util::FailureOr<std::vector<uint8_t>> maybe_input =
        random_input(seed, 24, FuzzTarget::Gfx1250);
    ASSERT_TRUE(maybe_input.succeeded());
    const auto input = maybe_input.value();
    ASSERT_GE(input.size(), 18u);
    EXPECT_EQ(input[17], 0x1du);
  }
  constexpr size_t wmma_prefix_end = kHeaderSize + 16;
  const util::FailureOr<std::vector<uint8_t>> maybe_wmma_prefix =
      random_input(257, wmma_prefix_end, FuzzTarget::Gfx1250);
  ASSERT_TRUE(maybe_wmma_prefix.succeeded());
  const std::vector<uint8_t> wmma_prefix = maybe_wmma_prefix.value();
  const util::FailureOr<std::vector<uint8_t>> maybe_entry_mutated =
      random_input(257, 32, FuzzTarget::Gfx1250);
  ASSERT_TRUE(maybe_entry_mutated.succeeded());
  const std::vector<uint8_t> entry_mutated = maybe_entry_mutated.value();
  EXPECT_TRUE(std::ranges::equal(wmma_prefix, std::span(entry_mutated).first(wmma_prefix_end)));
}

TEST(ExecutionFuzzInputTest, TargetOverrideKeepsInputLayout) {
  constexpr uint64_t kGfx1250Selector = static_cast<uint8_t>(FuzzTarget::Gfx1250);
  const util::FailureOr<std::vector<uint8_t>> maybe_automatic = random_input(kGfx1250Selector, 24);
  ASSERT_TRUE(maybe_automatic.succeeded());
  const auto automatic = maybe_automatic.value();
  const util::FailureOr<std::vector<uint8_t>> maybe_pinned =
      random_input(kGfx1250Selector, 24, FuzzTarget::Gfx1250);
  ASSERT_TRUE(maybe_pinned.succeeded());
  const auto pinned = maybe_pinned.value();
  EXPECT_EQ(automatic, pinned);
  const util::FailureOr<RunStats> maybe_stats = run_input(automatic);
  ASSERT_TRUE(maybe_stats.succeeded());
  const util::FailureOr<RunStats> maybe_stats_2 = run_input(pinned, FuzzTarget::Gfx1250);
  ASSERT_TRUE(maybe_stats_2.succeeded());
  EXPECT_EQ(maybe_stats.value().executed, maybe_stats_2.value().executed);
}

TEST(ExecutionFuzzInputTest, PinnedGeneratorEncodesPinnedTarget) {
  const util::FailureOr<std::vector<uint8_t>> maybe_input = random_input(5, 24, FuzzTarget::Gfx90a);
  ASSERT_TRUE(maybe_input.succeeded());
  const auto input = maybe_input.value();
  ASSERT_GE(input.size(), kHeaderSize);
  EXPECT_EQ(input[0], static_cast<uint8_t>(FuzzTarget::Gfx90a));
  EXPECT_EQ(indexed_target(input[0]), FuzzTarget::Gfx90a);
}

TEST(ExecutionFuzzInputTest, ExplicitModesOverrideInputSelection) {
  const util::FailureOr<std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>>
      maybe_resources = make_async_execution_resources(2);
  ASSERT_TRUE(maybe_resources.succeeded());
  auto resources = maybe_resources.value();
  const util::FailureOr<std::vector<uint8_t>> maybe_async_input =
      random_input(17, 24, FuzzTarget::Gfx1250);
  ASSERT_TRUE(maybe_async_input.succeeded());
  const auto async_input = maybe_async_input.value();
  const util::FailureOr<RunStats> maybe_synchronous =
      run_input(async_input, FuzzTarget::Gfx1250, resources, AsyncMode::Disabled, 2);
  ASSERT_TRUE(maybe_synchronous.succeeded());
  const RunStats synchronous = maybe_synchronous.value();
  EXPECT_EQ(synchronous.async_enabled_inputs, 0u);
  EXPECT_EQ(synchronous.async_executed, 0u);
  EXPECT_EQ(synchronous.concurrent_inputs, 1u);
  EXPECT_EQ(synchronous.issuer_executions, 2u);

  const util::FailureOr<std::vector<uint8_t>> maybe_synchronous_input =
      random_input(1, 24, FuzzTarget::Gfx1250);
  ASSERT_TRUE(maybe_synchronous_input.succeeded());
  const auto synchronous_input = maybe_synchronous_input.value();
  const util::FailureOr<RunStats> maybe_asynchronous =
      run_input(synchronous_input, FuzzTarget::Gfx1250, resources, AsyncMode::Enabled, 1);
  ASSERT_TRUE(maybe_asynchronous.succeeded());
  const RunStats asynchronous = maybe_asynchronous.value();
  EXPECT_EQ(asynchronous.async_enabled_inputs, 1u);
  EXPECT_EQ(asynchronous.async_executed, 0u);
  EXPECT_EQ(asynchronous.concurrent_inputs, 0u);
  EXPECT_EQ(asynchronous.issuer_executions, 1u);
}

TEST(ExecutionFuzzInputTest, HarnessModesPreserveInitialScalarCondition) {
  constexpr std::array<uint8_t, 3> kReplayModes = {
      kWaitCheckingBit,
      kCacheMemoryBit,
      kWaitCheckingBit | kCacheMemoryBit,
  };
  const util::FailureOr<std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>>
      maybe_resources = make_async_execution_resources(1);
  ASSERT_TRUE(maybe_resources.succeeded());
  const auto resources = maybe_resources.value();
  bool saw_taken_branch = false;
  bool saw_fallthrough = false;
  for (uint8_t marker = 0; marker < 64; ++marker) {
    SCOPED_TRACE(static_cast<unsigned>(marker));
    std::vector<uint8_t> input{static_cast<uint8_t>(FuzzTarget::Gfx1250), 0};
    const auto append = [&](auto words) {
      for (const uint32_t word : words)
        for (unsigned byte = 0; byte < sizeof(word); ++byte)
          input.push_back(static_cast<uint8_t>(word >> (8 * byte)));
    };
    // The trailing marker changes the seeded SCC, which controls whether the NOP runs.
    append(cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 1}));
    append(cdna5::build_sopp(cdna5::kSNopSopp));
    append(cdna5::build_sopp(cdna5::kSEndpgmSopp));
    input.push_back(marker);
    const util::FailureOr<RunStats> maybe_synchronous =
        run_input(input, std::nullopt, resources, AsyncMode::Disabled, 1);
    ASSERT_TRUE(maybe_synchronous.succeeded());
    const RunStats synchronous = maybe_synchronous.value();
    const util::FailureOr<RunStats> maybe_asynchronous =
        run_input(input, std::nullopt, resources, AsyncMode::Enabled, 1);
    ASSERT_TRUE(maybe_asynchronous.succeeded());
    const RunStats asynchronous = maybe_asynchronous.value();
    EXPECT_EQ(synchronous.executed, asynchronous.executed);
    EXPECT_EQ(synchronous.inputs_halted, 1u);
    EXPECT_EQ(asynchronous.inputs_halted, 1u);
    saw_taken_branch |= synchronous.executed == 2;
    saw_fallthrough |= synchronous.executed == 3;
    for (uint8_t mode : kReplayModes) {
      SCOPED_TRACE(static_cast<unsigned>(mode));
      input[1] = mode;
      const util::FailureOr<RunStats> maybe_replay =
          run_input(input, std::nullopt, resources, AsyncMode::Disabled, 1);
      ASSERT_TRUE(maybe_replay.succeeded());
      const RunStats replay = maybe_replay.value();
      EXPECT_EQ(replay.executed, synchronous.executed);
      EXPECT_EQ(replay.inputs_halted, 1u);
    }
  }
  EXPECT_TRUE(saw_taken_branch);
  EXPECT_TRUE(saw_fallthrough);
}

TEST(ExecutionFuzzInputTest, ThreadedInputsShareAsyncWmmaHelpers) {
  constexpr uint64_t kThreads = 8;
  constexpr uint64_t kInputsPerThread = 2;
  const util::FailureOr<std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>>
      maybe_resources = make_async_execution_resources(4);
  ASSERT_TRUE(maybe_resources.succeeded());
  auto resources = maybe_resources.value();
  std::vector<RunStats> results(kThreads);
  std::vector<std::jthread> workers;
  workers.reserve(kThreads);
  for (uint64_t worker = 0; worker != kThreads; ++worker) {
    workers.emplace_back([&, worker] {
      for (uint64_t input = 0; input != kInputsPerThread; ++input) {
        const uint64_t seed = 1 + 16 * (worker + input * kThreads);
        const util::FailureOr<std::vector<uint8_t>> maybe_input =
            random_input(seed, 24, FuzzTarget::Gfx1250);
        ASSERT_TRUE(maybe_input.succeeded());
        const util::FailureOr<RunStats> maybe_stats =
            run_input(maybe_input.value(), FuzzTarget::Gfx1250, resources, AsyncMode::Enabled);
        ASSERT_TRUE(maybe_stats.succeeded());
        const RunStats stats = maybe_stats.value();
        EXPECT_EQ(stats.async_enabled_inputs, 1u);
        if (!(seed & 16))
          EXPECT_EQ(stats.async_executed, 0u);
        results[worker].executed += stats.executed;
        results[worker].async_executed += stats.async_executed;
        results[worker].inputs += stats.inputs;
        results[worker].inputs_executed += stats.inputs_executed;
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  uint64_t async_executed = 0;
  for (const RunStats &stats : results) {
    EXPECT_EQ(stats.inputs, kInputsPerThread);
    EXPECT_EQ(stats.inputs_executed, kInputsPerThread);
    async_executed += stats.async_executed;
  }
  EXPECT_GT(async_executed, 0u);
}

} // namespace
} // namespace rocjitsu::execution_fuzz
