// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"

#include "../dbi_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace rocjitsu {
namespace {
static_assert(
    std::is_same_v<decltype(std::declval<DropBarrierSelection &>().kind()), FaultMutationKind>);
static_assert(std::is_same_v<decltype(std::declval<DropBarrierSelection &>().owner()),
                             const FaultMutationOwner &>);
static_assert(std::is_same_v<decltype(std::declval<DropBarrierSelection &>().barrier()),
                             const StaticBarrier &>);
static_assert(!std::is_aggregate_v<DropBarrierSelection>);

constexpr rj_code_target_id_t kTarget = ROCJITSU_CODE_TARGET_GFX1250;
constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
constexpr uint32_t kSignal = 0xbe804ec1u;
constexpr uint32_t kWait = 0xbf94ffffu;
constexpr uint32_t kWaitDscnt = 0xbfc60000u;
constexpr uint32_t kMov = 0xbe800081u;
constexpr uint32_t kEnd = 0xbfb00000u;

std::vector<uint8_t> make_gfx1250_barrier_elf(std::vector<uint32_t> words = {kWaitDscnt, kSignal,
                                                                             kMov, kWait,
                                                                             kWaitDscnt, kEnd}) {
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250, 0, 0, false, false, false, false, true);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}
DropBarrierSelection select_barrier_for_test(const FaultBarrierInventory &inventory,
                                             size_t logical_ordinal = 0) {
  FaultMutationError error;
  FailureOr<DropBarrierSelection> selected =
      select_drop_barrier(inventory, {.exact = std::nullopt, .ordinal = logical_ordinal}, error);
  EXPECT_TRUE(selected.succeeded()) << error.message;
  return selected.succeeded() ? selected.value() : DropBarrierSelection{};
}

// Recipe oracle lives only in tests. It never calls apply or a production output checker.
Result matches_drop_output(const FaultMutationContext &context,
                           const DropBarrierSelection &selection,
                           std::span<const uint8_t> candidate, FaultMutationError &error) {
  if (!selection.belongs_to(context)) {
    error = {.code = FaultMutationErrorCode::BadIdentity,
             .offset = 0,
             .message = "test oracle: wrong source"};
    return Result::failure();
  }
  FailureOr<FaultBarrierInventory> pristine =
      discover_drop_barriers(context, selection.owner().entry, error);
  if (pristine.failed())
    return Result::failure();
  auto mismatch = [&] {
    error = {.code = FaultMutationErrorCode::ValidationFailure,
             .offset = 0,
             .message = "test oracle: incorrect complete Drop"};
    return Result::failure();
  };
  const std::span<const StaticBarrier> barriers = pristine.value().barriers();
  const std::span<const StaticBarrier>::iterator selected =
      std::ranges::find(barriers, selection.barrier());
  if (selected == barriers.end() || candidate.size() != context.image().size())
    return mismatch();
  std::vector<uint8_t> expected(context.image().begin(), context.image().end());
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), context.source().target_id());
  for (const StaticBarrierMember &member : selected->members) {
    const size_t begin = pristine.value().section().file_offset + member.offset;
    if (member.bytes.size() != 4 || begin + 4 > candidate.size())
      return mismatch();
    uint32_t word = 0;
    std::memcpy(&word, candidate.data() + begin, 4);
    DecodeResult instruction = decoder->decode_window(std::span(&word, 1), member.offset);
    if (instruction.failed() || instruction.value()->size() != 4 ||
        instruction.value()->mnemonic() != "s_nop" ||
        instruction.value()->num_src_operands() != 1 ||
        instruction.value()->src_operand(0)->encoding_value() != 0)
      return mismatch();
    std::memcpy(expected.data() + begin, &word, 4);
  }
  if (!std::ranges::equal(expected, candidate))
    return mismatch();
  const FaultMutationContext mutant(candidate);
  FailureOr<FaultBarrierInventory> remaining =
      discover_drop_barriers(mutant, selection.owner().entry, error);
  if (remaining.failed())
    return mismatch();
  std::vector<StaticBarrier> wanted(barriers.begin(), barriers.end());
  std::erase(wanted, *selected);
  if (!std::ranges::equal(remaining.value().barriers(), wanted))
    return mismatch();
  return Result::success();
}

std::vector<uint8_t> make_gfx1250_two_kernel_barrier_elf(
    const std::vector<uint32_t> &first = {kSignal, kMov, kWait, kEnd},
    const std::vector<uint32_t> &second = {kSignal, kMov, kWait, kEnd}) {
  constexpr uint32_t kSecondEntryByteOffset = 256;
  std::vector<uint32_t> words(kSecondEntryByteOffset / sizeof(uint32_t), 0);
  std::copy(first.begin(), first.end(), words.begin());
  words.insert(words.end(), second.begin(), second.end());
  std::vector<uint8_t> image = test::make_gfx950_two_kernel_elf(words, 0);
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  std::memcpy(image.data(), &header, sizeof(header));
  const AmdGpuCodeObject object(image.data(), image.size());
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const KernelDescriptorInfo &kernel : kernels) {
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    descriptor.kernel_code_properties |= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    if (kernel.kernel_name == "test_kernel1")
      descriptor.kernel_code_entry_byte_offset += kSecondEntryByteOffset;
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  return image;
}

struct DropTargetRow {
  const char *name;
  rj_code_target_id_t target;
  rj_code_arch_t arch;
  uint32_t wave_size;
  bool split;
};
constexpr DropTargetRow kRdna3DropRows[] = {
    {.name = "Gfx1100Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1100,
     .arch = ROCJITSU_CODE_ARCH_RDNA3,
     .wave_size = 32,
     .split = false},
    {.name = "Gfx1100Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1100,
     .arch = ROCJITSU_CODE_ARCH_RDNA3,
     .wave_size = 64,
     .split = false},
};
constexpr DropTargetRow kDropRows[] = {
    {.name = "Gfx1200Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1200,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1200Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1200,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 64,
     .split = true},
    {.name = "Gfx1201Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1201,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1201Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1201,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 64,
     .split = true},
    {.name = "Gfx1250Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1250,
     .arch = ROCJITSU_CODE_ARCH_CDNA5,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1251Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1251,
     .arch = ROCJITSU_CODE_ARCH_CDNA5,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx942Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX942,
     .arch = ROCJITSU_CODE_ARCH_CDNA3,
     .wave_size = 64,
     .split = false},
    {.name = "Gfx950Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX950,
     .arch = ROCJITSU_CODE_ARCH_CDNA4,
     .wave_size = 64,
     .split = false},
    kRdna3DropRows[0],
    kRdna3DropRows[1],
};

std::vector<uint32_t> make_barrier_words(const DropTargetRow &row) {
  // RDNA3 generated test_encodings.h: s_waitcnt and full s_barrier differ
  // from CDNA. Offset 4/16 barriers surround unchanged waits and s_mov_b32.
  if (row.arch == ROCJITSU_CODE_ARCH_RDNA3)
    return {0xbf890000u, 0xbfbd0000u, kMov, 0xbf890000u, 0xbfbd0000u, build_s_endpgm(row.arch)};
  const uint32_t wait = row.split ? kWaitDscnt : 0xbf8c0000u;
  if (row.split)
    return {wait, kSignal, kMov, kWait, wait, kSignal, kMov, kWait, build_s_endpgm(row.arch)};
  return {wait, 0xbf8a0000u, kMov, wait, 0xbf8a0000u, build_s_endpgm(row.arch)};
}

std::vector<uint8_t> make_barrier_elf_for_target(const DropTargetRow &row,
                                                 std::vector<uint32_t> words,
                                                 bool siblings = false) {
  constexpr uint32_t kSecondEntryByteOffset = 256;
  const IsaGpuTargetDescription *binding =
      default_isa_target_registry().find_gpu_target(row.target);
  if (!binding)
    return {};
  if (siblings) {
    const std::vector<uint32_t> body = words;
    words.resize(kSecondEntryByteOffset / sizeof(uint32_t), build_s_nop(0, row.arch));
    words.insert(words.end(), body.begin(), body.end());
  }
  std::vector<uint8_t> image =
      siblings ? test::make_gfx950_two_kernel_elf(words, 0)
               : test::make_amdgpu_kernel_elf(words, 0, 3, binding->elf_machine);
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  header.e_flags = binding->elf_machine;
  std::memcpy(image.data(), &header, sizeof(header));
  const AmdGpuCodeObject object(image.data(), image.size());
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const KernelDescriptorInfo &kernel : descriptors) {
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    if (row.wave_size == 32)
      descriptor.kernel_code_properties |= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    else
      descriptor.kernel_code_properties &= ~amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    if (siblings && kernel.kernel_name == "test_kernel1")
      descriptor.kernel_code_entry_byte_offset += kSecondEntryByteOffset;
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  return image;
}

class FaultDropBarrierFamily : public ::testing::TestWithParam<DropTargetRow> {};

TEST_P(FaultDropBarrierFamily, SelectorUsesLogicalOrderExactPrecedenceAndImmutableRecords) {
  const DropTargetRow &row = GetParam();
  const FaultMutationContext context(make_barrier_elf_for_target(row, make_barrier_words(row)));
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<FaultBarrierInventory> again =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(again.succeeded());
  EXPECT_EQ(inventory.value().barriers().data(), again.value().barriers().data());
  ASSERT_EQ(inventory.value().barriers().size(), 2u);
  FailureOr<DropBarrierSelection> first = select_drop_barrier(inventory.value(), {}, error);
  FailureOr<DropBarrierSelection> second =
      select_drop_barrier(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  EXPECT_LT(first.value().barrier().members.front().offset,
            second.value().barrier().members.front().offset);
  EXPECT_EQ(first.value().barrier().members.size(), row.split ? 2u : 1u);
  const FailureOr<DropBarrierSelection> exact = select_drop_barrier(
      again.value(), {.exact = second.value(), .ordinal = std::numeric_limits<size_t>::max()},
      error);
  ASSERT_TRUE(exact.succeeded()) << error.message;
  EXPECT_EQ(exact.value().barrier(), second.value().barrier());
  const FailureOr<DropBarrierResult> emitted = apply_drop_barrier(context, exact.value(), error);
  ASSERT_TRUE(emitted.succeeded()) << error.message;
  const FailureOr<DropBarrierResult> repeated = apply_drop_barrier(context, exact.value(), error);
  ASSERT_TRUE(repeated.succeeded());
  EXPECT_EQ(emitted.value().image, repeated.value().image);
  EXPECT_TRUE(
      matches_drop_output(context, exact.value(), repeated.value().image, error).succeeded());
}

TEST(FaultDropBarrier, SelectorRejectsEmptyOutOfRangeAndUnboundExact) {
  FaultMutationError error;
  const FaultMutationContext empty(make_gfx1250_barrier_elf({kMov, kEnd}));
  const FailureOr<FaultBarrierInventory> no_barriers =
      discover_drop_barriers(empty, "test_kernel", error);
  ASSERT_TRUE(no_barriers.succeeded());
  EXPECT_TRUE(select_drop_barrier(no_barriers.value(), {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
  const FaultMutationContext context(make_gfx1250_barrier_elf());
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded());
  EXPECT_TRUE(select_drop_barrier(
                  inventory.value(),
                  {.exact = std::nullopt, .ordinal = std::numeric_limits<size_t>::max()}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
  EXPECT_TRUE(
      select_drop_barrier(inventory.value(), {.exact = DropBarrierSelection{}, .ordinal = 0}, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultDropBarrier, ExactSelectorRejectsOtherKernelAndEqualBytesInAnotherContext) {
  const std::vector<uint8_t> image = make_gfx1250_two_kernel_barrier_elf();
  const FaultMutationContext context(image), other(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> first =
      discover_drop_barriers(context, "test_kernel0", error);
  const FailureOr<FaultBarrierInventory> second =
      discover_drop_barriers(context, "test_kernel1", error);
  const FailureOr<FaultBarrierInventory> foreign =
      discover_drop_barriers(other, "test_kernel0", error);
  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  ASSERT_TRUE(foreign.succeeded());
  const FailureOr<DropBarrierSelection> selected = select_drop_barrier(first.value(), {}, error);
  ASSERT_TRUE(selected.succeeded());
  for (const FaultBarrierInventory *inventory : {&second.value(), &foreign.value()}) {
    EXPECT_TRUE(
        select_drop_barrier(*inventory, {.exact = selected.value(), .ordinal = 0}, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  }
}

TEST(FaultDropBarrier, CachedQualificationFailureDoesNotPoisonAnotherKernel) {
  const FaultMutationContext context(make_gfx1250_two_kernel_barrier_elf({kSignal, kEnd}));
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::IncompleteSelection);
  const FailureOr<FaultBarrierInventory> good =
      discover_drop_barriers(context, "test_kernel1", error);
  ASSERT_TRUE(good.succeeded()) << error.message;
  const FailureOr<DropBarrierSelection> selected = select_drop_barrier(good.value(), {}, error);
  ASSERT_TRUE(selected.succeeded());
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selected.value(), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(
      matches_drop_output(context, selected.value(), result.value().image, error).succeeded());
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::IncompleteSelection);
}

TEST_P(FaultDropBarrierFamily, CompleteLogicalBarrierPreservesTargetWaveAndOtherBarrier) {
  const DropTargetRow &row = GetParam();
  const std::vector<uint8_t> image = make_barrier_elf_for_target(row, make_barrier_words(row));
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 2u);
  const DropBarrierSelection selection =
      select_barrier_for_test(inventory.value(), inventory.value().barriers().size() - 1);
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().logical_mutation_count, 1u);
  EXPECT_EQ(result.value().selection.barrier().members.size(), row.split ? 2u : 1u);
  EXPECT_EQ(AmdGpuCodeObject(result.value().image.data(), result.value().image.size()).target_id(),
            row.target);
  EXPECT_FALSE(std::ranges::any_of(result.value().assumptions, [](const std::string &assumption) {
    return assumption.find("clustered dispatch") != std::string::npos;
  }));
  std::vector<uint8_t> expected = image;
  const uint32_t nop = build_s_nop(0, row.arch);
  for (const StaticBarrierMember &member : selection.barrier().members)
    std::memcpy(expected.data() + inventory.value().section().file_offset + member.offset, &nop,
                sizeof(nop));
  EXPECT_EQ(result.value().image, expected);
  EXPECT_TRUE(matches_drop_output(context, selection, result.value().image, error).succeeded())
      << error.message;

  // Drop is a recipe consumer of the same production replacement API. The
  // independent full-image oracle above also guards against shared wrong output.
  Instrumentor instrumentor(context.source(), row.arch);
  for (const StaticBarrierMember &member : selection.barrier().members) {
    InstructionReplacement replacement = {.expected_bytes = member.bytes,
                                          .replacement_bytes = std::vector<uint8_t>(sizeof(nop))};
    std::memcpy(replacement.replacement_bytes.data(), &nop, sizeof(nop));
    instrumentor.add_point({.anchor_offset = member.offset,
                            .kind = InstrumentationKind::ReplaceWithInst,
                            .probe_symbol = {},
                            .probe_args = {},
                            .replacement = std::move(replacement)});
  }
  const InstrumentedCodeObjectDebug direct = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(direct.errors.empty()) << direct.errors.front();
  EXPECT_FALSE(direct.edit_error);
  EXPECT_TRUE(direct.patches.empty());
  EXPECT_EQ(direct.elf_bytes, result.value().image);
  ASSERT_EQ(direct.sites.size(), selection.barrier().members.size());
  for (size_t index = 0; index < direct.sites.size(); ++index) {
    const StaticBarrierMember &member = selection.barrier().members[index];
    const InstrumentedSite &site = direct.sites[index];
    EXPECT_EQ(site.kind, InstrumentationKind::ReplaceWithInst);
    EXPECT_EQ(site.original_offset, member.offset);
    EXPECT_EQ(site.original_size, member.bytes.size());
    EXPECT_EQ(site.instrumentation_offset, member.offset);
    EXPECT_EQ(site.instrumentation_size, member.bytes.size());
    EXPECT_EQ(site.guest_size, 0u);
  }
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
  const FaultMutationContext mutant(result.value().image);
  const Section &text = *mutant.source().text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
      mutant.image(), text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernel_wavefront_size(row.arch, kernels.front().descriptor), row.wave_size);
}

TEST_P(FaultDropBarrierFamily, ExactSiblingSelectionAndCandidateCorruption) {
  const DropTargetRow &row = GetParam();
  const std::vector<uint8_t> image =
      make_barrier_elf_for_target(row, make_barrier_words(row), true);
  const FaultMutationContext context(image);
  FaultMutationError error;
  for (const std::string entry : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(entry);
    const FailureOr<FaultBarrierInventory> inventory =
        discover_drop_barriers(context, entry, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<FaultBarrierInventory> sibling = discover_drop_barriers(
        context, entry == "test_kernel0" ? "test_kernel1" : "test_kernel0", error);
    ASSERT_TRUE(sibling.succeeded()) << error.message;
    const DropBarrierSelection selection =
        select_barrier_for_test(inventory.value(), inventory.value().barriers().size() - 1);
    const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    std::vector<uint8_t> expected = image;
    const uint32_t nop = build_s_nop(0, row.arch);
    for (const StaticBarrierMember &member : selection.barrier().members)
      std::memcpy(expected.data() + inventory.value().section().file_offset + member.offset, &nop,
                  sizeof(nop));
    EXPECT_EQ(result.value().image, expected);
    const FaultMutationContext mutant(expected);
    const FailureOr<FaultBarrierInventory> fresh_sibling =
        discover_drop_barriers(mutant, sibling.value().owner().entry, error);
    ASSERT_TRUE(fresh_sibling.succeeded()) << error.message;
    EXPECT_TRUE(std::ranges::equal(fresh_sibling.value().barriers(), sibling.value().barriers()));
    const uint64_t text = inventory.value().section().file_offset;
    for (const uint64_t offset :
         {text + inventory.value().barriers().front().members.front().offset,
          text + inventory.value().owner().entry_offset,
          text + selection.barrier().members.front().offset,
          sibling.value().owner().descriptor_file_offset,
          uint64_t{offsetof(Elf64_Ehdr, e_flags)}}) {
      std::vector<uint8_t> corrupted = expected;
      corrupted[offset] ^= 1;
      EXPECT_TRUE(matches_drop_output(context, selection, corrupted, error).failed()) << offset;
      EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
    }
  }
}

TEST_P(FaultDropBarrierFamily, PreservesAdditionalExecutableSection) {
  const DropTargetRow &row = GetParam();
  std::vector<uint8_t> image = make_barrier_elf_for_target(row, make_barrier_words(row));
  const AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  const std::vector<Elf64_Shdr> sections = CodeObjectPatcher(original).section_headers();
  const std::vector<Elf64_Shdr>::const_iterator data =
      std::ranges::find_if(sections, [](const Elf64_Shdr &candidate) {
        return candidate.sh_type == SHT_PROGBITS && (candidate.sh_flags & SHF_ALLOC) != 0 &&
               (candidate.sh_flags & SHF_EXECINSTR) == 0;
      });
  ASSERT_NE(data, sections.end());
  const size_t data_index = static_cast<size_t>(data - sections.begin());
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  Elf64_Shdr section = *data;
  const size_t offset = header.e_shoff + data_index * header.e_shentsize;
  section.sh_flags |= SHF_EXECINSTR;
  std::memcpy(image.data() + offset, &section, sizeof(section));
  const FaultMutationContext context(image);
  ASSERT_EQ(context.source().text_sections().size(), 1u);
  ASSERT_EQ(context.source().allocated_executable_sections().size(), 2u);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  std::vector<uint8_t> expected = image;
  const uint32_t nop = build_s_nop(0, row.arch);
  for (const StaticBarrierMember &member : selection.barrier().members)
    std::memcpy(expected.data() + inventory.value().section().file_offset + member.offset, &nop, 4);
  EXPECT_EQ(result.value().image, expected);
  EXPECT_TRUE(matches_drop_output(context, selection, expected, error).succeeded());
  expected[section.sh_offset] ^= 1;
  EXPECT_TRUE(matches_drop_output(context, selection, expected, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
}

TEST_P(FaultDropBarrierFamily, RejectsDefaultAndForeignSelection) {
  const DropTargetRow &row = GetParam();
  const std::vector<uint8_t> image = make_barrier_elf_for_target(row, make_barrier_words(row));
  const FaultMutationContext context(image);
  const FaultMutationContext other(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selected = select_barrier_for_test(inventory.value());
  EXPECT_TRUE(apply_drop_barrier(other, selected, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_drop_barrier(context, DropBarrierSelection{}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST_P(FaultDropBarrierFamily, StaticBarrierInLoopAndFamilySpecificForms) {
  const DropTargetRow &row = GetParam();
  std::vector<uint32_t> words = make_barrier_words(row);
  words.back() = build_s_branch(-static_cast<int32_t>(words.size()), row.arch);
  const FaultMutationContext context(make_barrier_elf_for_target(row, words));
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  EXPECT_TRUE(
      apply_drop_barrier(context, select_barrier_for_test(inventory.value()), error).succeeded())
      << error.message;
  if (row.split) {
    for (const uint32_t selector : {193u, 194u, 195u, 196u}) {
      const int32_t id = -static_cast<int32_t>(selector - 192);
      const FaultMutationContext builtin(make_barrier_elf_for_target(
          row, {0xbe804e00u | selector, 0xbf940000u | static_cast<uint16_t>(id),
                build_s_endpgm(row.arch)}));
      const FailureOr<FaultBarrierInventory> discovered =
          discover_drop_barriers(builtin, "test_kernel", error);
      if (id == -2 || id == -4 || (row.arch == ROCJITSU_CODE_ARCH_RDNA4 && id == -3)) {
        EXPECT_TRUE(discovered.failed());
        EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
      } else {
        ASSERT_TRUE(discovered.succeeded()) << error.message;
        ASSERT_EQ(discovered.value().barriers().size(), 1u);
        EXPECT_EQ(discovered.value().barriers().front().scope,
                  id == -1 ? StaticBarrierScope::Workgroup : StaticBarrierScope::Cluster);
        EXPECT_TRUE(apply_drop_barrier(builtin, select_barrier_for_test(discovered.value()), error)
                        .succeeded())
            << error.message;
      }
    }
    for (const std::vector<uint32_t> &incomplete : std::vector<std::vector<uint32_t>>{
             {kSignal, build_s_endpgm(row.arch)},
             {kWait, build_s_endpgm(row.arch)},
             {0xbe804fc1u, kWait, build_s_endpgm(row.arch)},
             {0xbe804e7du, kWait, build_s_endpgm(row.arch)},
             {0xbe804effu, 0xffffffffu, kWait, build_s_endpgm(row.arch)},
             {0xbe804e81u, 0xbf940001u, build_s_endpgm(row.arch)}}) {
      const FaultMutationContext unsupported(make_barrier_elf_for_target(row, incomplete));
      EXPECT_TRUE(discover_drop_barriers(unsupported, "test_kernel", error).failed());
    }
  } else if (row.arch == ROCJITSU_CODE_ARCH_CDNA3 || row.arch == ROCJITSU_CODE_ARCH_CDNA4) {
    // CDNA's wave helper normalizes to 64; its raw Wave32 bit is still invalid.
    // RDNA3 singleton barriers admit both descriptor-selected wave modes.
    DropTargetRow wrong_wave = row;
    wrong_wave.wave_size = 32;
    const FaultMutationContext unsupported(
        make_barrier_elf_for_target(wrong_wave, make_barrier_words(row)));
    EXPECT_TRUE(discover_drop_barriers(unsupported, "test_kernel", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  }
}

TEST(FaultDropBarrier, RejectsUndecodableUnselectedRdnaPadding) {
  for (const DropTargetRow &row : kDropRows) {
    if (row.arch != ROCJITSU_CODE_ARCH_RDNA4)
      continue;
    std::vector<uint8_t> image = make_barrier_elf_for_target(row, make_barrier_words(row), true);
    const AmdGpuCodeObject source(image.data(), image.size());
    const uint32_t invalid = 0;
    std::memcpy(image.data() + source.text_sections().front()->sectionOffset() + 128, &invalid, 4);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<FaultBarrierInventory> inventory =
        discover_drop_barriers(context, "test_kernel0", error);
    // Full-text input facts are built once, before exposing a selectable inventory.
    EXPECT_TRUE(inventory.failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    EXPECT_FALSE(error.message.empty());
  }
}

TEST(FaultDropBarrier, ConcreteGfx1251DecoderSurvivesSourceAndEmitterChecks) {
  const DropTargetRow &variant = kDropRows[5];
  std::vector<uint32_t> words = make_barrier_words(variant);
  // Already used in InstrumentorProbePatch: gfx1251-only VOP3P, legal only with its features.
  words.insert(words.begin(), {0xcc4c4004u, 0x1a021908u});
  const std::vector<uint8_t> image = make_barrier_elf_for_target(variant, words);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(AmdGpuCodeObject(result.value().image.data(), result.value().image.size()).target_id(),
            ROCJITSU_CODE_TARGET_GFX1251);
  EXPECT_EQ(std::memcmp(result.value().image.data() + inventory.value().section().file_offset,
                        image.data() + inventory.value().section().file_offset, 8),
            0);
  EXPECT_TRUE(matches_drop_output(context, selection, result.value().image, error).succeeded());
  const FaultMutationContext wrong_variant(make_barrier_elf_for_target(kDropRows[4], words));
  EXPECT_TRUE(discover_drop_barriers(wrong_variant, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
}

TEST(FaultDropBarrier, RejectsFamiliesWithoutCompleteDropPolicyAndUnregisteredTargets) {
  for (const uint32_t machine :
       {EF_AMDGPU_MACH_AMDGCN_GFX908, EF_AMDGPU_MACH_AMDGCN_GFX90A, EF_AMDGPU_MACH_AMDGCN_GFX940,
        EF_AMDGPU_MACH_AMDGCN_GFX941, EF_AMDGPU_MACH_AMDGCN_GFX1010, EF_AMDGPU_MACH_AMDGCN_GFX1030,
        EF_AMDGPU_MACH_AMDGCN_GFX1150, uint32_t{0xfe}}) {
    std::vector<uint8_t> image = make_gfx1250_barrier_elf();
    Elf64_Ehdr header = {};
    std::memcpy(&header, image.data(), sizeof(header));
    header.e_flags = machine;
    std::memcpy(image.data(), &header, sizeof(header));
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedTarget);
  }
}

INSTANTIATE_TEST_SUITE_P(TargetWaves, FaultDropBarrierFamily, ::testing::ValuesIn(kDropRows),
                         [](const ::testing::TestParamInfo<DropTargetRow> &info) {
                           return info.param.name;
                         });

class FaultDropBarrierRdna3 : public ::testing::TestWithParam<DropTargetRow> {};

TEST_P(FaultDropBarrierRdna3, GoldenEncodingsAndExactDeltaPreserveDescriptor) {
  const DropTargetRow &row = GetParam();
  const std::vector<uint32_t> words = make_barrier_words(row);
  const std::vector<uint8_t> image = make_barrier_elf_for_target(row, words);
  const FaultMutationContext context(image);
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), row.target);
  ASSERT_TRUE(decoder);
  // Golden bytes are independent of the production NOP emitter. RDNA3's
  // s_barrier is SOPP 61; CDNA's SOPP 10 is s_wait_idle on this target.
  constexpr uint32_t nop = 0xbf800000u;
  EXPECT_EQ(build_s_nop(0, row.arch), nop);
  constexpr std::array<std::string_view, 6> mnemonics = {"s_waitcnt", "s_barrier", "s_mov_b32",
                                                         "s_waitcnt", "s_barrier", "s_endpgm"};
  ASSERT_EQ(words.size(), mnemonics.size());
  for (size_t index = 0; index < words.size(); ++index) {
    DecodeResult instruction = decoder->decode_window(std::span(words).subspan(index), index * 4);
    ASSERT_TRUE(instruction.succeeded()) << index;
    EXPECT_EQ(instruction.value()->size(), 4u);
    EXPECT_EQ(instruction.value()->mnemonic(), mnemonics[index]);
  }
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 2u);
  for (size_t index = 0; index < 2; ++index) {
    const StaticBarrier &barrier = inventory.value().barriers()[index];
    EXPECT_EQ(barrier.form, StaticBarrierForm::FullSingleton);
    EXPECT_EQ(barrier.scope, StaticBarrierScope::Workgroup);
    EXPECT_EQ(barrier.barrier_id, -1);
    ASSERT_EQ(barrier.members.size(), 1u);
    EXPECT_EQ(barrier.members.front().offset, index == 0 ? 4u : 16u);
    EXPECT_EQ(barrier.members.front().bytes, (std::vector<uint8_t>{0, 0, 0xbd, 0xbf}));
  }
  const FailureOr<DropBarrierSelection> second =
      select_drop_barrier(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(second.succeeded());
  const FailureOr<DropBarrierSelection> exact =
      select_drop_barrier(inventory.value(), {.exact = second.value(), .ordinal = 0}, error);
  ASSERT_TRUE(exact.succeeded());
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, exact.value(), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().logical_mutation_count, 1u);
  ASSERT_EQ(result.value().selection.barrier().members.size(), 1u);
  EXPECT_EQ(result.value().selection.barrier().members.front().offset, 16u);
  const size_t replacement_offset = inventory.value().section().file_offset + 16;
  ASSERT_GE(result.value().image.size(), replacement_offset + 4);
  EXPECT_EQ((std::vector<uint8_t>(result.value().image.begin() + replacement_offset,
                                  result.value().image.begin() + replacement_offset + 4)),
            (std::vector<uint8_t>{0, 0, 0x80, 0xbf}));
  std::vector<uint8_t> expected = image;
  const uint64_t selected = inventory.value().section().file_offset + 16;
  std::memcpy(expected.data() + selected, &nop, sizeof(nop));
  EXPECT_EQ(result.value().image, expected);
  EXPECT_EQ(AmdGpuCodeObject(result.value().image.data(), result.value().image.size()).target_id(),
            ROCJITSU_CODE_TARGET_GFX1100);
  const FaultMutationContext mutant(result.value().image);
  const Section &text = *mutant.source().text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors = scan_kernel_descriptors(
      mutant.image(), text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(descriptors.size(), 1u);
  EXPECT_EQ(kernel_wavefront_size(row.arch, descriptors.front().descriptor), row.wave_size);
  EXPECT_EQ((descriptors.front().descriptor.kernel_code_properties &
             amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0,
            row.wave_size == 32);
  EXPECT_EQ(descriptors.front().entry_text_offset, 0u);
  EXPECT_EQ(descriptors.front().descriptor_file_offset,
            inventory.value().owner().descriptor_file_offset);
  EXPECT_EQ(std::memcmp(mutant.image().data() + descriptors.front().descriptor_file_offset,
                        image.data() + inventory.value().owner().descriptor_file_offset,
                        sizeof(amdhsa::kernel_descriptor_t)),
            0);
  EXPECT_TRUE(matches_drop_output(context, exact.value(), mutant.image(), error).succeeded());
  expected[selected] = 1; // A decoded s_nop with a different cycle count is not this recipe.
  EXPECT_TRUE(matches_drop_output(context, exact.value(), expected, error).failed());
}

TEST_P(FaultDropBarrierRdna3, CdnaBarrierEncodingIsAnOrdinaryWait) {
  const DropTargetRow &row = GetParam();
  const std::array<uint32_t, 1> words = {0xbf8a0000u};
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), row.target);
  ASSERT_TRUE(decoder);
  DecodeResult instruction = decoder->decode_window(words, 0);
  ASSERT_TRUE(instruction.succeeded());
  EXPECT_EQ(instruction.value()->mnemonic(), "s_wait_idle");
  EXPECT_FALSE(instruction.value()->is_barrier());
  const FaultMutationContext context(
      make_barrier_elf_for_target(row, {words.front(), build_s_endpgm(row.arch)}));
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  EXPECT_TRUE(inventory.value().barriers().empty());
  EXPECT_TRUE(select_drop_barrier(inventory.value(), {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
}

TEST_P(FaultDropBarrierRdna3, RejectsOtherBarrierFormsAndClauseMembers) {
  const DropTargetRow &row = GetParam();
  // RDNA3 ds_gws_barrier is not a full workgroup s_barrier. A clause member
  // and foreign split encodings also cannot enter this singleton inventory.
  for (const std::vector<uint32_t> &words :
       std::vector<std::vector<uint32_t>>{{0xd8740000u, 0, 0xbfbd0000u, build_s_endpgm(row.arch)},
                                          {0xbf850000u, 0xbfbd0000u, build_s_endpgm(row.arch)},
                                          {kWaitDscnt, kSignal, kMov, kWait, kWaitDscnt, kEnd}}) {
    const FaultMutationContext context(make_barrier_elf_for_target(row, words));
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
  }
}

TEST_P(FaultDropBarrierRdna3, SelectionCannotReplayChangedWaveEntryOrTarget) {
  const DropTargetRow &row = GetParam();
  const std::vector<uint8_t> image = make_barrier_elf_for_target(row, make_barrier_words(row));
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value(), 1);
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    std::vector<uint8_t> changed = image;
    amdhsa::kernel_descriptor_t descriptor = {};
    const uint64_t offset = inventory.value().owner().descriptor_file_offset;
    std::memcpy(&descriptor, changed.data() + offset, sizeof(descriptor));
    if (mode == 0)
      descriptor.kernel_code_properties ^= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    else if (mode == 1)
      descriptor.kernel_code_entry_byte_offset += 4;
    else {
      Elf64_Ehdr header = {};
      std::memcpy(&header, changed.data(), sizeof(header));
      header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1200;
      std::memcpy(changed.data(), &header, sizeof(header));
    }
    std::memcpy(changed.data() + offset, &descriptor, sizeof(descriptor));
    const FaultMutationContext other(changed);
    EXPECT_TRUE(apply_drop_barrier(other, selection, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
    if (mode < 2) {
      const FailureOr<FaultBarrierInventory> fresh =
          discover_drop_barriers(other, "test_kernel", error);
      ASSERT_TRUE(fresh.succeeded()) << error.message;
      EXPECT_TRUE(
          select_drop_barrier(fresh.value(), {.exact = selection, .ordinal = 0}, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
    }
  }
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded());
  EXPECT_TRUE(matches_drop_output(context, selection, result.value().image, error).succeeded());
}

INSTANTIATE_TEST_SUITE_P(Rdna3Waves, FaultDropBarrierRdna3, ::testing::ValuesIn(kRdna3DropRows),
                         [](const ::testing::TestParamInfo<DropTargetRow> &info) {
                           return info.param.name;
                         });

TEST(FaultDropBarrier, ExactKernelSelectionPreservesIndependentSibling) {
  const std::vector<uint8_t> image = make_gfx1250_two_kernel_barrier_elf();
  for (const std::string entry : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(entry);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<FaultBarrierInventory> inventory =
        discover_drop_barriers(context, entry, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().barriers().size(), 1u);
    const std::string sibling_entry = entry == "test_kernel0" ? "test_kernel1" : "test_kernel0";
    const FailureOr<FaultBarrierInventory> sibling =
        discover_drop_barriers(context, sibling_entry, error);
    ASSERT_TRUE(sibling.succeeded()) << error.message;
    const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
    const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    std::vector<uint8_t> expected = image;
    const uint32_t nop = build_s_nop(0, kArch);
    for (const StaticBarrierMember &member : selection.barrier().members)
      std::memcpy(expected.data() + inventory.value().section().file_offset + member.offset, &nop,
                  sizeof(nop));
    EXPECT_EQ(result.value().image, expected);
    const FaultMutationContext mutant(expected);
    const FailureOr<FaultBarrierInventory> fresh = discover_drop_barriers(mutant, entry, error);
    const FailureOr<FaultBarrierInventory> fresh_sibling =
        discover_drop_barriers(mutant, sibling_entry, error);
    ASSERT_TRUE(fresh.succeeded()) << error.message;
    ASSERT_TRUE(fresh_sibling.succeeded()) << error.message;
    EXPECT_TRUE(fresh.value().barriers().empty());
    EXPECT_EQ(fresh_sibling.value().owner(), sibling.value().owner());
    EXPECT_TRUE(std::ranges::equal(fresh_sibling.value().barriers(), sibling.value().barriers()));
    for (const uint64_t offset :
         {sibling.value().owner().descriptor_file_offset,
          sibling.value().section().file_offset + sibling.value().owner().entry_offset}) {
      std::vector<uint8_t> corrupted = expected;
      corrupted[offset] ^= 1;
      EXPECT_TRUE(matches_drop_output(context, selection, corrupted, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
    }
  }
}

TEST(FaultDropBarrier, UnselectedKernelBarrierFormDoesNotChangeExactSelection) {
  const std::vector<uint8_t> image = make_gfx1250_two_kernel_barrier_elf(
      {kSignal, kMov, kWait, kEnd}, {0xbe804e81u, 0xbf940001u, kEnd});
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  EXPECT_TRUE(
      apply_drop_barrier(context, select_barrier_for_test(inventory.value()), error).succeeded())
      << error.message;
}

TEST(FaultDropBarrier, SelectionRejectsAnotherContextEvenWithIdenticalBytes) {
  const std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  const FaultMutationContext context(image);
  const FaultMutationContext other(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(apply_drop_barrier(other, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(matches_drop_output(other, selection, result.value().image, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultDropBarrier, ExpiredSelectionAndInventoryCannotBindANewContext) {
  const std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  std::optional<FaultMutationContext> context;
  context.emplace(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(*context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(*context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  context.reset();
  context.emplace(image); // Reuse the context's address, not its source lifetime.
  EXPECT_TRUE(apply_drop_barrier(*context, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(select_drop_barrier(inventory.value(), {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(matches_drop_output(*context, selection, result.value().image, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultDropBarrier, SourceSnapshotSurvivesCallerEditsButSelectionCannotReplayThem) {
  std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  const std::vector<uint8_t> original = image;
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  const uint32_t changed_mov = 0xbe800082u;
  const uint64_t mov_file_offset = inventory.value().section().file_offset + 8;
  std::memcpy(image.data() + mov_file_offset, &changed_mov, sizeof(changed_mov));
  EXPECT_TRUE(std::ranges::equal(context.image(), original));
  const FaultMutationContext changed(image);
  EXPECT_TRUE(apply_drop_barrier(changed, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(std::memcmp(result.value().image.data() + mov_file_offset,
                        original.data() + mov_file_offset, sizeof(changed_mov)),
            0);
}

TEST(FaultDropBarrier, ExactSelectionRejectsDuplicateNames) {
  std::vector<uint8_t> image = make_gfx1250_two_kernel_barrier_elf();
  const FaultMutationContext context(image);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  const CodeObjectPatcher patcher(context.source());
  for (const Elf64_Shdr &section : patcher.section_headers()) {
    if (section.sh_type != SHT_SYMTAB)
      continue;
    Elf64_Sym first_symbol = {}, second_symbol = {};
    std::memcpy(&first_symbol, image.data() + section.sh_offset + sizeof(Elf64_Sym),
                sizeof(first_symbol));
    std::memcpy(&second_symbol, image.data() + section.sh_offset + 2 * sizeof(Elf64_Sym),
                sizeof(second_symbol));
    second_symbol.st_name = first_symbol.st_name;
    std::memcpy(image.data() + section.sh_offset + 2 * sizeof(Elf64_Sym), &second_symbol,
                sizeof(second_symbol));
  }
  const FaultMutationContext ambiguous(image);
  EXPECT_TRUE(discover_drop_barriers(ambiguous, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::AmbiguousSelection);
}

TEST(FaultDropBarrier, RejectsCrossKernelEntryAndInteriorBranches) {
  for (const uint64_t destination : {256u, 260u}) {
    SCOPED_TRACE(destination);
    const std::vector<uint8_t> image = make_gfx1250_two_kernel_barrier_elf(
        {kSignal, kWait, build_s_branch(static_cast<int32_t>((destination - 12) / 4), kArch)});
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel0", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
  }
}

TEST(FaultDropBarrier, RejectsSharedReachableBlocksAndUnresolvedSibling) {
  std::vector<uint32_t> first(35, 0);
  first[0] = build_s_branch(31, kArch); // Both entries reach the pair at .text+128.
  first[32] = kSignal;
  first[33] = kWait;
  first[34] = kEnd;
  const FaultMutationContext shared(
      make_gfx1250_two_kernel_barrier_elf(first, {build_s_branch(-33, kArch)}));
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(shared, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
  const FaultMutationContext unresolved(
      make_gfx1250_two_kernel_barrier_elf({kSignal, kWait, kEnd}, {0xbe802000u}));
  EXPECT_TRUE(discover_drop_barriers(unresolved, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
}

TEST(FaultDropBarrier, IndependentSiblingMalformedDescriptorCannotDisappearFromOwners) {
  std::vector<uint8_t> image = make_gfx1250_two_kernel_barrier_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  const CodeObjectPatcher patcher(object);
  for (const Elf64_Shdr &section : patcher.section_headers()) {
    if (section.sh_type != SHT_SYMTAB)
      continue;
    Elf64_Sym symbol = {};
    const uint64_t offset = section.sh_offset + 2 * sizeof(symbol);
    std::memcpy(&symbol, image.data() + offset, sizeof(symbol));
    symbol.st_size = 0;
    std::memcpy(image.data() + offset, &symbol, sizeof(symbol));
  }
  const FaultMutationContext context(image);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
}

TEST(FaultDropBarrier, PreservesUnselectedSectionButRejectsActualTextRelocations) {
  const std::vector<uint8_t> original = make_gfx1250_two_kernel_barrier_elf();
  const AmdGpuCodeObject source(original.data(), original.size());
  CodeObjectPatcher patcher(source);
  Elf64_Rela relocation = {};
  relocation.r_offset = source.text_sections().front()->vaddr();
  relocation.r_info = R_AMDGPU_ABS64;
  const auto payload =
      std::span(reinterpret_cast<const uint8_t *>(&relocation), sizeof(relocation));
  ASSERT_TRUE(patcher.append_nonalloc_section(".fi01.extra", payload));
  std::vector<uint8_t> image = patcher.emit();
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  std::vector<uint8_t> expected = image;
  const uint32_t nop = build_s_nop(0, kArch);
  for (const StaticBarrierMember &member : selection.barrier().members)
    std::memcpy(expected.data() + inventory.value().section().file_offset + member.offset, &nop, 4);
  EXPECT_EQ(result.value().image, expected);
  const CodeObjectPatcher updated(context.source());
  std::vector<Elf64_Shdr> sections = updated.section_headers();
  Elf64_Shdr &extra = sections.back();
  extra.sh_type = SHT_RELA;
  extra.sh_info = inventory.value().section().index;
  extra.sh_entsize = sizeof(Elf64_Rela);
  for (size_t index = 0; index < sections.size(); ++index)
    if (sections[index].sh_type == SHT_SYMTAB)
      extra.sh_link = index;
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  std::memcpy(image.data() + header.e_shoff + (sections.size() - 1) * sizeof(extra), &extra,
              sizeof(extra));
  const FaultMutationContext referenced(image);
  EXPECT_TRUE(discover_drop_barriers(referenced, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
}

TEST(FaultDropBarrier, RejectsDeclaredExecutableOwnerOutsideAnalyzedText) {
  const std::vector<uint8_t> original = make_gfx1250_barrier_elf();
  const AmdGpuCodeObject source(original.data(), original.size());
  CodeObjectPatcher patcher(source);
  const size_t code_index = patcher.section_headers().size();
  const std::array<uint32_t, 2> words = {build_s_nop(0, kArch), build_s_endpgm(kArch)};
  const std::array<uint8_t, 7> names = {0, 'e', 'x', 't', 'r', 'a', 0};
  std::array<Elf64_Sym, 2> symbols = {};
  symbols[1].st_name = 1;
  symbols[1].st_info = (kElfSymbolBindGlobal << 4) | kElfSymbolTypeFunc;
  symbols[1].st_shndx = static_cast<uint16_t>(code_index);
  symbols[1].st_value = 0x6000;
  symbols[1].st_size = sizeof(words);
  ASSERT_TRUE(patcher.append_nonalloc_section(
      ".fi01.code", {reinterpret_cast<const uint8_t *>(words.data()), sizeof(words)}));
  ASSERT_TRUE(patcher.append_nonalloc_section(".fi01.names", names));
  ASSERT_TRUE(patcher.append_nonalloc_section(
      ".fi01.symbols", {reinterpret_cast<const uint8_t *>(symbols.data()), sizeof(symbols)}));
  std::vector<Elf64_Shdr> sections = patcher.section_headers();
  ASSERT_EQ(sections.size(), code_index + 3);
  sections[code_index].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[code_index].sh_addr = symbols[1].st_value;
  sections[code_index + 1].sh_type = SHT_STRTAB;
  Elf64_Shdr &symtab = sections[code_index + 2];
  symtab.sh_type = SHT_SYMTAB;
  symtab.sh_entsize = sizeof(Elf64_Sym);
  symtab.sh_link = static_cast<uint32_t>(code_index + 1);
  symtab.sh_info = 1;
  std::vector<uint8_t> image = patcher.emit();
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  const FaultMutationContext context(image);
  ASSERT_TRUE(context.source().is_valid());
  ASSERT_EQ(context.source().allocated_executable_sections().size(), 2u);
  // Generic replacement can preserve this layout without proving recipe ownership.
  const Section &text = *context.source().text_sections().front();
  InstructionReplacement replacement = {.expected_bytes = std::vector<uint8_t>(4),
                                        .replacement_bytes = std::vector<uint8_t>(4)};
  std::memcpy(replacement.expected_bytes.data(), text.data(), 4);
  std::memcpy(replacement.replacement_bytes.data(), words.data(), 4);
  Instrumentor instrumentor(context.source(), kArch);
  instrumentor.add_point({.anchor_offset = 0,
                          .kind = InstrumentationKind::ReplaceWithInst,
                          .probe_symbol = {},
                          .probe_args = {},
                          .replacement = std::move(replacement)});
  const InstrumentedCodeObject emitted = instrumentor.patch();
  ASSERT_TRUE(emitted.errors.empty()) << emitted.errors.front();
  EXPECT_FALSE(emitted.elf_bytes.empty());
  ASSERT_EQ(emitted.sites.size(), 1u);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_NE(error.message.find("executable owners outside .text"), std::string::npos);
}

TEST(FaultDropBarrier, MultiTextHasAnExplicitDiagnostic) {
  const std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  const AmdGpuCodeObject source(image.data(), image.size());
  CodeObjectPatcher patcher(source);
  const uint32_t nop = build_s_nop(0, kArch);
  ASSERT_TRUE(patcher.append_nonalloc_section(
      ".text", std::span(reinterpret_cast<const uint8_t *>(&nop), sizeof(nop))));
  std::vector<uint8_t> multiple = patcher.emit();
  const AmdGpuCodeObject appended(multiple.data(), multiple.size());
  std::vector<Elf64_Shdr> sections = CodeObjectPatcher(appended).section_headers();
  Elf64_Shdr &extra = sections.back();
  extra.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  extra.sh_addr = 0x4000;
  Elf64_Ehdr header = {};
  std::memcpy(&header, multiple.data(), sizeof(header));
  std::memcpy(multiple.data() + header.e_shoff + (sections.size() - 1) * sizeof(extra), &extra,
              sizeof(extra));
  const FaultMutationContext context(multiple);
  ASSERT_EQ(context.source().text_sections().size(), 2u);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_NE(error.message.find("multi-text"), std::string::npos);
}

TEST(FaultDropBarrier, LongSameBlockPairUsesCompleteMembersWithoutDistanceCutoff) {
  std::vector<uint32_t> words(36, build_s_nop(0, kArch));
  words[0] = kSignal;
  words[34] = kWait;
  words[35] = kEnd;
  const FaultMutationContext context(make_gfx1250_barrier_elf(words));
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 1u);
  const FailureOr<DropBarrierResult> result =
      apply_drop_barrier(context, select_barrier_for_test(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().logical_mutation_count, 1u);
  EXPECT_EQ(result.value().selection.barrier().members.size(), 2u);
  EXPECT_TRUE(matches_drop_output(context, result.value().selection, result.value().image, error)
                  .succeeded())
      << error.message;
}

enum class FixtureLoadRange { Text, KernelDescriptor };

constexpr uint32_t kElfReadPermission = 0x4; // ELF PF_R; shared ELF constants expose only PF_X.

struct FixtureFileRange {
  uint64_t offset;
  uint64_t size;
};

std::optional<FixtureFileRange> fixture_file_range(const std::vector<uint8_t> &image,
                                                   FixtureLoadRange range) {
  const AmdGpuCodeObject source(image.data(), image.size());
  if (!source.is_valid() || source.text_sections().size() != 1)
    return std::nullopt;
  const Section &text = *source.text_sections().front();
  if (range == FixtureLoadRange::Text)
    return FixtureFileRange{.offset = text.sectionOffset(), .size = text.size()};
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  if (kernels.size() != 1)
    return std::nullopt;
  return FixtureFileRange{.offset = kernels.front().descriptor_file_offset,
                          .size = sizeof(amdhsa::kernel_descriptor_t)};
}

std::optional<size_t> load_segment_index(const std::vector<uint8_t> &image,
                                         FixtureLoadRange range) {
  const std::optional<FixtureFileRange> requested = fixture_file_range(image, range);
  if (!requested)
    return std::nullopt;
  const uint64_t file_offset = requested->offset;
  const uint64_t size = requested->size;
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  if (!segments)
    return std::nullopt;
  std::optional<size_t> found;
  for (size_t index = 0; index < segments->size(); ++index) {
    const Elf64_Phdr &segment = (*segments)[index];
    if (segment.p_type != PT_LOAD || file_offset < segment.p_offset)
      continue;
    const uint64_t delta = file_offset - segment.p_offset;
    if (delta <= segment.p_filesz && size <= segment.p_filesz - delta) {
      if (found)
        return std::nullopt;
      found = index;
    }
  }
  return found;
}

template <typename Edit>
void edit_segment(std::vector<uint8_t> &image, FixtureLoadRange range, Edit edit) {
  const std::optional<size_t> index = load_segment_index(image, range);
  ASSERT_TRUE(index) << "fixture must uniquely map the requested file range";
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  ASSERT_EQ(header.e_phentsize, sizeof(Elf64_Phdr));
  const size_t offset = header.e_phoff + *index * header.e_phentsize;
  ASSERT_LE(offset + sizeof(Elf64_Phdr), image.size());
  Elf64_Phdr segment = {};
  std::memcpy(&segment, image.data() + offset, sizeof(segment));
  edit(segment);
  std::memcpy(image.data() + offset, &segment, sizeof(segment));
}

void append_disjoint_load(std::vector<uint8_t> &image, FixtureFileRange file_range, uint32_t flags,
                          bool prepend) {
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  ASSERT_EQ(header.e_phentsize, sizeof(Elf64_Phdr));
  ASSERT_LT(header.e_phnum, std::numeric_limits<uint16_t>::max());
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  uint64_t memory_end = 0;
  for (const Elf64_Phdr &segment : *segments) {
    if (segment.p_type == PT_LOAD)
      memory_end = std::max(memory_end, segment.p_vaddr + segment.p_memsz);
  }
  const uint64_t memory_size = std::max(file_range.size, uint64_t{sizeof(uint32_t)});
  ASSERT_LT(memory_end, std::numeric_limits<uint64_t>::max() - memory_size);
  const Elf64_Phdr added = {.p_type = PT_LOAD,
                            .p_flags = flags,
                            .p_offset = file_range.offset,
                            .p_vaddr = memory_end + 1,
                            .p_paddr = memory_end + 1,
                            .p_filesz = file_range.size,
                            .p_memsz = memory_size,
                            .p_align = 1};
  ASSERT_LE(added.p_offset, image.size());
  ASSERT_LE(added.p_filesz, image.size() - added.p_offset);
  const uint64_t next_header_offset =
      header.e_phoff + uint64_t{header.e_phnum} * header.e_phentsize;
  ASSERT_LE(next_header_offset, image.size());
  ASSERT_LE(sizeof(added), image.size() - next_header_offset);
  ASSERT_LE(next_header_offset + sizeof(added), header.e_shoff);
  for (const Elf64_Shdr &section : CodeObjectPatcher(source).section_headers()) {
    if (section.sh_type != SHT_NOBITS && section.sh_size != 0) {
      ASSERT_TRUE(section.sh_offset >= next_header_offset + sizeof(added) ||
                  section.sh_offset + section.sh_size <= next_header_offset);
    }
  }
  // The fixture reserves space before its first payload for one more program header.
  if (prepend)
    std::memmove(image.data() + header.e_phoff + sizeof(added), image.data() + header.e_phoff,
                 size_t{header.e_phnum} * sizeof(added));
  std::memcpy(image.data() + (prepend ? header.e_phoff : next_header_offset), &added,
              sizeof(added));
  ++header.e_phnum;
  std::memcpy(image.data(), &header, sizeof(header));
}

void expect_load_mapping_rejection(const std::vector<uint8_t> &image) {
  FaultMutationError error;
  const FaultMutationContext malformed(image);
  EXPECT_TRUE(discover_drop_barriers(malformed, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
}

TEST(FaultDropBarrier, RejectsTextNotFileBacked) {
  for (const uint64_t file_size : {0u, 4u}) {
    SCOPED_TRACE(file_size);
    std::vector<uint8_t> image = make_gfx1250_barrier_elf();
    edit_segment(image, FixtureLoadRange::Text,
                 [&](Elf64_Phdr &segment) { segment.p_filesz = file_size; });
    expect_load_mapping_rejection(image);
  }
}

TEST(FaultDropBarrier, RejectsDescriptorNotFileBacked) {
  for (const uint64_t file_size : {size_t{0}, sizeof(amdhsa::kernel_descriptor_t) - 1}) {
    SCOPED_TRACE(file_size);
    std::vector<uint8_t> image = make_gfx1250_barrier_elf();
    edit_segment(image, FixtureLoadRange::KernelDescriptor,
                 [&](Elf64_Phdr &segment) { segment.p_filesz = file_size; });
    expect_load_mapping_rejection(image);
  }
}

TEST(FaultDropBarrier, RejectsTextLoadAddressMismatch) {
  std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  edit_segment(image, FixtureLoadRange::Text, [](Elf64_Phdr &segment) {
    segment.p_vaddr += 0x2000;
    segment.p_paddr += 0x2000;
  });
  expect_load_mapping_rejection(image);
}

TEST(FaultDropBarrier, RejectsDescriptorLoadAddressMismatch) {
  std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  edit_segment(image, FixtureLoadRange::KernelDescriptor, [](Elf64_Phdr &segment) {
    segment.p_vaddr += 0x2000;
    segment.p_paddr += 0x2000;
  });
  expect_load_mapping_rejection(image);
}

TEST(FaultDropBarrier, RejectsMissingOrNonExecutableTextLoad) {
  for (const bool missing : {false, true}) {
    SCOPED_TRACE(missing);
    std::vector<uint8_t> image = make_gfx1250_barrier_elf();
    edit_segment(image, FixtureLoadRange::Text, [&](Elf64_Phdr &segment) {
      if (missing)
        segment.p_type = PT_NOTE;
      else
        segment.p_flags &= ~PF_X;
    });
    expect_load_mapping_rejection(image);
  }
}

TEST(FaultDropBarrier, RejectsOverlappingLoadMappings) {
  for (const FixtureLoadRange range :
       {FixtureLoadRange::Text, FixtureLoadRange::KernelDescriptor}) {
    for (const bool zero_fill : {false, true}) {
      SCOPED_TRACE(range == FixtureLoadRange::Text ? "text load" : "descriptor load");
      SCOPED_TRACE(zero_fill);
      std::vector<uint8_t> image = make_gfx1250_barrier_elf();
      const AmdGpuCodeObject original(image.data(), image.size());
      ASSERT_TRUE(original.is_valid());
      ASSERT_EQ(original.text_sections().size(), 1u);
      const std::optional<size_t> index = load_segment_index(image, range);
      ASSERT_TRUE(index) << "fixture must uniquely map the requested file range";
      Elf64_Ehdr header = {};
      std::memcpy(&header, image.data(), sizeof(header));
      ASSERT_EQ(header.e_phentsize, sizeof(Elf64_Phdr));
      ASSERT_LT(*index, header.e_phnum);
      Elf64_Phdr overlap = {};
      std::memcpy(&overlap, image.data() + header.e_phoff + *index * header.e_phentsize,
                  sizeof(overlap));
      overlap.p_vaddr += 4;
      overlap.p_paddr += 4;
      overlap.p_filesz = zero_fill ? 0 : 4;
      overlap.p_memsz = 4;
      overlap.p_align = 1;
      const uint64_t next_header_offset =
          header.e_phoff + uint64_t{header.e_phnum} * header.e_phentsize;
      ASSERT_LE(next_header_offset + sizeof(overlap),
                original.text_sections().front()->sectionOffset());
      ++header.e_phnum;
      std::memcpy(image.data(), &header, sizeof(header));
      std::memcpy(image.data() + next_header_offset, &overlap, sizeof(overlap));
      expect_load_mapping_rejection(image);
    }
  }
}

TEST(FaultDropBarrier, RejectsPartialAndCompleteFileAliasesAtDisjointAddresses) {
  enum class AliasShape { Prefix, Interior, Suffix, Complete };
  for (const FixtureLoadRange range :
       {FixtureLoadRange::Text, FixtureLoadRange::KernelDescriptor}) {
    for (const AliasShape shape :
         {AliasShape::Prefix, AliasShape::Interior, AliasShape::Suffix, AliasShape::Complete}) {
      for (const uint32_t flags : {kElfReadPermission, kElfReadPermission | PF_X}) {
        for (const bool prepend : {false, true}) {
          SCOPED_TRACE(range == FixtureLoadRange::Text ? "text" : "descriptor");
          SCOPED_TRACE(static_cast<int>(shape));
          SCOPED_TRACE(flags);
          SCOPED_TRACE(prepend);
          std::vector<uint8_t> image = make_gfx1250_barrier_elf();
          const std::optional<FixtureFileRange> requested = fixture_file_range(image, range);
          const std::optional<FixtureFileRange> text =
              fixture_file_range(image, FixtureLoadRange::Text);
          ASSERT_TRUE(requested);
          ASSERT_TRUE(text);
          constexpr uint64_t word_size = sizeof(uint32_t);
          ASSERT_GE(requested->offset, word_size);
          ASSERT_GE(requested->size, 2 * word_size);
          FixtureFileRange alias = *requested;
          switch (shape) {
          case AliasShape::Prefix:
            alias = {.offset = requested->offset - word_size, .size = 2 * word_size};
            break;
          case AliasShape::Interior:
            alias = {.offset = requested->offset + word_size, .size = word_size};
            break;
          case AliasShape::Suffix:
            alias = {.offset = requested->offset + requested->size - word_size,
                     .size = 2 * word_size};
            break;
          case AliasShape::Complete:
            break;
          }
          ASSERT_NO_FATAL_FAILURE(append_disjoint_load(image, alias, flags, prepend));
          const std::vector<uint8_t> pristine = image;
          const FaultMutationContext context(image);
          ASSERT_TRUE(context.source().is_valid());
          FaultMutationError error;
          EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
          EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
          EXPECT_EQ(error.offset, 0u);
          // Descriptor bytes immediately follow text in this fixture. A descriptor-prefix
          // alias therefore also overlaps text and must fail the earlier text check.
          const bool overlaps_text =
              alias.offset < text->offset + text->size && text->offset < alias.offset + alias.size;
          EXPECT_NE(error.message.find(overlaps_text ? "file-backed executable .text"
                                                     : "descriptor load mapping"),
                    std::string::npos)
              << error.message;
          EXPECT_EQ(image, pristine);
          EXPECT_TRUE(std::ranges::equal(context.image(), pristine));
        }
      }
    }
  }
}

TEST(FaultDropBarrier, AcceptsTouchingAndEmptyFileRangesAtDisjointAddresses) {
  const std::vector<uint8_t> original = make_gfx1250_barrier_elf();
  const std::optional<FixtureFileRange> text = fixture_file_range(original, FixtureLoadRange::Text);
  const std::optional<FixtureFileRange> descriptor =
      fixture_file_range(original, FixtureLoadRange::KernelDescriptor);
  ASSERT_TRUE(text);
  ASSERT_TRUE(descriptor);
  constexpr uint64_t word_size = sizeof(uint32_t);
  ASSERT_GE(text->offset, word_size);
  ASSERT_EQ(text->offset + text->size, descriptor->offset);
  // Touch only the outside boundaries of the adjacent text/descriptor union.
  const FixtureFileRange ranges[] = {
      {.offset = text->offset - word_size, .size = word_size},
      {.offset = descriptor->offset + descriptor->size, .size = word_size},
      {.offset = text->offset + word_size, .size = 0},
      {.offset = descriptor->offset + word_size, .size = 0},
  };
  for (const FixtureFileRange range : ranges) {
    for (const uint32_t flags : {kElfReadPermission, kElfReadPermission | PF_X}) {
      for (const bool prepend : {false, true}) {
        SCOPED_TRACE(range.offset);
        SCOPED_TRACE(range.size);
        SCOPED_TRACE(flags);
        SCOPED_TRACE(prepend);
        std::vector<uint8_t> image = original;
        ASSERT_NO_FATAL_FAILURE(append_disjoint_load(image, range, flags, prepend));
        const std::vector<uint8_t> pristine = image;
        const FaultMutationContext context(image);
        ASSERT_TRUE(context.source().is_valid());
        FaultMutationError error;
        const FailureOr<FaultBarrierInventory> inventory =
            discover_drop_barriers(context, "test_kernel", error);
        ASSERT_TRUE(inventory.succeeded()) << error.message;
        const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
        const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
        ASSERT_TRUE(result.succeeded()) << error.message;
        EXPECT_TRUE(
            matches_drop_output(context, selection, result.value().image, error).succeeded())
            << error.message;
        EXPECT_EQ(image, pristine);
        EXPECT_TRUE(std::ranges::equal(context.image(), pristine));
      }
    }
  }
}

void exclude_trailing_text_padding(std::vector<uint8_t> &image) {
  const AmdGpuCodeObject original(image.data(), image.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.text_sections().size(), 1u);
  const Section &text = *original.text_sections().front();
  const std::optional<FixtureFileRange> descriptor =
      fixture_file_range(image, FixtureLoadRange::KernelDescriptor);
  ASSERT_TRUE(descriptor);
  ASSERT_EQ(text.sectionOffset() + text.size(), descriptor->offset);
  ASSERT_GE(text.size(), sizeof(uint32_t));
  uint32_t padding = 1;
  std::memcpy(&padding, image.data() + descriptor->offset - sizeof(padding), sizeof(padding));
  ASSERT_EQ(padding, 0u);
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  ASSERT_EQ(header.e_shentsize, sizeof(Elf64_Shdr));
  const std::optional<size_t> text_index = text.sectionHeaderIndex();
  ASSERT_TRUE(text_index);
  ASSERT_LT(*text_index, header.e_shnum);
  const uint64_t header_offset = header.e_shoff + *text_index * sizeof(Elf64_Shdr);
  ASSERT_LE(header_offset + sizeof(Elf64_Shdr), image.size());
  Elf64_Shdr text_header = {};
  std::memcpy(&text_header, image.data() + header_offset, sizeof(text_header));
  text_header.sh_size -= sizeof(padding);
  ASSERT_EQ(text_header.sh_offset + text_header.sh_size + sizeof(padding), descriptor->offset);
  std::memcpy(image.data() + header_offset, &text_header, sizeof(text_header));
}

void prepare_descriptor_relocation_load(std::vector<uint8_t> &image) {
  ASSERT_NO_FATAL_FAILURE(exclude_trailing_text_padding(image));
  edit_segment(image, FixtureLoadRange::KernelDescriptor, [](Elf64_Phdr &segment) {
    constexpr uint64_t word_size = sizeof(uint32_t);
    segment.p_offset -= word_size;
    segment.p_vaddr -= word_size;
    segment.p_paddr -= word_size;
    segment.p_filesz += word_size;
    // Keep writes beginning just before the descriptor and at its preload field loaded.
    segment.p_memsz += 2 * word_size;
  });
}

void append_descriptor_relocation(std::vector<uint8_t> &image, std::string_view entry,
                                  uint32_t type, int64_t descriptor_delta, bool dynamic,
                                  bool partial_tail) {
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.text_sections().size(), 1u);
  const Section &text = *source.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  const std::vector<KernelDescriptorInfo>::const_iterator descriptor = std::ranges::find_if(
      kernels, [&](const KernelDescriptorInfo &kernel) { return kernel.kernel_name == entry; });
  ASSERT_NE(descriptor, kernels.end());
  CodeObjectPatcher patcher(source);
  std::vector<Elf64_Shdr> sections = patcher.section_headers();
  std::optional<size_t> owner_index;
  uint64_t descriptor_address = 0;
  for (size_t index = 0; index < sections.size(); ++index) {
    const Elf64_Shdr &section = sections[index];
    if (section.sh_type == SHT_NOBITS || (section.sh_flags & SHF_ALLOC) == 0 ||
        descriptor->descriptor_file_offset < section.sh_offset)
      continue;
    const uint64_t delta = descriptor->descriptor_file_offset - section.sh_offset;
    if (delta > section.sh_size || sizeof(descriptor->descriptor) > section.sh_size - delta)
      continue;
    ASSERT_FALSE(owner_index);
    owner_index = index;
    descriptor_address = section.sh_addr + delta;
  }
  ASSERT_TRUE(owner_index);
  std::optional<size_t> symtab_index;
  std::optional<size_t> symbol_index;
  for (size_t index = 0; index < sections.size(); ++index) {
    const Elf64_Shdr &section = sections[index];
    if (section.sh_type != SHT_SYMTAB)
      continue;
    const std::optional<std::vector<Elf64_Sym>> symbols = read_elf_table<Elf64_Sym>(
        image, section.sh_offset, section.sh_size / sizeof(Elf64_Sym), section.sh_entsize);
    ASSERT_TRUE(symbols);
    for (size_t symbol = 0; symbol < symbols->size(); ++symbol) {
      if ((*symbols)[symbol].st_shndx != *owner_index ||
          (*symbols)[symbol].st_value != descriptor_address ||
          (*symbols)[symbol].st_size != sizeof(descriptor->descriptor))
        continue;
      ASSERT_FALSE(symbol_index);
      symtab_index = index;
      symbol_index = symbol;
    }
  }
  ASSERT_TRUE(symtab_index);
  ASSERT_TRUE(symbol_index);
  const std::optional<uint64_t> place = add_signed_offset(descriptor_address, descriptor_delta);
  ASSERT_TRUE(place);
  const uint64_t write_size =
      type == R_AMDGPU_ABS64 || type == R_AMDGPU_RELATIVE64 ? sizeof(uint64_t) : sizeof(uint32_t);
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(segments);
  ASSERT_TRUE(std::ranges::any_of(*segments, [&](const Elf64_Phdr &segment) {
    if (segment.p_type != PT_LOAD || *place < segment.p_vaddr)
      return false;
    const uint64_t delta = *place - segment.p_vaddr;
    return delta <= segment.p_memsz && write_size <= segment.p_memsz - delta;
  })) << "the complete relocation write must fit a loaded range";
  const Elf64_Rela relocation = {
      .r_offset = *place, .r_info = (uint64_t{*symbol_index} << 32) | type, .r_addend = 0};
  std::vector<uint8_t> payload(sizeof(relocation) + (partial_tail ? 1u : 0u), 0);
  std::memcpy(payload.data(), &relocation, sizeof(relocation));
  ASSERT_TRUE(patcher.append_nonalloc_section(".fi01.descriptor_rela", payload));
  image = patcher.emit();
  const AmdGpuCodeObject appended(image.data(), image.size());
  ASSERT_TRUE(appended.is_valid());
  sections = CodeObjectPatcher(appended).section_headers();
  Elf64_Shdr &relocs = sections.back();
  ASSERT_EQ(relocs.sh_size, payload.size());
  relocs.sh_type = SHT_RELA;
  relocs.sh_link = static_cast<uint32_t>(*symtab_index);
  relocs.sh_info = dynamic ? SHN_UNDEF : static_cast<uint32_t>(*owner_index);
  relocs.sh_entsize = sizeof(Elf64_Rela);
  relocs.sh_addralign = alignof(Elf64_Rela);
  std::memcpy(&header, image.data(), sizeof(header));
  ASSERT_EQ(header.e_shentsize, sizeof(Elf64_Shdr));
  ASSERT_EQ(header.e_shnum, sections.size());
  const uint64_t relocation_header = header.e_shoff + (sections.size() - 1) * header.e_shentsize;
  ASSERT_LE(relocation_header + sizeof(relocs), image.size());
  std::memcpy(image.data() + relocation_header, &relocs, sizeof(relocs));
  EXPECT_EQ(classify_rocr_relocation_section(header, sections, relocs),
            dynamic ? RocrRelocationSectionMode::Dynamic
                    : RocrRelocationSectionMode::ExplicitTarget);
  if (dynamic && descriptor_delta < 0) {
    EXPECT_FALSE(elf_relocation_place_is_allocated(header, sections, relocs, *place))
        << "the prefix control starts in loaded padding outside allocated sections";
  }
}

void expect_descriptor_relocation_rejection(const std::vector<uint8_t> &image,
                                            std::string_view selected_entry) {
  const std::vector<uint8_t> pristine = image;
  const FaultMutationContext context(image);
  ASSERT_TRUE(context.source().is_valid());
  Instrumentor shared(context.source(), kArch);
  TextEditError shared_error;
  ASSERT_NE(shared.instruction_boundaries(shared_error), nullptr) << shared_error.message;
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, selected_entry, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_EQ(error.offset, 0u);
  EXPECT_EQ(error.message, "dynamic relocation writes a kernel descriptor");
  EXPECT_EQ(image, pristine);
  EXPECT_TRUE(std::ranges::equal(context.image(), pristine));
}

TEST(FaultDropBarrier, RejectsActiveRelocationsOverlappingDescriptors) {
  using Descriptor = amdhsa::kernel_descriptor_t;
  for (const uint32_t type : {R_AMDGPU_ABS32_LO, R_AMDGPU_ABS32_HI, R_AMDGPU_ABS32, R_AMDGPU_ABS64,
                              R_AMDGPU_RELATIVE64}) {
    const uint64_t write_size =
        type == R_AMDGPU_ABS64 || type == R_AMDGPU_RELATIVE64 ? sizeof(uint64_t) : sizeof(uint32_t);
    const int64_t places[] = {
        static_cast<int64_t>(offsetof(Descriptor, group_segment_fixed_size)),
        static_cast<int64_t>(offsetof(Descriptor, kernel_code_entry_byte_offset)),
        static_cast<int64_t>(offsetof(Descriptor, kernel_code_properties)),
        static_cast<int64_t>(offsetof(Descriptor, kernarg_preload)),
        -static_cast<int64_t>(write_size / 2),
    };
    for (const int64_t place : places) {
      for (const bool partial_tail : {false, true}) {
        SCOPED_TRACE(type);
        SCOPED_TRACE(place);
        SCOPED_TRACE(partial_tail);
        std::vector<uint8_t> image =
            make_gfx1250_barrier_elf({kWaitDscnt, kSignal, kMov, kWait, kWaitDscnt, kEnd, 0});
        ASSERT_NO_FATAL_FAILURE(prepare_descriptor_relocation_load(image));
        ASSERT_NO_FATAL_FAILURE(
            append_descriptor_relocation(image, "test_kernel", type, place, true, partial_tail));
        ASSERT_NO_FATAL_FAILURE(expect_descriptor_relocation_rejection(image, "test_kernel"));
      }
    }
  }
}

TEST(FaultDropBarrier, PreservesStaticDescriptorRelocationsAndUnrelatedDynamicWrites) {
  struct Control {
    uint32_t type;
    // Descriptor-relative byte offset; negative values address loaded prefix padding.
    int64_t place;
    bool dynamic;
  };
  using Descriptor = amdhsa::kernel_descriptor_t;
  const Control controls[] = {
      {.type = R_AMDGPU_ABS64,
       .place = offsetof(Descriptor, kernel_code_entry_byte_offset),
       .dynamic = false},
      {.type = R_AMDGPU_RELATIVE64,
       .place = offsetof(Descriptor, kernarg_preload),
       .dynamic = false},
      {.type = R_AMDGPU_ABS32_LO, .place = -int64_t{sizeof(uint32_t)}, .dynamic = true},
      {.type = R_AMDGPU_ABS32_HI, .place = -int64_t{sizeof(uint32_t)}, .dynamic = true},
      {.type = R_AMDGPU_ABS32, .place = -int64_t{sizeof(uint32_t)}, .dynamic = true},
  };
  for (const Control &control : controls) {
    for (const bool partial_tail : {false, true}) {
      SCOPED_TRACE(control.type);
      SCOPED_TRACE(control.place);
      SCOPED_TRACE(control.dynamic);
      SCOPED_TRACE(partial_tail);
      std::vector<uint8_t> image =
          make_gfx1250_barrier_elf({kWaitDscnt, kSignal, kMov, kWait, kWaitDscnt, kEnd, 0});
      ASSERT_NO_FATAL_FAILURE(prepare_descriptor_relocation_load(image));
      ASSERT_NO_FATAL_FAILURE(append_descriptor_relocation(
          image, "test_kernel", control.type, control.place, control.dynamic, partial_tail));
      const std::vector<uint8_t> pristine = image;
      const FaultMutationContext context(image);
      ASSERT_TRUE(context.source().is_valid());
      FaultMutationError error;
      const FailureOr<FaultBarrierInventory> inventory =
          discover_drop_barriers(context, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
      const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      EXPECT_TRUE(matches_drop_output(context, selection, result.value().image, error).succeeded())
          << error.message;
      EXPECT_EQ(image, pristine);
      EXPECT_TRUE(std::ranges::equal(context.image(), pristine));
    }
  }
}

TEST(FaultDropBarrier, RejectsRelocatedUnselectedSiblingDescriptor) {
  for (const bool partial_tail : {false, true}) {
    SCOPED_TRACE(partial_tail);
    std::vector<uint8_t> image = make_gfx1250_two_kernel_barrier_elf();
    ASSERT_NO_FATAL_FAILURE(append_descriptor_relocation(
        image, "test_kernel1", R_AMDGPU_RELATIVE64,
        offsetof(amdhsa::kernel_descriptor_t, kernel_code_entry_byte_offset), true, partial_tail));
    ASSERT_NO_FATAL_FAILURE(expect_descriptor_relocation_rejection(image, "test_kernel0"));
  }
}

TEST(FaultDropBarrier, AcceptsFileBackedRangesInsideLargerLoadSegments) {
  // Keep actual padding between text and the descriptor. Extending the descriptor's
  // load backward into an adjacent instruction would create a forbidden file alias.
  std::vector<uint8_t> image =
      make_gfx1250_barrier_elf({kWaitDscnt, kSignal, kMov, kWait, kWaitDscnt, kEnd, 0});
  ASSERT_NO_FATAL_FAILURE(exclude_trailing_text_padding(image));
  for (const FixtureLoadRange range :
       {FixtureLoadRange::Text, FixtureLoadRange::KernelDescriptor}) {
    edit_segment(image, range, [](Elf64_Phdr &segment) {
      segment.p_offset -= 4;
      segment.p_vaddr -= 4;
      segment.p_paddr -= 4;
      segment.p_filesz += 4;
      segment.p_memsz += 4;
    });
  }
  const FaultMutationContext context(image);
  ASSERT_TRUE(context.source().is_valid());
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  const FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(matches_drop_output(context, selection, result.value().image, error).succeeded())
      << error.message;
}

TEST(FaultDropBarrier, CompleteSeparatedPairPreservesGuestWaitsAndMetadata) {
  const std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 1u);
  EXPECT_EQ(inventory.value().barriers().front().members[0].offset, 4u);
  EXPECT_EQ(inventory.value().barriers().front().members[1].offset, 12u);
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().logical_mutation_count, 1u);
  EXPECT_EQ(result.value().selection.barrier().members.size(), 2u);
  EXPECT_EQ(AmdGpuCodeObject(result.value().image.data(), result.value().image.size()).target_id(),
            context.source().target_id());
  EXPECT_EQ(result.value().execution_validation, FaultQualification::Unverified);
  EXPECT_EQ(result.value().detector_qualification, FaultQualification::Unverified);
  std::vector<uint8_t> expected = image;
  const uint32_t nop = build_s_nop(0, kArch);
  for (const uint64_t offset : {4u, 12u})
    std::memcpy(expected.data() + inventory.value().section().file_offset + offset, &nop, 4);
  EXPECT_EQ(result.value().image, expected);
  EXPECT_TRUE(matches_drop_output(context, selection, expected, error).succeeded())
      << error.message;
  const FaultMutationContext mutant(expected);
  EXPECT_TRUE(apply_drop_barrier(mutant, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultDropBarrier, ExactSecondPairSelectionLeavesRedundantFirstBarrier) {
  const std::vector<uint8_t> image =
      make_gfx1250_barrier_elf({kSignal, kWait, kSignal, kMov, kWait, kEnd});
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 2u);
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(context, select_barrier_for_test(inventory.value(), 1), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  FailureOr<FaultBarrierInventory> fresh =
      discover_drop_barriers(FaultMutationContext(result.value().image), "test_kernel", error);
  ASSERT_TRUE(fresh.succeeded()) << error.message;
  ASSERT_EQ(fresh.value().barriers().size(), 1u);
  EXPECT_EQ(fresh.value().barriers().front(), inventory.value().barriers().front());
  EXPECT_EQ(result.value().detector_qualification, FaultQualification::Unverified);
}

TEST(FaultDropBarrier, RejectsPartialInterleavedDynamicAndNamedLifecycle) {
  const std::vector<std::vector<uint32_t>> cases = {{kSignal, kEnd},
                                                    {kWait, kEnd},
                                                    {kSignal, kSignal, kWait, kEnd},
                                                    {kSignal, 0xbf94fffeu, kEnd},
                                                    {0xbe804e7du, kWait, kEnd},
                                                    {0xbe804e81u, 0xbf940001u, kEnd},
                                                    {0xbe804fc1u, kWait, kEnd},
                                                    {kSignal, 0xbe8050c1u, kWait, kEnd}};
  for (const std::vector<uint32_t> &words : cases) {
    const std::vector<uint8_t> image = make_gfx1250_barrier_elf(words);
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
    EXPECT_NE(error.code, FaultMutationErrorCode::None);
  }
}

TEST(FaultDropBarrier, SameBlockStaticPairInCycleAffectsAllOccurrences) {
  const std::vector<uint8_t> image =
      make_gfx1250_barrier_elf({kSignal, kMov, kWait, 0xbfa1fffcu, kEnd});
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 1u);
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(context, select_barrier_for_test(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  ASSERT_FALSE(result.value().assumptions.empty());
  EXPECT_NE(result.value().assumptions.front().find("every executed occurrence"),
            std::string::npos);
  const uint64_t branch_file_offset = inventory.value().section().file_offset + 12;
  EXPECT_EQ(std::memcmp(result.value().image.data() + branch_file_offset,
                        image.data() + branch_file_offset, 8),
            0);
}

TEST(FaultDropBarrier, RejectsCrossBlockPairAndUnresolvedControlFlow) {
  for (const std::vector<uint32_t> &words :
       std::vector<std::vector<uint32_t>>{{kSignal, build_s_branch(0, kArch), kWait, kEnd},
                                          {kSignal, kWait, build_s_branch(100, kArch)},
                                          {kSignal, kWait, 0xbe802000u}}) {
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(FaultMutationContext(make_gfx1250_barrier_elf(words)),
                                       "test_kernel", error)
                    .failed());
  }
}

TEST(FaultDropBarrier, AbiAdmissionFollowsSharedDbi) {
  std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  // Change only the ABI tag to compare admission with the shared DBI path.
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V4;
  const FaultMutationContext context(image);
  ASSERT_TRUE(context.source().is_valid());

  Instrumentor instrumentor(context.source(), kArch);
  instrumentor.add_point_by_offset(8);
  const InstrumentedCodeObject instrumented = instrumentor.patch();
  ASSERT_TRUE(instrumented.errors.empty()) << instrumented.errors.front();
  ASSERT_FALSE(instrumented.elf_bytes.empty());
  EXPECT_EQ(instrumented.elf_bytes[EI_ABIVERSION], image[EI_ABIVERSION]);

  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 1u);
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(matches_drop_output(context, selection, result.value().image, error).succeeded())
      << error.message;
  EXPECT_EQ(result.value().image[EI_ABIVERSION], image[EI_ABIVERSION]);
}

TEST(FaultDropBarrier, RejectsForeignHeadersAndNonexactEntry) {
  const std::vector<uint8_t> original = make_gfx1250_barrier_elf();
  for (const size_t index : {size_t{EI_CLASS}, size_t{EI_DATA}, size_t{EI_OSABI}}) {
    std::vector<uint8_t> image = original;
    image[index] = 0;
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  }
  const FaultMutationContext context(original);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  std::vector<uint8_t> foreign = original;
  Elf64_Ehdr header = {};
  std::memcpy(&header, foreign.data(), sizeof(header));
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX90A;
  std::memcpy(foreign.data(), &header, sizeof(header));
  const FaultMutationContext wrong_target(foreign);
  EXPECT_TRUE(discover_drop_barriers(wrong_target, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedTarget);
}

TEST(FaultDropBarrier, EmitterOracleRejectsHalfDropAndUnrelatedCorruption) {
  const std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = select_barrier_for_test(inventory.value());
  FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  const uint64_t text = inventory.value().section().file_offset;
  for (const uint64_t offset :
       {text + 4, text + 12, text + 8, text + 16, inventory.value().owner().descriptor_file_offset,
        uint64_t{offsetof(Elf64_Ehdr, e_flags)}}) {
    std::vector<uint8_t> candidate = result.value().image;
    candidate[offset] ^= 1;
    EXPECT_TRUE(matches_drop_output(context, selection, candidate, error).failed()) << offset;
  }
  EXPECT_TRUE(matches_drop_output(context, selection, image, error).failed());
}

TEST(FaultDropBarrier, CapabilityDescribesTransformationOnly) {
  const std::vector<FaultMutationCapability> capabilities = fault_mutation_capabilities();
  ASSERT_EQ(capabilities.size(), std::size(kDropRows));
  for (const DropTargetRow &row : kDropRows) {
    const std::vector<FaultMutationCapability>::const_iterator match =
        std::ranges::find_if(capabilities, [&](const FaultMutationCapability &capability) {
          return capability.target == row.target && capability.wave_size == row.wave_size;
        });
    ASSERT_NE(match, capabilities.end()) << row.name;
    EXPECT_EQ(match->architecture, row.arch);
    EXPECT_EQ(match->form,
              row.split ? StaticBarrierForm::BuiltinSplitPair : StaticBarrierForm::FullSingleton);
    EXPECT_FALSE(match->admission.empty());
  }
  constexpr std::array<IsaTargetDescriptor, 0> providers = {};
  const IsaTargetRegistry missing(providers);
  EXPECT_FALSE(static_barrier_policy(kTarget, missing));
  // The actual standalone target is linked to model-only providers.
  EXPECT_FALSE(
      default_isa_target_registry().find_gpu_target(kTarget)->capabilities.execution_implemented);
}

TEST(FaultDropBarrier, UserBuiltinIdsKeepDistinctStaticScopes) {
  for (const uint32_t selector : {193u, 195u}) {
    const int32_t id = -static_cast<int32_t>(selector - 192);
    const std::vector<uint8_t> image = make_gfx1250_barrier_elf(
        {0xbe804e00u | selector, 0xbf940000u | static_cast<uint16_t>(id), kEnd});
    const FaultMutationContext context(image);
    FaultMutationError error;
    FailureOr<FaultBarrierInventory> inventory =
        discover_drop_barriers(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().barriers().size(), 1u);
    EXPECT_EQ(inventory.value().barriers().front().barrier_id, id);
    EXPECT_EQ(inventory.value().barriers().front().scope,
              id == -1 ? StaticBarrierScope::Workgroup : StaticBarrierScope::Cluster);
    const FailureOr<DropBarrierResult> result =
        apply_drop_barrier(context, select_barrier_for_test(inventory.value()), error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    for (const std::string_view required_fact :
         {"clustered dispatch", "exactly one signal per workgroup",
          "after workgroup synchronization", "every wave waiting",
          "does not verify workgroup size or this protocol"}) {
      EXPECT_EQ(std::ranges::any_of(result.value().assumptions,
                                    [&](const std::string &assumption) {
                                      return assumption.find(required_fact) != std::string::npos;
                                    }),
                id == -3)
          << required_fact;
    }
  }
}

TEST(FaultDropBarrier, TrapFormsCannotConsumeDefaultUserBarrierSelection) {
  for (const DropTargetRow &row : kDropRows) {
    if (!row.split)
      continue;
    SCOPED_TRACE(row.name);
    for (const int32_t id : {-2, -4}) {
      SCOPED_TRACE(id);
      const uint32_t signal = 0xbe804e00u | static_cast<uint32_t>(192 - id);
      const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(id);
      const FaultMutationContext context(make_barrier_elf_for_target(
          row, {signal, wait, kSignal, kWait, build_s_endpgm(row.arch)}));
      FaultMutationError error;
      // Reject the unsupported form before an inert pair can take ordinal zero.
      EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
      EXPECT_EQ(error.offset, 0u);
      // RDNA4 has no -4 signal operand; decoding rejects it before qualification.
      const std::string_view diagnostic = row.arch == ROCJITSU_CODE_ARCH_RDNA4 && id == -4
                                              ? "invalid scalar source selector"
                                              : "trap-only";
      EXPECT_NE(error.message.find(diagnostic), std::string::npos) << error.message;
    }
  }
}

TEST(FaultDropBarrier, RejectsStandaloneAndMismatchedTrapWaits) {
  for (const DropTargetRow &row : kDropRows) {
    if (!row.split)
      continue;
    SCOPED_TRACE(row.name);
    for (const int32_t id : {-2, -4}) {
      SCOPED_TRACE(id);
      const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(id);
      for (const bool with_signal : {false, true}) {
        SCOPED_TRACE(with_signal);
        std::vector<uint32_t> words = {wait, build_s_endpgm(row.arch)};
        if (with_signal)
          words.insert(words.begin(), kSignal);
        const FaultMutationContext context(make_barrier_elf_for_target(row, words));
        FaultMutationError error;
        EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
        EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
        EXPECT_EQ(error.offset, with_signal ? 4u : 0u);
        EXPECT_NE(error.message.find("trap-only"), std::string::npos);
      }
    }
  }
}

TEST(FaultDropBarrier, RejectsMismatchedAdmittedWorkgroupAndClusterPairs) {
  for (const int32_t signal_id : {-1, -3}) {
    SCOPED_TRACE(signal_id);
    const int32_t wait_id = signal_id == -1 ? -3 : -1;
    const uint32_t signal = 0xbe804e00u | static_cast<uint32_t>(192 - signal_id);
    const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(wait_id);
    const FaultMutationContext context(make_gfx1250_barrier_elf({signal, wait, kEnd}));
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::IncompleteSelection);
    EXPECT_EQ(error.offset, 4u);
  }
}

TEST(FaultDropBarrier, RejectsAliasedEntryOwnersAndMalformedOmittedDescriptor) {
  for (const bool malformed : {false, true}) {
    std::vector<uint8_t> image = test::make_gfx950_two_kernel_elf({kSignal, kWait, kEnd}, 0);
    Elf64_Ehdr header = {};
    std::memcpy(&header, image.data(), sizeof(header));
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
    std::memcpy(image.data(), &header, sizeof(header));
    const AmdGpuCodeObject object(image.data(), image.size());
    const CodeObjectPatcher patcher(object);
    for (const Elf64_Shdr &section : patcher.section_headers()) {
      if (section.sh_type != SHT_SYMTAB)
        continue;
      for (size_t ordinal = 1; ordinal <= 2; ++ordinal) {
        Elf64_Sym symbol = {};
        std::memcpy(&symbol, image.data() + section.sh_offset + ordinal * sizeof(symbol),
                    sizeof(symbol));
        // Both descriptors belong to the .rodata section.
        const Elf64_Shdr owner = patcher.section_headers()[symbol.st_shndx];
        amdhsa::kernel_descriptor_t descriptor = {};
        const size_t offset = owner.sh_offset + symbol.st_value - owner.sh_addr;
        std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
        descriptor.kernel_code_properties |= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
        std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
        if (malformed && ordinal == 2) {
          symbol.st_size = 0; // scanner skips this owner; frontend must reject it.
          std::memcpy(image.data() + section.sh_offset + ordinal * sizeof(symbol), &symbol,
                      sizeof(symbol));
        }
      }
    }
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel0", error).failed());
    EXPECT_EQ(error.code, malformed ? FaultMutationErrorCode::UnsupportedForm
                                    : FaultMutationErrorCode::AmbiguousSelection)
        << error.message;
  }
}

TEST(FaultDropBarrier, Gfx1250WaveModeComesFromTargetDespiteMissingLegacyBit) {
  std::vector<uint8_t> image = make_gfx1250_barrier_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(descriptors.size(), 1u);
  amdhsa::kernel_descriptor_t descriptor = descriptors.front().descriptor;
  descriptor.kernel_code_properties &= ~amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  std::memcpy(image.data() + descriptors.front().descriptor_file_offset, &descriptor,
              sizeof(descriptor));
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<DropBarrierResult> result =
      apply_drop_barrier(context, select_barrier_for_test(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(std::memcmp(result.value().image.data() + descriptors.front().descriptor_file_offset,
                        image.data() + descriptors.front().descriptor_file_offset,
                        sizeof(descriptor)),
            0);
}

TEST(FaultDropBarrier, RejectsBarrierInsideClause) {
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(FaultMutationContext(make_gfx1250_barrier_elf(
                                         {0xbf850001u, kSignal, kWait, kEnd})),
                                     "test_kernel", error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
}

TEST(FaultDropBarrier, DiscoveryUsesFullTextClauseFactsAcrossUnreachableGap) {
  // Synthetic structural control: reachable CFG decoding skips the clause at 4,
  // while checked edits conservatively retain its linear membership at signal 12.
  const std::vector<uint8_t> image = make_gfx1250_barrier_elf(
      {build_s_branch(2, kArch), 0xbf850001u, build_s_nop(0, kArch), kSignal, kWait, kEnd});
  const FaultMutationContext context(image);
  TextEditError edit_error;
  Instrumentor instrumentor(context.source(), kArch);
  const InstructionBoundaries *boundaries = instrumentor.instruction_boundaries(edit_error);
  ASSERT_NE(boundaries, nullptr) << edit_error.message;
  const InstructionBoundaries::Entry *signal = boundaries->find(12);
  ASSERT_NE(signal, nullptr);
  ASSERT_TRUE(signal->clause_member);
  InstructionReplacement replacement = {.expected_bytes = std::vector<uint8_t>(4),
                                        .replacement_bytes = std::vector<uint8_t>(4)};
  const uint32_t nop = build_s_nop(0, kArch);
  std::memcpy(replacement.expected_bytes.data(), &kSignal, sizeof(kSignal));
  std::memcpy(replacement.replacement_bytes.data(), &nop, sizeof(nop));
  instrumentor.add_point({.anchor_offset = 12,
                          .kind = InstrumentationKind::ReplaceWithInst,
                          .probe_symbol = {},
                          .probe_args = {},
                          .replacement = std::move(replacement)});
  const InstrumentedCodeObject emitted = instrumentor.patch();
  EXPECT_FALSE(emitted.errors.empty());
  EXPECT_TRUE(emitted.elf_bytes.empty());
  EXPECT_TRUE(emitted.sites.empty());
  ASSERT_TRUE(emitted.edit_error);
  EXPECT_EQ(emitted.edit_error->code, TextEditErrorCode::Clause);
  EXPECT_EQ(emitted.edit_error->offset, 12u);

  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_EQ(error.offset, 12u);
}

TEST(FaultDropBarrier, RejectsTrapBetweenSignalAndWait) {
  FaultMutationError error;
  const std::vector<uint8_t> image = make_gfx1250_barrier_elf({kSignal, 0xbf900001u, kWait, kEnd});
  const FaultMutationContext context(image);
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
}

#ifdef RJ_FAULT_INJECTION_CORPUS_OBJECT
TEST(FaultDropBarrier, CompiledCorpusCompleteDrop) {
  std::ifstream input(RJ_FAULT_INJECTION_CORPUS_OBJECT, std::ios::binary);
  ASSERT_TRUE(input.is_open());
  const std::vector<uint8_t> image((std::istreambuf_iterator<char>(input)), {});
  ASSERT_FALSE(input.bad());
  const FaultMutationContext context(image);
  FaultMutationError error;
  constexpr std::string_view entry = "_Z16fa_barrier_epochPKtPjj";
  FailureOr<FaultBarrierInventory> inventory = discover_drop_barriers(context, entry, error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 1u);
  const StaticBarrier &group = inventory.value().barriers().front();
  EXPECT_EQ(group.barrier_id, -1);
  EXPECT_EQ(group.scope, StaticBarrierScope::Workgroup);
  ASSERT_EQ(group.members.size(), 2u);
  EXPECT_GT(group.members[1].offset, group.members[0].offset + group.members[0].bytes.size());
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(context, select_barrier_for_test(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().selection.barrier().members.size(), 2u);
  EXPECT_TRUE(matches_drop_output(context, result.value().selection, result.value().image, error)
                  .succeeded())
      << error.message;
  std::ofstream output(RJ_FAULT_INJECTION_CORPUS_MUTANT, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(output.is_open());
  const std::vector<uint8_t> &mutant = result.value().image;
  output.write(reinterpret_cast<const char *>(mutant.data()), mutant.size());
  output.close();
  ASSERT_TRUE(output.good());
}
#endif

} // namespace
} // namespace rocjitsu
