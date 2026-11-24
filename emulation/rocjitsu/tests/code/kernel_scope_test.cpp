// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Coverage for the shared kernel reachability walk (code/kernel_scope.h), which
// DBT and DBI both use to decide which blocks belong to one kernel.

#include "../dbi_test_util.h"
#include "decode_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/gfx1250_vgpr_msb.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

constexpr rocjitsu::IsaFamily kArch = rocjitsu::IsaFamily::Cdna4;

// s_cbranch_scc0 is SOPP opcode 5 on CDNA; simm16 is the dword delta from the
// next instruction.
[[nodiscard]] constexpr uint32_t build_s_cbranch_scc0(int16_t offset_dwords) {
  return pack_sopp(5, static_cast<uint16_t>(offset_dwords));
}

class TestTextSection : public Section {
public:
  TestTextSection(std::unique_ptr<char[]> data, std::size_t size)
      : Section(".text", std::move(data)), size_(size) {}

  std::size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  std::size_t size_;
};

class TestCodeObject : public CodeObject {
public:
  explicit TestCodeObject(std::vector<uint32_t> words) {
    const auto byte_size = words.size() * sizeof(uint32_t);
    image_.resize(byte_size);
    std::memcpy(image_.data(), words.data(), byte_size);

    auto data = std::make_unique<char[]>(byte_size);
    std::memcpy(data.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<TestTextSection>(std::move(data), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

// Two kernels in one .text. Kernel A branches into kernel B's entry, which is
// the edge the kernel_entries stop exists to cut.
//
//   0x00  A entry   s_cbranch_scc0 -> 0x08     (fallthrough 0x04)
//   0x04  A mid     s_branch       -> 0x10     (into B's entry)
//   0x08  A tail    s_nop
//   0x0c            s_endpgm
//   0x10  B entry   s_cbranch_scc0 -> 0x18     (fallthrough 0x14)
//   0x14  B mid     s_branch       -> 0x18
//   0x18  B tail    s_endpgm
constexpr uint64_t kEntryA = 0x00;
constexpr uint64_t kMidA = 0x04;
constexpr uint64_t kTailA = 0x08;
constexpr uint64_t kEntryB = 0x10;
constexpr uint64_t kMidB = 0x14;
constexpr uint64_t kTailB = 0x18;

[[nodiscard]] std::vector<uint32_t> two_kernel_words() {
  constexpr util::FailureOr<uint32_t> kBranch2Result = build_s_branch(2, kArch);
  static_assert(kBranch2Result.succeeded());
  constexpr util::FailureOr<uint32_t> kNopResult = build_s_nop(0, kArch);
  static_assert(kNopResult.succeeded());
  constexpr util::FailureOr<uint32_t> kEndpgmResult = build_s_endpgm(kArch);
  static_assert(kEndpgmResult.succeeded());
  constexpr util::FailureOr<uint32_t> kBranch0Result = build_s_branch(0, kArch);
  static_assert(kBranch0Result.succeeded());

  return {
      build_s_cbranch_scc0(1), // 0x00: -> A tail at 0x08.
      kBranch2Result.value(),  // 0x04: -> B entry at 0x10.
      kNopResult.value(),      // 0x08: A tail, first of two instructions.
      kEndpgmResult.value(),   // 0x0c.
      build_s_cbranch_scc0(1), // 0x10: -> B tail at 0x18.
      kBranch0Result.value(),  // 0x14: -> B tail at 0x18.
      kEndpgmResult.value(),   // 0x18.
  };
}

[[nodiscard]] std::vector<uint64_t> start_offsets(const std::vector<BasicBlock *> &blocks) {
  std::vector<uint64_t> offsets;
  offsets.reserve(blocks.size());
  for (const BasicBlock *block : blocks)
    offsets.push_back(block->start_offset());
  return offsets;
}

class KernelScopeTest : public ::testing::Test {
protected:
  void SetUp() override {
    decoder_ = Decoder::create(kArch);
    ASSERT_NE(decoder_, nullptr);
    object_ = std::make_unique<TestCodeObject>(two_kernel_words());
    blocks_ = build_valid_blocks(*object_, *decoder_, kArch);
    ASSERT_EQ(blocks_.size(), 6u);
    offset_index_ = build_block_offset_index(blocks_);
    position_index_ = build_block_position_index(blocks_);

    // Without this edge the kernel_entries stop would have nothing to cut and
    // ScopeStopsAtAnotherKernelEntry would pass vacuously.
    const BasicBlock *mid_a = block_for_offset(offset_index_, kMidA);
    ASSERT_NE(mid_a, nullptr);
    ASSERT_TRUE(std::ranges::any_of(mid_a->successors(), [](const BasicBlock *succ) {
      return succ->start_offset() == kEntryB;
    })) << "kernel A must branch into kernel B's entry";
  }

  [[nodiscard]] std::vector<BasicBlock *> scope_from(uint64_t entry_offset,
                                                     const KernelScopeSpec &spec) {
    BasicBlock *entry = block_for_offset(offset_index_, entry_offset);
    EXPECT_NE(entry, nullptr);
    if (entry == nullptr)
      return {};
    return reachable_kernel_blocks(blocks_, offset_index_, position_index_, *entry, spec);
  }

  std::unique_ptr<Decoder> decoder_;
  std::unique_ptr<TestCodeObject> object_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  BlockOffsetIndex offset_index_;
  BlockPositionIndex position_index_;
};

std::vector<uint8_t> two_owner_image(const std::vector<uint32_t> &words, uint64_t second_entry,
                                     uint32_t flags = EF_AMDGPU_MACH_AMDGCN_GFX950) {
  std::vector<uint8_t> image = test::make_amdgpu_two_kernel_elf(words, 0, 3, flags);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const size_t offset =
      0x100 + words.size() * sizeof(uint32_t) + sizeof(amdhsa::kernel_descriptor_t);
  amdhsa::kernel_descriptor_t second = {};
  std::memcpy(&second, image.data() + offset, sizeof(second));
  second.kernel_code_entry_byte_offset += static_cast<int64_t>(second_entry);
  std::memcpy(image.data() + offset, &second, sizeof(second));
  return image;
}

std::vector<uint32_t> shared_helper_words(bool indirect, bool clobber,
                                          rocjitsu::IsaFamily arch = kArch) {
  const uint32_t nop = build_s_nop(0, arch).value();
  const uint32_t end = build_s_endpgm(arch).value();
  const auto caller = [&](uint32_t entry) {
    if (!indirect)
      return std::vector<uint32_t>{build_s_call_b64(30, (512 - entry - 4) / 4, arch).value(), end};
    return std::vector<uint32_t>{build_s_getpc_b64(8, arch).value(),
                                 build_s_add_u32(8, 8, 255, arch).value(),
                                 512 - entry - 4,
                                 build_s_addc_u32(9, 9, 128, arch).value(),
                                 build_s_swappc_b64(30, 8, arch).value(),
                                 end};
  };
  std::vector<uint32_t> words = caller(0);
  words.resize(128, nop);
  const std::vector<uint32_t> second = caller(256);
  std::ranges::copy(second, words.begin() + 64);
  words.push_back(clobber ? build_s_mov_b32(30, 0, arch).value() : nop);
  words.push_back(build_s_setpc_b64(30, arch).value());
  return words;
}

void write_pc_builder(std::vector<uint32_t> &words, uint32_t offset, uint16_t pair,
                      uint32_t target) {
  const std::array<uint32_t, 4> builder = {
      build_s_getpc_b64(pair, kArch).value(),
      build_s_add_u32(pair, pair, 255, kArch).value(),
      target - offset - 4,
      build_s_addc_u32(pair + 1, pair + 1, 128, kArch).value(),
  };
  std::ranges::copy(builder, words.begin() + offset / sizeof(uint32_t));
}

void add_function_symbol(std::vector<uint8_t> &image, uint64_t offset, bool dynamic,
                         uint8_t symbol_type = kElfSymbolTypeFunc) {
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  for (size_t index = 0; index < header.e_shnum; ++index) {
    Elf64_Shdr symbols = {};
    const size_t section_offset = header.e_shoff + index * sizeof(Elf64_Shdr);
    std::memcpy(&symbols, image.data() + section_offset, sizeof(symbols));
    if (symbols.sh_type != SHT_SYMTAB)
      continue;
    const size_t strings_header_offset = header.e_shoff + symbols.sh_link * sizeof(Elf64_Shdr);
    Elf64_Shdr strings = {};
    std::memcpy(&strings, image.data() + strings_header_offset, sizeof(strings));
    std::vector<uint8_t> names(image.begin() + strings.sh_offset,
                               image.begin() + strings.sh_offset + strings.sh_size);
    const uint32_t name_offset = static_cast<uint32_t>(names.size());
    constexpr char name[] = "shared_helper";
    names.insert(names.end(), name, name + sizeof(name));
    std::vector<uint8_t> table(image.begin() + symbols.sh_offset,
                               image.begin() + symbols.sh_offset + symbols.sh_size);
    Elf64_Sym symbol = {};
    symbol.st_name = name_offset;
    symbol.st_info = elf_symbol_info(kElfSymbolBindGlobal, symbol_type);
    symbol.st_shndx = 1;
    symbol.st_value = 0x1100 + offset;
    symbol.st_size = 4;
    const size_t old_size = table.size();
    table.resize(old_size + sizeof(symbol));
    std::memcpy(table.data() + old_size, &symbol, sizeof(symbol));
    strings.sh_offset = image.size();
    strings.sh_size = names.size();
    image.insert(image.end(), names.begin(), names.end());
    symbols.sh_offset = image.size();
    symbols.sh_size = table.size();
    if (dynamic)
      symbols.sh_type = SHT_DYNSYM;
    image.insert(image.end(), table.begin(), table.end());
    std::memcpy(image.data() + section_offset, &symbols, sizeof(symbols));
    std::memcpy(image.data() + strings_header_offset, &strings, sizeof(strings));
    return;
  }
  ADD_FAILURE() << "fixture has no symbol table";
}

TEST(KernelOwnershipTest, SourceBoundSharedDirectHelperHasAllOwners) {
  const uint32_t call_a = build_s_call_b64(30, 3, kArch).value();
  const uint32_t call_b = build_s_call_b64(30, 1, kArch).value();
  const uint32_t end = build_s_endpgm(kArch).value();
  const uint32_t nop = build_s_nop(0, kArch).value();
  const uint32_t ret = build_s_setpc_b64(30, kArch).value();
  const std::vector<uint32_t> words = {call_a, end, call_b, end, nop, ret};
  std::vector<uint8_t> image = test::make_gfx950_two_kernel_elf(words, 0, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  constexpr size_t descriptor_bytes = sizeof(amdhsa::kernel_descriptor_t);
  const size_t second_offset = 0x100 + words.size() * sizeof(uint32_t) + descriptor_bytes;
  amdhsa::kernel_descriptor_t second = {};
  std::memcpy(&second, image.data() + second_offset, sizeof(second));
  second.kernel_code_entry_byte_offset += 8;
  std::memcpy(image.data() + second_offset, &second, sizeof(second));
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  ASSERT_TRUE(ownership.succeeded()) << error;
  const std::optional<std::span<const KernelExecutionOwner>> owners =
      ownership.value().owners_at(16);
  ASSERT_TRUE(owners);
  ASSERT_EQ(owners->size(), 2u);
  EXPECT_EQ((*owners)[0],
            (KernelExecutionOwner{.kernel_index = 0, .proof = KernelOwnerProof::DirectCall}));
  EXPECT_EQ((*owners)[1],
            (KernelExecutionOwner{.kernel_index = 1, .proof = KernelOwnerProof::DirectCall}));
  EXPECT_FALSE(ownership.value().owners_at(17));
  EXPECT_FALSE(ownership.value().owners_at(words.size() * sizeof(uint32_t)));
  EXPECT_FALSE(ownership.value().owners_at(std::numeric_limits<uint64_t>::max()));
  EXPECT_EQ(ownership.value().instruction_at(0, 17), nullptr);
  EXPECT_EQ(ownership.value().instruction_at(0, words.size() * sizeof(uint32_t)), nullptr);
  EXPECT_TRUE(ownership.value().belongs_to(*source));
  EXPECT_TRUE(KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX942, error).failed());
}

TEST(KernelOwnershipTest, RecoveredAndDirectSharedCallsProveAllOwners) {
  for (bool indirect : {false, true}) {
    const std::vector<uint8_t> image = two_owner_image(shared_helper_words(indirect, false), 256);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
    ASSERT_TRUE(ownership.succeeded()) << error;
    const std::optional<std::span<const KernelExecutionOwner>> owners =
        ownership.value().owners_at(512);
    ASSERT_TRUE(owners);
    ASSERT_EQ(owners->size(), 2u);
    EXPECT_EQ((*ownership.value().kernels())[(*owners)[0].kernel_index].kernel_name,
              "test_kernel0");
    EXPECT_EQ((*ownership.value().kernels())[(*owners)[1].kernel_index].kernel_name,
              "test_kernel1");
    EXPECT_EQ((*owners)[0].proof,
              indirect ? KernelOwnerProof::RecoveredIndirectCall : KernelOwnerProof::DirectCall);
    EXPECT_EQ((*owners)[1].proof, (*owners)[0].proof);
  }
}

TEST(KernelOwnershipTest, RejectsSharedIndirectTargetsWithoutKernelProvenance) {
  std::vector<uint32_t> words(194, build_s_nop(0, kArch).value());
  write_pc_builder(words, 0, 8, 768);
  words[4] = build_s_branch(123, kArch).value(); // 16 -> shared dispatch at 512.
  write_pc_builder(words, 256, 8, 772);
  words[68] = build_s_branch(59, kArch).value(); // 272 -> shared dispatch at 512.
  words[128] = build_s_setpc_b64(8, kArch).value();
  words[192] = build_s_endpgm(kArch).value();
  words[193] = build_s_endpgm(kArch).value();
  const std::vector<uint8_t> image = two_owner_image(words, 256);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  EXPECT_TRUE(ownership.failed());
  EXPECT_EQ(error, "shared indirect targets require kernel-specific proof");

  write_pc_builder(words, 256, 8, 768);
  const std::vector<uint8_t> same_target_image = two_owner_image(words, 256);
  const std::shared_ptr<const AmdGpuCodeObject> same_target_source =
      std::make_shared<AmdGpuCodeObject>(same_target_image.data(), same_target_image.size());
  const FailureOr<KernelOwnership> same_target =
      KernelOwnership::build(same_target_source, ROCJITSU_CODE_TARGET_GFX950, error);
  ASSERT_TRUE(same_target.succeeded()) << error;
  ASSERT_TRUE(same_target.value().owners_at(768));
  EXPECT_EQ(same_target.value().owners_at(768)->size(), 2u);
}

TEST(KernelOwnershipTest, RejectsPcBuilderCarriedAcrossClobberingIndirectCall) {
  std::vector<uint32_t> words(194, build_s_nop(0, kArch).value());
  write_pc_builder(words, 0, 8, 768);
  write_pc_builder(words, 16, 10, 512);
  words[8] = build_s_swappc_b64(30, 10, kArch).value();
  words[9] = build_s_setpc_b64(8, kArch).value();
  words[64] = build_s_branch(128, kArch).value(); // B enters the actual target at 772.
  write_pc_builder(words, 512, 8, 772);           // Callee replaces A's pending target.
  words[132] = build_s_setpc_b64(30, kArch).value();
  words[192] = build_s_endpgm(kArch).value();
  words[193] = build_s_endpgm(kArch).value();
  const std::vector<uint8_t> image = two_owner_image(words, 256);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  EXPECT_TRUE(ownership.failed());
  EXPECT_EQ(error, "call-carried PC builder requires callee preservation proof");
}

TEST(KernelOwnershipTest, FreshPcBuilderAfterIndirectCallProvesActualOwners) {
  std::vector<uint32_t> words(194, build_s_nop(0, kArch).value());
  write_pc_builder(words, 0, 8, 768);
  write_pc_builder(words, 16, 10, 512);
  words[8] = build_s_swappc_b64(30, 10, kArch).value();
  write_pc_builder(words, 36, 8, 772);
  words[13] = build_s_setpc_b64(8, kArch).value();
  words[64] = build_s_branch(128, kArch).value();
  write_pc_builder(words, 512, 8, 772);
  words[132] = build_s_setpc_b64(30, kArch).value();
  words[192] = build_s_endpgm(kArch).value();
  words[193] = build_s_endpgm(kArch).value();
  const std::vector<uint8_t> image = two_owner_image(words, 256);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  ASSERT_TRUE(ownership.succeeded()) << error;
  ASSERT_TRUE(ownership.value().owners_at(772));
  EXPECT_EQ(ownership.value().owners_at(772)->size(), 2u);
}

TEST(KernelOwnershipTest, SignedTemplateRequiresClosedArithmeticArmEntries) {
  for (bool padded : {false, true}) {
    for (size_t scenario = 0; scenario < 6; ++scenario) {
      SCOPED_TRACE(padded);
      SCOPED_TRACE(scenario);
      const bool alternate_entry = scenario == 1;
      const uint16_t temporary = scenario == 4 ? 8 : scenario == 5 ? 9 : 12;
      const uint32_t target = padded ? 64 : 52;
      const uint32_t add_arm = padded ? 40 : 36;
      const uint32_t other_target = padded ? 768 : 512;
      std::vector<uint32_t> words(196, build_s_nop(0, kArch).value());
      const std::vector<uint32_t> prefix = {
          pack_sop1(0x1c, 8, 0),
          pack_sop2(2, temporary, 255, 132),
          target - 8,
          pack_sopc(3, scenario == 3 ? 13 : temporary, 128),
          pack_sopp(scenario == 2 ? 4 : 5, static_cast<uint16_t>((add_arm - 20) / 4)),
          pack_sop1(0x30, temporary, temporary),
          pack_sop2(1, 8, 8, temporary),
          pack_sop2(5, 9, 9, 128),
          build_s_setpc_b64(8, kArch).value(),
      };
      std::ranges::copy(prefix, words.begin());
      words[add_arm / 4] = pack_sop2(0, 8, 8, temporary);
      words[add_arm / 4 + 1] = pack_sop2(4, 9, 9, 128);
      words[add_arm / 4 + 2] = build_s_setpc_b64(8, kArch).value();
      const std::array<uint32_t, 2> store =
          cdna4::build_ds(cdna4::kDsWriteB32Ds, {.addr = 2, .data0 = 3});
      for (uint32_t site : {target, other_target}) {
        words[site / 4] = store[0];
        words[site / 4 + 1] = store[1];
        words[site / 4 + 2] = build_s_endpgm(kArch).value();
      }
      if (alternate_entry) {
        write_pc_builder(words, 256, 8, other_target);
        words[68] = build_s_mov_b32(12, 128, kArch).value();
        words[69] =
            build_s_branch(static_cast<int16_t>((static_cast<int32_t>(add_arm) - 280) / 4), kArch)
                .value();
      } else {
        words[64] = build_s_endpgm(kArch).value();
      }
      const std::vector<uint8_t> image = two_owner_image(words, 256);
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      const FailureOr<KernelOwnership> ownership =
          KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
      if (scenario != 0) {
        EXPECT_TRUE(ownership.failed());
      } else {
        ASSERT_TRUE(ownership.succeeded()) << error;
        ASSERT_TRUE(ownership.value().owners_at(target));
        EXPECT_EQ(ownership.value().owners_at(target)->size(), 1u);
      }
    }
  }
}

TEST(KernelOwnershipTest, RejectsUnprovedBuilderEnteringAnotherCallee) {
  for (bool loop_backedge : {false, true}) {
    SCOPED_TRACE(loop_backedge);
    std::vector<uint32_t> words(196, build_s_nop(0, kArch).value());
    write_pc_builder(words, 0, 8, 768);
    write_pc_builder(words, 16, 10, 512);
    words[8] = build_s_swappc_b64(30, 10, kArch).value();
    write_pc_builder(words, 36, 10, 640);
    words[13] = build_s_swappc_b64(30, 10, kArch).value();
    words[14] = build_s_endpgm(kArch).value();
    words[64] = build_s_branch(128, kArch).value();
    write_pc_builder(words, 512, 8, 772);
    words[132] = build_s_setpc_b64(30, kArch).value();
    words[160] = build_s_cbranch_scc0(1);
    words[161] = build_s_setpc_b64(30, kArch).value();
    if (loop_backedge)
      words[162] = build_s_cbranch_scc0(-3);
    words[163] = build_s_setpc_b64(8, kArch).value();
    words[192] = build_s_endpgm(kArch).value();
    const std::array<uint32_t, 2> store =
        cdna4::build_ds(cdna4::kDsWriteB32Ds, {.addr = 2, .data0 = 3});
    words[193] = store[0];
    words[194] = store[1];
    words[195] = build_s_endpgm(kArch).value();
    const std::vector<uint8_t> image = two_owner_image(words, 256);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
    EXPECT_TRUE(ownership.failed());
  }
}

TEST(KernelOwnershipTest, SplitPcAdjustmentRequiresMatchingHighHalfAndCarry) {
  struct Case {
    uint32_t start = 0;
    uint16_t low_op = cdna4::kSAddU32Sop2;
    uint32_t low = 508;
    uint16_t high_op = cdna4::kSAddcU32Sop2;
    uint32_t high = 0;
    bool gap = false;
    bool overwrite_scc = false;
    bool finish = true;
    bool proved = true;
  };
  const std::array<Case, 11> cases = {{
      {},
      {.high = 0xffffffffu, .proved = false},
      {.high_op = cdna4::kSSubbU32Sop2, .proved = false},
      {.start = 512, .low = 0xfffffe00u, .high = 0xffffffffu},
      {.start = 512, .low = 0xfffffe00u, .proved = false},
      {.start = 512, .low_op = cdna4::kSSubU32Sop2, .low = 512, .high_op = cdna4::kSSubbU32Sop2},
      {.low_op = cdna4::kSSubU32Sop2,
       .low = 0xfffffe04u,
       .high_op = cdna4::kSSubbU32Sop2,
       .high = 0xffffffffu},
      {.low_op = cdna4::kSAddI32Sop2, .proved = false},
      {.gap = true},
      {.overwrite_scc = true, .proved = false},
      {.finish = false, .proved = false},
  }};
  for (size_t index = 0; index < cases.size(); ++index) {
    SCOPED_TRACE(index);
    const Case &test = cases[index];
    std::vector<uint32_t> words(144, build_s_nop(0, kArch).value());
    words[64] = build_s_endpgm(kArch).value();
    const uint32_t target = test.start == 0 ? 512 : 4;
    words[target / 4] = build_s_endpgm(kArch).value();
    if (test.start != 0)
      words[0] = build_s_branch(127, kArch).value();
    size_t cursor = test.start / 4;
    words[cursor++] = build_s_getpc_b64(8, kArch).value();
    words[cursor++] = pack_sop2(test.low_op, 8, 8, 255);
    words[cursor++] = test.low;
    if (test.gap)
      words[cursor++] = build_s_mov_b32(20, 128, kArch).value();
    if (test.overwrite_scc)
      words[cursor++] = build_s_cmp_lg_u32(128, 129, kArch).value();
    if (test.finish) {
      words[cursor++] = pack_sop2(test.high_op, 9, 9, 255);
      words[cursor++] = test.high;
    }
    words[cursor] = build_s_setpc_b64(8, kArch).value();
    const std::vector<uint8_t> image = two_owner_image(words, 256);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
    if (test.proved) {
      ASSERT_TRUE(ownership.succeeded()) << error;
      ASSERT_TRUE(ownership.value().owners_at(target));
      EXPECT_EQ(ownership.value().owners_at(target)->size(), 1u);
    } else {
      EXPECT_TRUE(ownership.failed());
    }
  }
}

TEST(KernelOwnershipTest, TemporaryPcAdjustmentRequiresProvedSignAndRegisters) {
  for (bool subtract : {false, true}) {
    for (size_t scenario = 0; scenario < 4; ++scenario) {
      const uint16_t temporary = scenario == 1 ? 8 : scenario == 2 ? 9 : 12;
      const bool negative = subtract != (scenario == 3);
      SCOPED_TRACE(subtract);
      SCOPED_TRACE(scenario);
      std::vector<uint32_t> words(144, build_s_nop(0, kArch).value());
      words[64] = build_s_endpgm(kArch).value();
      const uint32_t target = negative ? 4 : 512;
      words[target / 4] = build_s_endpgm(kArch).value();
      if (negative)
        words[0] = build_s_branch(127, kArch).value();
      size_t cursor = negative ? 128 : 0;
      words[cursor++] = build_s_getpc_b64(8, kArch).value();
      words[cursor++] = pack_sop2(cdna4::kSAddI32Sop2, temporary, 255, 132);
      words[cursor++] = negative ? 0xfffffdfcu : 504;
      if (subtract)
        words[cursor++] = pack_sop1(0x30, temporary, temporary);
      words[cursor++] =
          pack_sop2(subtract ? cdna4::kSSubU32Sop2 : cdna4::kSAddU32Sop2, 8, 8, temporary);
      words[cursor++] =
          pack_sop2(subtract ? cdna4::kSSubbU32Sop2 : cdna4::kSAddcU32Sop2, 9, 9, 128);
      words[cursor] = build_s_setpc_b64(8, kArch).value();
      const std::vector<uint8_t> image = two_owner_image(words, 256);
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      const FailureOr<KernelOwnership> ownership =
          KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
      if (scenario == 0) {
        ASSERT_TRUE(ownership.succeeded()) << error;
        ASSERT_TRUE(ownership.value().owners_at(target));
        EXPECT_EQ(ownership.value().owners_at(target)->size(), 1u);
      } else {
        EXPECT_TRUE(ownership.failed());
      }
    }
  }
}

TEST(KernelOwnershipTest, PendingCarryCannotCrossTemporaryAdjustment) {
  std::vector<uint32_t> words(144, build_s_nop(0, kArch).value());
  words[0] = build_s_branch(127, kArch).value();
  words[2] = build_s_endpgm(kArch).value();
  words[64] = build_s_endpgm(kArch).value();
  const std::vector<uint32_t> builder = {
      build_s_getpc_b64(8, kArch).value(),
      pack_sop2(cdna4::kSAddU32Sop2, 8, 8, 255),
      0xfffffe00u,
      pack_sop2(cdna4::kSAddI32Sop2, 12, 255, 132),
      0,
      pack_sop2(cdna4::kSAddU32Sop2, 8, 8, 12),
      pack_sop2(cdna4::kSAddcU32Sop2, 9, 9, 128),
      pack_sop2(cdna4::kSAddcU32Sop2, 9, 9, 255),
      0xffffffffu,
      build_s_setpc_b64(8, kArch).value(),
  };
  std::ranges::copy(builder, words.begin() + 128);
  const std::vector<uint8_t> image = two_owner_image(words, 256);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  EXPECT_TRUE(ownership.failed());
}

TEST(KernelOwnershipTest, Gfx1250PcTemporaryMustBeWritableSgpr) {
  constexpr IsaFamily arch = IsaFamily::Cdna5;
  for (bool split : {false, true}) {
    for (uint8_t temporary : {12, 124}) {
      SCOPED_TRACE(split);
      SCOPED_TRACE(static_cast<unsigned>(temporary));
      std::vector<uint32_t> words(144, build_s_nop(0, arch).value());
      words[64] = build_s_endpgm(arch).value();
      words[128] = build_s_endpgm(arch).value();
      std::vector<uint32_t> builder = {
          build_s_getpc_b64(8, arch).value(),
          cdna5::build_sop2(cdna5::kSAddCoI32Sop2,
                            {.ssrc0 = 255, .ssrc1 = 132, .sdst = temporary})[0],
          504,
      };
      if (split) {
        const std::vector<uint32_t> subtract = {
            cdna5::build_sopc(cdna5::kSCmpGeI32Sopc, {.ssrc0 = temporary, .ssrc1 = 128})[0],
            cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 4})[0],
            cdna5::build_sop1(cdna5::kSAbsI32Sop1, {.ssrc0 = temporary, .sdst = temporary})[0],
            cdna5::build_sop2(cdna5::kSSubCoU32Sop2,
                              {.ssrc0 = 8, .ssrc1 = temporary, .sdst = 8})[0],
            cdna5::build_sop2(cdna5::kSSubCoCiU32Sop2, {.ssrc0 = 9, .ssrc1 = 128, .sdst = 9})[0],
            build_s_setpc_b64(8, arch).value(),
        };
        builder.insert(builder.end(), subtract.begin(), subtract.end());
      }
      builder.push_back(
          cdna5::build_sop2(cdna5::kSAddCoU32Sop2, {.ssrc0 = 8, .ssrc1 = temporary, .sdst = 8})[0]);
      builder.push_back(
          cdna5::build_sop2(cdna5::kSAddCoCiU32Sop2, {.ssrc0 = 9, .ssrc1 = 128, .sdst = 9})[0]);
      builder.push_back(build_s_setpc_b64(8, arch).value());
      std::ranges::copy(builder, words.begin());
      const std::vector<uint8_t> image = two_owner_image(words, 256, EF_AMDGPU_MACH_AMDGCN_GFX1250);
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      const FailureOr<KernelOwnership> ownership =
          KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX1250, error);
      if (temporary == 12) {
        ASSERT_TRUE(ownership.succeeded()) << error;
        ASSERT_TRUE(ownership.value().owners_at(512));
        EXPECT_EQ(ownership.value().owners_at(512)->size(), 1u);
      } else {
        EXPECT_TRUE(ownership.failed());
      }
    }
  }
}

TEST(KernelOwnershipTest, RejectsDroppedTargetInAnotherExecutableSection) {
  std::vector<uint32_t> words(256, build_s_nop(0, kArch).value());
  words[0] = build_s_cbranch_scc0(7); // 0 -> 32.
  write_pc_builder(words, 4, 8, 512);
  words[5] = build_s_branch(10, kArch).value(); // 20 -> join at 64.
  write_pc_builder(words, 32, 8, 896);
  words[12] = build_s_branch(3, kArch).value(); // 48 -> join at 64.
  words[16] = build_s_setpc_b64(8, kArch).value();
  words[64] = build_s_branch(127, kArch).value(); // B -> shared store at 768.
  words[128] = build_s_endpgm(kArch).value();
  const std::array<uint32_t, 2> store =
      cdna4::build_ds(cdna4::kDsWriteB32Ds, {.addr = 2, .data0 = 3});
  words[192] = store[0];
  words[193] = store[1];
  words[194] = build_s_endpgm(kArch).value();
  words[224] = build_s_branch(-33, kArch).value(); // Auxiliary section: 896 -> 768.
  std::vector<uint8_t> image = two_owner_image(words, 256);

  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  Elf64_Shdr &names_header = sections[header.e_shstrndx];
  std::vector<uint8_t> names(image.begin() + names_header.sh_offset,
                             image.begin() + names_header.sh_offset + names_header.sh_size);
  Elf64_Shdr auxiliary = sections[1];
  auxiliary.sh_name = test::add_elf_name(names, ".aux");
  auxiliary.sh_offset += 896;
  auxiliary.sh_addr += 896;
  auxiliary.sh_size = sizeof(uint32_t);
  sections[1].sh_size = 800;
  names_header.sh_offset = image.size();
  names_header.sh_size = names.size();
  image.insert(image.end(), names.begin(), names.end());
  sections.push_back(auxiliary);
  header.e_shoff = test::align_up_for_test(image.size(), alignof(Elf64_Shdr));
  header.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(header.e_shoff + sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data(), &header, sizeof(header));

  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  ASSERT_TRUE(source->is_valid());
  ASSERT_EQ(source->text_sections().size(), 1u);
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  EXPECT_TRUE(ownership.failed());
  EXPECT_EQ(error, "incomplete indirect ownership");
}

TEST(KernelOwnershipTest, AbortTrapClosesOnlyItsOwnArm) {
  std::vector<uint32_t> words(65, build_s_nop(0, kArch).value());
  words[0] = build_s_cbranch_scc0(2); // 0 -> endpgm at 12.
  words[1] = build_s_trap(kArch, 2).value();
  words[2] = 0xffffffffu; // Dead after abort, never decoded.
  words[3] = build_s_endpgm(kArch).value();
  words[64] = build_s_endpgm(kArch).value();
  const std::vector<uint8_t> image = two_owner_image(words, 256);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  ASSERT_TRUE(ownership.succeeded()) << error;
  EXPECT_NE(ownership.value().instruction_at(0, 4), nullptr);
  EXPECT_EQ(ownership.value().instruction_at(0, 8), nullptr);
  EXPECT_NE(ownership.value().instruction_at(0, 12), nullptr);
}

TEST(KernelOwnershipTest, DebugTrapResumesWithoutProvingUnknownIndirectTargets) {
  for (bool unresolved : {false, true}) {
    SCOPED_TRACE(unresolved);
    std::vector<uint32_t> words(65, build_s_nop(0, kArch).value());
    words[0] = build_s_trap(kArch, 3).value();
    words[1] = unresolved ? build_s_setpc_b64(8, kArch).value() : build_s_endpgm(kArch).value();
    words[64] = build_s_endpgm(kArch).value();
    const std::vector<uint8_t> image = two_owner_image(words, 256);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
    if (unresolved) {
      EXPECT_TRUE(ownership.failed());
      EXPECT_EQ(error, "unresolved indirect ownership");
    } else {
      ASSERT_TRUE(ownership.succeeded()) << error;
      EXPECT_NE(ownership.value().instruction_at(0, 4), nullptr);
    }
  }
}

TEST(KernelOwnershipTest, DebugTrapDoesNotHideReturnLinkWrites) {
  for (bool clobber : {false, true}) {
    SCOPED_TRACE(clobber);
    std::vector<uint32_t> words = shared_helper_words(false, false);
    words[128] = build_s_trap(kArch, 3).value();
    words[129] = clobber ? build_s_mov_b32(30, 128, kArch).value() : build_s_nop(0, kArch).value();
    words.push_back(build_s_setpc_b64(30, kArch).value());
    const std::vector<uint8_t> image = two_owner_image(words, 256);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
    if (clobber) {
      EXPECT_TRUE(ownership.failed());
      EXPECT_EQ(error, "unresolved indirect ownership");
    } else {
      ASSERT_TRUE(ownership.succeeded()) << error;
      ASSERT_TRUE(ownership.value().owners_at(512));
      EXPECT_EQ(ownership.value().owners_at(512)->size(), 2u);
    }
  }
}

TEST(KernelOwnershipTest, RelativeScalarReturnLinkWritesAreIndependentOfPcRecovery) {
  for (bool relative_destination : {false, true}) {
    SCOPED_TRACE(relative_destination);
    std::vector<uint32_t> words = shared_helper_words(false, false);
    words[128] = build_s_mov_b32(scalar_operand_m0(kArch).value(), 158, kArch).value();
    const std::array<uint32_t, 1> relative =
        cdna4::build_sop1(relative_destination ? cdna4::kSMovreldB32Sop1 : cdna4::kSMovrelsB32Sop1,
                          {.ssrc0 = 1, .sdst = 0});
    words[129] = relative[0];
    words.push_back(build_s_setpc_b64(30, kArch).value());
    const std::vector<uint8_t> image = two_owner_image(words, 256);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
    if (relative_destination) {
      EXPECT_TRUE(ownership.failed());
      EXPECT_EQ(error, "unresolved indirect ownership");
    } else {
      ASSERT_TRUE(ownership.succeeded()) << error;
      ASSERT_TRUE(ownership.value().owners_at(512));
      EXPECT_EQ(ownership.value().owners_at(512)->size(), 2u);
    }
  }
}

TEST(KernelOwnershipTest, Gfx942ReturnLinksUseEachDescriptorAllocation) {
  constexpr IsaFamily arch = IsaFamily::Cdna3;
  struct Link {
    uint16_t pair;
    uint16_t special_destination;
  };
  const std::array<Link, 2> links = {{
      {.pair = 46, .special_destination = cdna3::OPR_SDST_VCC_LO},
      {.pair = 42, .special_destination = cdna3::OPR_SDST_FLAT_SCRATCH_LO},
  }};
  for (const Link &link : links) {
    for (uint32_t sgpr_count : {48, 64}) {
      SCOPED_TRACE(link.pair);
      SCOPED_TRACE(sgpr_count);
      std::vector<uint32_t> words(130, build_s_nop(0, arch).value());
      words[0] = build_s_call_b64(link.pair, 127, arch).value();
      words[1] = build_s_endpgm(arch).value();
      const std::array<uint32_t, 2> store =
          cdna3::build_ds(cdna3::kDsWriteB32Ds, {.addr = 2, .data0 = 3});
      words[2] = store[0];
      words[3] = store[1];
      words[4] = build_s_endpgm(arch).value();
      words[64] = build_s_branch(-63, arch).value();
      words[128] = build_s_add_u32(link.special_destination, link.pair, 132, arch).value();
      words[129] = build_s_setpc_b64(link.pair, arch).value();
      std::vector<uint8_t> image = two_owner_image(words, 256, EF_AMDGPU_MACH_AMDGCN_GFX942);
      for (size_t kernel_index = 0; kernel_index < 2; ++kernel_index) {
        const size_t descriptor_offset = 0x100 + words.size() * sizeof(uint32_t) +
                                         kernel_index * sizeof(amdhsa::kernel_descriptor_t);
        amdhsa::kernel_descriptor_t descriptor = {};
        std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
        // B has no call. Its smaller allocation must not determine A's aliases.
        const uint32_t allocation = kernel_index == 0 ? sgpr_count : 48;
        RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                           amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                           allocation / 8 - 1);
        std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
      }
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      const FailureOr<KernelOwnership> ownership =
          KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX942, error);
      if (sgpr_count == 48) {
        EXPECT_TRUE(ownership.failed());
      } else {
        ASSERT_TRUE(ownership.succeeded()) << error;
        const std::optional<std::span<const KernelExecutionOwner>> owners =
            ownership.value().owners_at(8);
        ASSERT_TRUE(owners);
        ASSERT_EQ(owners->size(), 1u);
        EXPECT_EQ(owners->front().kernel_index, 1u);
      }
    }
  }
}

TEST(KernelOwnershipTest, Gfx942RecoveredBuildersUseEachDescriptorAllocation) {
  constexpr IsaFamily arch = IsaFamily::Cdna3;
  for (bool named_special : {false, true}) {
    for (uint16_t pair : {42, 46}) {
      for (uint32_t sgpr_count : {48, 64}) {
        SCOPED_TRACE(named_special);
        SCOPED_TRACE(pair);
        SCOPED_TRACE(sgpr_count);
        const uint16_t special_selector =
            pair == 46 ? cdna3::OPR_SDST_VCC_LO : cdna3::OPR_SDST_FLAT_SCRATCH_LO;
        const uint16_t encoded_pair = named_special ? special_selector : pair;
        std::vector<uint32_t> words(132, build_s_nop(0, arch).value());
        words[0] = build_s_getpc_b64(encoded_pair, arch).value();
        words[1] = build_s_add_u32(encoded_pair, encoded_pair, 255, arch).value();
        words[2] = 508;
        words[3] = build_s_addc_u32(encoded_pair + 1, encoded_pair + 1, 128, arch).value();
        const uint16_t destination = named_special
                                         ? static_cast<uint16_t>(sgpr_count - (pair == 46 ? 2 : 6))
                                         : special_selector;
        words[4] = build_s_add_u32(destination, encoded_pair, 132, arch).value();
        words[5] = build_s_setpc_b64(encoded_pair, arch).value();
        words[64] = build_s_branch(64, arch).value();
        words[128] = build_s_endpgm(arch).value();
        const std::array<uint32_t, 2> store =
            cdna3::build_ds(cdna3::kDsWriteB32Ds, {.addr = 2, .data0 = 3});
        words[129] = store[0];
        words[130] = store[1];
        words[131] = build_s_endpgm(arch).value();
        std::vector<uint8_t> image = two_owner_image(words, 256, EF_AMDGPU_MACH_AMDGCN_GFX942);
        for (size_t kernel_index = 0; kernel_index < 2; ++kernel_index) {
          const size_t descriptor_offset = 0x100 + words.size() * sizeof(uint32_t) +
                                           kernel_index * sizeof(amdhsa::kernel_descriptor_t);
          amdhsa::kernel_descriptor_t descriptor = {};
          std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
          const uint32_t allocation = kernel_index == 0 ? sgpr_count : 48;
          RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                             amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                             allocation / 8 - 1);
          std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
        }
        const std::shared_ptr<const AmdGpuCodeObject> source =
            std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
        std::string error;
        const FailureOr<KernelOwnership> ownership =
            KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX942, error);
        if (named_special || sgpr_count == 48) {
          EXPECT_TRUE(ownership.failed());
        } else {
          ASSERT_TRUE(ownership.succeeded()) << error;
          const std::optional<std::span<const KernelExecutionOwner>> owners =
              ownership.value().owners_at(516);
          ASSERT_TRUE(owners);
          ASSERT_EQ(owners->size(), 1u);
          EXPECT_EQ(owners->front().kernel_index, 1u);
        }
      }
    }
  }
}

TEST(KernelOwnershipTest, Gfx950VccAliasesUseEachDescriptorAllocation) {
  for (bool return_link : {false, true}) {
    for (uint32_t sgpr_count : {48, 64}) {
      SCOPED_TRACE(return_link);
      SCOPED_TRACE(sgpr_count);
      const uint32_t store_offset = return_link ? 8 : 516;
      std::vector<uint32_t> words(132, build_s_nop(0, kArch).value());
      if (return_link) {
        words[0] = build_s_call_b64(46, 127, kArch).value();
        words[1] = build_s_endpgm(kArch).value();
      } else {
        write_pc_builder(words, 0, 46, 512);
        words[128] = build_s_endpgm(kArch).value();
      }
      const size_t clobber_index = return_link ? 128 : 4;
      words[clobber_index] =
          build_s_add_u32(scalar_operand_vcc_lo(kArch).value(), 46, 132, kArch).value();
      words[clobber_index + 1] = build_s_setpc_b64(46, kArch).value();
      words[64] = build_s_branch(
                      static_cast<int16_t>((static_cast<int32_t>(store_offset) - 260) / 4), kArch)
                      .value();
      const std::array<uint32_t, 2> store =
          cdna4::build_ds(cdna4::kDsWriteB32Ds, {.addr = 2, .data0 = 3});
      words[store_offset / 4] = store[0];
      words[store_offset / 4 + 1] = store[1];
      words[store_offset / 4 + 2] = build_s_endpgm(kArch).value();
      std::vector<uint8_t> image = two_owner_image(words, 256);
      for (size_t kernel_index = 0; kernel_index < 2; ++kernel_index) {
        const size_t descriptor_offset = 0x100 + words.size() * sizeof(uint32_t) +
                                         kernel_index * sizeof(amdhsa::kernel_descriptor_t);
        amdhsa::kernel_descriptor_t descriptor = {};
        std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
        const uint32_t allocation = kernel_index == 0 ? sgpr_count : 48;
        RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                           amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                           allocation / 8 - 1);
        std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
      }
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      const FailureOr<KernelOwnership> ownership =
          KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
      if (sgpr_count == 48) {
        EXPECT_TRUE(ownership.failed());
      } else {
        ASSERT_TRUE(ownership.succeeded()) << error;
        const std::optional<std::span<const KernelExecutionOwner>> owners =
            ownership.value().owners_at(store_offset);
        ASSERT_TRUE(owners);
        ASSERT_EQ(owners->size(), 1u);
        EXPECT_EQ(owners->front().kernel_index, 1u);
      }
    }
  }
}

struct LegacyCdnaTarget {
  IsaFamily arch;
  rj_code_target_id_t target;
  uint32_t flags;
};

constexpr std::array<LegacyCdnaTarget, 2> kLegacyCdnaTargets = {{
    {.arch = IsaFamily::Cdna1,
     .target = ROCJITSU_CODE_TARGET_GFX908,
     .flags = EF_AMDGPU_MACH_AMDGCN_GFX908},
    {.arch = IsaFamily::Cdna2,
     .target = ROCJITSU_CODE_TARGET_GFX90A,
     .flags = EF_AMDGPU_MACH_AMDGCN_GFX90A},
}};

TEST(KernelOwnershipTest, LegacyVccAliasesUseEachDescriptorAllocation) {
  for (const LegacyCdnaTarget &target : kLegacyCdnaTargets) {
    for (bool return_link : {false, true}) {
      for (uint32_t sgpr_count : {48, 64}) {
        for (bool canonical_field : {false, true}) {
          SCOPED_TRACE(static_cast<int>(target.arch));
          SCOPED_TRACE(return_link);
          SCOPED_TRACE(sgpr_count);
          SCOPED_TRACE(canonical_field);
          const uint32_t store_offset = return_link ? 8 : 516;
          std::vector<uint32_t> words(132, build_s_nop(0, target.arch).value());
          if (return_link) {
            words[0] = build_s_call_b64(46, 127, target.arch).value();
            words[1] = build_s_endpgm(target.arch).value();
          } else {
            words[0] = build_s_getpc_b64(46, target.arch).value();
            words[1] = build_s_add_u32(46, 46, 255, target.arch).value();
            words[2] = 508;
            words[3] = build_s_addc_u32(47, 47, 128, target.arch).value();
            words[128] = build_s_endpgm(target.arch).value();
          }
          const size_t clobber = return_link ? 128 : 4;
          words[clobber] =
              build_s_add_u32(scalar_operand_vcc_lo(target.arch).value(), 46, 132, target.arch)
                  .value();
          words[clobber + 1] = build_s_setpc_b64(46, target.arch).value();
          words[64] =
              build_s_branch(static_cast<int16_t>((static_cast<int32_t>(store_offset) - 260) / 4),
                             target.arch)
                  .value();
          const std::array<uint32_t, 2> store =
              target.arch == IsaFamily::Cdna1
                  ? cdna1::build_ds(cdna1::kDsWriteB32Ds, {.addr = 2, .data0 = 3})
                  : cdna2::build_ds(cdna2::kDsWriteB32Ds, {.addr = 2, .data0 = 3});
          words[store_offset / 4] = store[0];
          words[store_offset / 4 + 1] = store[1];
          words[store_offset / 4 + 2] = build_s_endpgm(target.arch).value();
          std::vector<uint8_t> image = two_owner_image(words, 256, target.flags);
          for (size_t kernel = 0; kernel < 2; ++kernel) {
            const size_t offset = 0x100 + words.size() * sizeof(uint32_t) +
                                  kernel * sizeof(amdhsa::kernel_descriptor_t);
            amdhsa::kernel_descriptor_t descriptor = {};
            std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
            const uint32_t allocation = kernel == 0 ? sgpr_count : 48;
            // Both fields round to the same physical 16-SGPR block. The even
            // field is the canonical GFX9 encoding; retain the original fixture too.
            RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                               amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                               allocation / 8 - (canonical_field ? 2 : 1));
            std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
          }
          const std::shared_ptr<const AmdGpuCodeObject> source =
              std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
          std::string error;
          const FailureOr<KernelOwnership> ownership =
              KernelOwnership::build(source, target.target, error);
          if (sgpr_count == 48) {
            EXPECT_TRUE(ownership.failed());
          } else {
            ASSERT_TRUE(ownership.succeeded()) << error;
            const std::optional<std::span<const KernelExecutionOwner>> owners =
                ownership.value().owners_at(store_offset);
            ASSERT_TRUE(owners);
            ASSERT_EQ(owners->size(), 1u);
            EXPECT_EQ(owners->front().kernel_index, 1u);
          }
        }
      }
    }
  }
}

TEST(KernelOwnershipTest, LegacyFlatScratchWritesInvalidateSavedLinks) {
  for (const LegacyCdnaTarget &target : kLegacyCdnaTargets) {
    for (bool before_call : {false, true}) {
      SCOPED_TRACE(static_cast<int>(target.arch));
      SCOPED_TRACE(before_call);
      std::vector<uint32_t> words = shared_helper_words(false, false, target.arch);
      const uint32_t flat_write =
          build_s_mov_b32(amdgpu::kFlatScratchSelectorFirst, 128, target.arch).value();
      if (before_call) {
        words[0] = flat_write;
        words[1] = build_s_call_b64(30, 126, target.arch).value();
        words[2] = build_s_endpgm(target.arch).value();
      } else {
        words[128] = flat_write;
      }
      const std::vector<uint8_t> image = two_owner_image(words, 256, target.flags);
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      const FailureOr<KernelOwnership> ownership =
          KernelOwnership::build(source, target.target, error);
      if (before_call) {
        ASSERT_TRUE(ownership.succeeded()) << error;
        ASSERT_TRUE(ownership.value().owners_at(512));
        EXPECT_EQ(ownership.value().owners_at(512)->size(), 2u);
      } else {
        EXPECT_TRUE(ownership.failed());
      }
    }
  }
}

TEST(KernelOwnershipTest, LegacyFlatScratchWritesRequireRecoveredPcBackingProof) {
  for (const LegacyCdnaTarget &target : kLegacyCdnaTargets) {
    SCOPED_TRACE(static_cast<int>(target.arch));
    std::vector<uint32_t> words(129, build_s_nop(0, target.arch).value());
    words[0] = build_s_mov_b32(amdgpu::kFlatScratchSelectorFirst, 128, target.arch).value();
    words[1] = build_s_getpc_b64(8, target.arch).value();
    words[2] = build_s_add_u32(8, 8, 255, target.arch).value();
    words[3] = 504;
    words[4] = build_s_addc_u32(9, 9, 128, target.arch).value();
    words[5] = build_s_setpc_b64(8, target.arch).value();
    words[64] = build_s_branch(63, target.arch).value();
    words[128] = build_s_endpgm(target.arch).value();
    const std::vector<uint8_t> image = two_owner_image(words, 256, target.flags);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    EXPECT_TRUE(KernelOwnership::build(source, target.target, error).failed());
  }
}

TEST(KernelOwnershipTest, LegacyVccDefinitionsInvalidateSavedLinks) {
  for (const LegacyCdnaTarget &target : kLegacyCdnaTargets) {
    for (bool vector_compare : {false, true}) {
      SCOPED_TRACE(static_cast<int>(target.arch));
      SCOPED_TRACE(vector_compare);
      std::vector<uint32_t> words = shared_helper_words(false, false, target.arch);
      if (vector_compare) {
        words[128] = target.arch == IsaFamily::Cdna1
                         ? cdna1::build_vopc(cdna1::kVCmpEqU32Vopc, {.src0 = 128, .vsrc1 = 0})[0]
                         : cdna2::build_vopc(cdna2::kVCmpEqU32Vopc, {.src0 = 128, .vsrc1 = 0})[0];
      } else {
        words[128] =
            build_s_mov_b32(scalar_operand_vcc_lo(target.arch).value() + 1, 129, target.arch)
                .value();
      }
      const std::vector<uint8_t> image = two_owner_image(words, 256, target.flags);
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      EXPECT_TRUE(KernelOwnership::build(source, target.target, error).failed());
    }
  }
}

TEST(KernelOwnershipTest, AliasedReturnLinksRemainProvedWithoutSpecialWrites) {
  struct Target {
    IsaFamily arch;
    rj_code_target_id_t target;
    uint32_t flags;
  };
  const std::array<Target, 4> targets = {{
      {.arch = IsaFamily::Cdna3,
       .target = ROCJITSU_CODE_TARGET_GFX942,
       .flags = EF_AMDGPU_MACH_AMDGCN_GFX942},
      {.arch = IsaFamily::Cdna4,
       .target = ROCJITSU_CODE_TARGET_GFX950,
       .flags = EF_AMDGPU_MACH_AMDGCN_GFX950},
      {.arch = IsaFamily::Cdna1,
       .target = ROCJITSU_CODE_TARGET_GFX908,
       .flags = EF_AMDGPU_MACH_AMDGCN_GFX908},
      {.arch = IsaFamily::Cdna2,
       .target = ROCJITSU_CODE_TARGET_GFX90A,
       .flags = EF_AMDGPU_MACH_AMDGCN_GFX90A},
  }};
  for (const Target &target : targets) {
    SCOPED_TRACE(static_cast<int>(target.arch));
    const std::vector<uint8_t> image =
        two_owner_image(shared_helper_words(false, false, target.arch), 256, target.flags);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, target.target, error);
    ASSERT_TRUE(ownership.succeeded()) << error;
    const std::optional<std::span<const KernelExecutionOwner>> owners =
        ownership.value().owners_at(512);
    ASSERT_TRUE(owners);
    EXPECT_EQ(owners->size(), 2u);
  }
}

TEST(KernelOwnershipTest, VccWritesInvalidateAliasedReturnLinks) {
  for (bool use_cdna4 : {false, true}) {
    const IsaFamily arch = use_cdna4 ? IsaFamily::Cdna4 : IsaFamily::Cdna3;
    const rj_code_target_id_t target =
        use_cdna4 ? ROCJITSU_CODE_TARGET_GFX950 : ROCJITSU_CODE_TARGET_GFX942;
    const uint32_t flags = use_cdna4 ? EF_AMDGPU_MACH_AMDGCN_GFX950 : EF_AMDGPU_MACH_AMDGCN_GFX942;
    for (bool vector_compare : {false, true}) {
      SCOPED_TRACE(use_cdna4);
      SCOPED_TRACE(vector_compare);
      std::vector<uint32_t> words = shared_helper_words(false, false, arch);
      if (vector_compare) {
        words[128] = use_cdna4
                         ? cdna4::build_vopc(cdna4::kVCmpEqU32Vopc, {.src0 = 128, .vsrc1 = 0})[0]
                         : cdna3::build_vopc(cdna3::kVCmpEqU32Vopc, {.src0 = 128, .vsrc1 = 0})[0];
      } else {
        words[128] = build_s_mov_b32(scalar_operand_vcc_lo(arch).value() + 1, 129, arch).value();
      }
      const std::vector<uint8_t> image = two_owner_image(words, 256, flags);
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      EXPECT_TRUE(KernelOwnership::build(source, target, error).failed());
    }
  }
}

TEST(KernelOwnershipTest, SharedHelpersUseEachConcreteTargetAndWave) {
  struct Target {
    rocjitsu::IsaFamily arch;
    rj_code_target_id_t target;
    uint32_t flags;
    uint8_t wave;
  };
  const std::array<Target, 2> targets = {{
      {.arch = rocjitsu::IsaFamily::Rdna4,
       .target = ROCJITSU_CODE_TARGET_GFX1201,
       .flags = EF_AMDGPU_MACH_AMDGCN_GFX1201,
       .wave = 64},
      {.arch = rocjitsu::IsaFamily::Cdna5,
       .target = ROCJITSU_CODE_TARGET_GFX1250,
       .flags = EF_AMDGPU_MACH_AMDGCN_GFX1250,
       .wave = 32},
  }};
  for (const Target &target : targets) {
    for (bool indirect : {false, true}) {
      const std::vector<uint8_t> image =
          two_owner_image(shared_helper_words(indirect, false, target.arch), 256, target.flags);
      const std::shared_ptr<const AmdGpuCodeObject> source =
          std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
      std::string error;
      const FailureOr<KernelOwnership> ownership =
          KernelOwnership::build(source, target.target, error);
      ASSERT_TRUE(ownership.succeeded()) << error;
      const std::optional<std::span<const KernelExecutionOwner>> owners =
          ownership.value().owners_at(512);
      ASSERT_TRUE(owners);
      ASSERT_EQ(owners->size(), 2u);
      EXPECT_EQ((*owners)[0].proof,
                indirect ? KernelOwnerProof::RecoveredIndirectCall : KernelOwnerProof::DirectCall);
      EXPECT_EQ((*owners)[1].proof, (*owners)[0].proof);
      const std::optional<KernelOwnerFlow> flow = ownership.value().owner_flow(0);
      ASSERT_TRUE(flow);
      EXPECT_EQ(flow->wave_size, target.wave);
      EXPECT_EQ(flow->target, target.target);
    }
  }
}

TEST(KernelOwnershipTest, RejectsUnresolvedFlowAndCrossKernelTransfers) {
  for (bool cross_kernel : {false, true}) {
    SCOPED_TRACE(cross_kernel);
    std::vector<uint32_t> words = shared_helper_words(false, false);
    words[0] =
        cross_kernel ? build_s_branch(63, kArch).value() : build_s_setpc_b64(8, kArch).value();
    const std::vector<uint8_t> image = two_owner_image(words, 256);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
    EXPECT_TRUE(ownership.failed());
    EXPECT_EQ(error, cross_kernel ? "control flow crosses another kernel entry"
                                  : "unresolved indirect ownership");
  }
}

TEST(KernelOwnershipTest, RejectsClobberedReturnAndAliasedRoots) {
  for (bool indirect : {false, true}) {
    const std::vector<uint8_t> image = two_owner_image(shared_helper_words(indirect, true), 256);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    EXPECT_TRUE(KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error).failed());
    std::vector<uint32_t> other_half_words = shared_helper_words(indirect, true);
    other_half_words[128] = build_s_mov_b32(31, 0, kArch).value();
    const std::vector<uint8_t> other_half_image = two_owner_image(other_half_words, 256);
    const std::shared_ptr<const AmdGpuCodeObject> other_half =
        std::make_shared<AmdGpuCodeObject>(other_half_image.data(), other_half_image.size());
    EXPECT_TRUE(KernelOwnership::build(other_half, ROCJITSU_CODE_TARGET_GFX950, error).failed());
  }
  const std::vector<uint8_t> image = two_owner_image(shared_helper_words(false, false), 0);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  EXPECT_TRUE(KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error).failed());
}

TEST(KernelOwnershipTest, PreloadEntryDoesNotInheritAnotherRootsReturnLink) {
  const uint32_t nop = build_s_nop(0, kArch).value();
  std::vector<uint32_t> words(128, nop);
  words[0] = build_s_call_b64(30, 127, kArch).value();
  words[1] = build_s_endpgm(kArch).value();
  words[64] = build_s_branch(63, kArch).value();
  words.push_back(build_s_setpc_b64(30, kArch).value());
  std::vector<uint8_t> image = test::make_gfx950_kernel_elf(words, 0, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const size_t descriptor_offset = 0x100 + words.size() * sizeof(uint32_t);
  amdhsa::kernel_descriptor_t descriptor = {};
  std::memcpy(&descriptor, image.data() + descriptor_offset, sizeof(descriptor));
  RJ_AMDHSA_BITS_SET(descriptor.kernarg_preload, amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  std::memcpy(image.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  EXPECT_TRUE(KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error).failed());
}

TEST(KernelOwnershipTest, RejectsMalformedSiblingAndDifferentSourceIdentity) {
  const std::vector<uint32_t> words = shared_helper_words(false, false);
  std::vector<uint8_t> image = two_owner_image(words, 256);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  const std::shared_ptr<const AmdGpuCodeObject> identical_but_distinct =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  ASSERT_TRUE(ownership.succeeded()) << error;
  EXPECT_TRUE(ownership.value().belongs_to(*source));
  EXPECT_FALSE(ownership.value().belongs_to(*identical_but_distinct));
  const size_t offset =
      0x100 + words.size() * sizeof(uint32_t) + sizeof(amdhsa::kernel_descriptor_t);
  amdhsa::kernel_descriptor_t second = {};
  std::memcpy(&second, image.data() + offset, sizeof(second));
  second.kernel_code_entry_byte_offset = std::numeric_limits<int64_t>::max();
  std::memcpy(image.data() + offset, &second, sizeof(second));
  const std::shared_ptr<const AmdGpuCodeObject> malformed =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  EXPECT_TRUE(KernelOwnership::build(malformed, ROCJITSU_CODE_TARGET_GFX950, error).failed());
  EXPECT_EQ(error, "descriptor was omitted from owner inventory");
}

TEST(KernelOwnershipTest, DeclaredHelperMustBeOwnedAndDynamicExportsAreRejected) {
  for (bool dynamic : {false, true}) {
    std::vector<uint8_t> image = two_owner_image(shared_helper_words(false, false), 256);
    add_function_symbol(image, 512, dynamic);
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership =
        KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
    if (dynamic) {
      EXPECT_TRUE(ownership.failed());
      EXPECT_EQ(error, "external dynamic function needs additional owner proof");
    } else {
      ASSERT_TRUE(ownership.succeeded()) << error;
      ASSERT_TRUE(ownership.value().owners_at(512));
      EXPECT_EQ(ownership.value().owners_at(512)->size(), 2u);
    }
  }
  std::vector<uint8_t> image = two_owner_image(shared_helper_words(false, false), 256);
  add_function_symbol(image, 128, false);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  EXPECT_TRUE(KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error).failed());
  EXPECT_EQ(error, "declared function has unresolved ownership");
  std::vector<uint8_t> notype = two_owner_image(shared_helper_words(false, false), 256);
  add_function_symbol(notype, 512, false, kElfSymbolTypeNone);
  const std::shared_ptr<const AmdGpuCodeObject> external_notype =
      std::make_shared<AmdGpuCodeObject>(notype.data(), notype.size());
  EXPECT_TRUE(KernelOwnership::build(external_notype, ROCJITSU_CODE_TARGET_GFX950, error).failed());
  EXPECT_EQ(error, "external executable symbol needs additional owner proof");
  std::vector<uint8_t> object = two_owner_image(shared_helper_words(false, false), 256);
  add_function_symbol(object, 512, false, kElfSymbolTypeObject);
  const std::shared_ptr<const AmdGpuCodeObject> executable_object =
      std::make_shared<AmdGpuCodeObject>(object.data(), object.size());
  EXPECT_TRUE(
      KernelOwnership::build(executable_object, ROCJITSU_CODE_TARGET_GFX950, error).failed());
  EXPECT_EQ(error, "unsupported executable object or kernel symbol");
}

TEST(KernelOwnershipTest, RejectsUnresolvedDynamicRelocations) {
  std::vector<uint8_t> image = two_owner_image(shared_helper_words(false, false), 256);
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  Elf64_Shdr relocation = {};
  relocation.sh_type = SHT_RELA;
  relocation.sh_entsize = sizeof(Elf64_Rela);
  sections.push_back(relocation);
  header.e_shoff = image.size();
  header.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(image.size() + sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data(), &header, sizeof(header));
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  EXPECT_TRUE(KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error).failed());
  EXPECT_EQ(error, "dynamic or malformed relocations require additional ownership proof");
}

TEST(KernelOwnershipTest, PreloadRootsFollowConcreteArchitecturePolicy) {
  for (bool cdna5 : {false, true}) {
    const rocjitsu::IsaFamily arch =
        cdna5 ? rocjitsu::IsaFamily::Cdna5 : rocjitsu::IsaFamily::Cdna4;
    const rj_code_target_id_t target =
        cdna5 ? ROCJITSU_CODE_TARGET_GFX1250 : ROCJITSU_CODE_TARGET_GFX950;
    const uint32_t flags = cdna5 ? EF_AMDGPU_MACH_AMDGCN_GFX1250 : EF_AMDGPU_MACH_AMDGCN_GFX950;
    const uint32_t nop = build_s_nop(0, arch).value();
    const uint32_t end = build_s_endpgm(arch).value();
    std::vector<uint32_t> words(65, nop);
    words[0] = end;
    words[64] = end;
    std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, flags);
    image[EI_DATA] = ELFDATA2LSB;
    image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    const size_t offset = 0x100 + words.size() * sizeof(uint32_t);
    amdhsa::kernel_descriptor_t descriptor = {};
    std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
    RJ_AMDHSA_BITS_SET(descriptor.kernarg_preload, amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
    std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
    const std::shared_ptr<const AmdGpuCodeObject> source =
        std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
    std::string error;
    const FailureOr<KernelOwnership> ownership = KernelOwnership::build(source, target, error);
    ASSERT_TRUE(ownership.succeeded()) << error;
    ASSERT_TRUE(ownership.value().owners_at(0));
    EXPECT_EQ(ownership.value().owners_at(0)->size(), 1u);
    const std::optional<KernelOwnerFlow> flow = ownership.value().owner_flow(0);
    ASSERT_TRUE(flow);
    EXPECT_EQ(flow->roots.size(), cdna5 ? 1u : 2u);
    EXPECT_EQ(flow->wave_size, cdna5 ? 32u : 64u);
    if (cdna5) {
      EXPECT_FALSE(ownership.value().owners_at(256));
    } else {
      ASSERT_TRUE(ownership.value().owners_at(256));
      EXPECT_EQ(ownership.value().owners_at(256)->size(), 1u);
      EXPECT_EQ(ownership.value().owners_at(256)->front().proof, KernelOwnerProof::KernelLocal);
    }
  }
}

TEST(KernelOwnershipTest, RejectsPreloadEntryInsideInstruction) {
  std::vector<uint32_t> words(65, build_s_nop(0, kArch).value());
  words[0] = build_s_branch(62, kArch).value();
  words[63] = build_s_mov_b32(0, 255, kArch).value();
  words[64] = build_s_endpgm(kArch).value();
  std::vector<uint8_t> image = test::make_gfx950_kernel_elf(words, 0, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const size_t offset = 0x100 + words.size() * sizeof(uint32_t);
  amdhsa::kernel_descriptor_t descriptor = {};
  std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
  RJ_AMDHSA_BITS_SET(descriptor.kernarg_preload, amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  EXPECT_TRUE(KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error).failed());
}

TEST(KernelOwnershipTest, ExpiredAndNullSourcesDoNotPublishFacts) {
  const std::vector<uint8_t> image = two_owner_image(shared_helper_words(false, false), 256);
  alignas(AmdGpuCodeObject) unsigned char storage[sizeof(AmdGpuCodeObject)];
  const auto destroy_source = [](const AmdGpuCodeObject *source) { source->~AmdGpuCodeObject(); };
  std::shared_ptr<const AmdGpuCodeObject> source(
      new (storage) AmdGpuCodeObject(image.data(), image.size()), destroy_source);
  const uintptr_t address = reinterpret_cast<uintptr_t>(source.get());
  std::string error;
  EXPECT_TRUE(KernelOwnership::build({}, ROCJITSU_CODE_TARGET_GFX950, error).failed());
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX950, error);
  ASSERT_TRUE(ownership.succeeded()) << error;
  ASSERT_TRUE(ownership.value().owners_at(512));
  ASSERT_TRUE(ownership.value().kernels());
  std::weak_ptr<const AmdGpuCodeObject> weak = source;
  source.reset();
  ASSERT_TRUE(weak.expired());
  EXPECT_FALSE(ownership.value().owners_at(512));
  EXPECT_FALSE(ownership.value().kernels());
  EXPECT_FALSE(ownership.value().owner_flow(0));
  EXPECT_EQ(ownership.value().instruction_at(0, 512), nullptr);
  const std::shared_ptr<const AmdGpuCodeObject> replacement(
      new (storage) AmdGpuCodeObject(image.data(), image.size()), destroy_source);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(replacement.get()), address);
  EXPECT_FALSE(ownership.value().belongs_to(*replacement));
  EXPECT_FALSE(ownership.value().owners_at(512));
}

TEST(KernelOwnershipTest, SharedGfx1250HelperBankUsesEachOwnersProvedFlow) {
  constexpr rocjitsu::IsaFamily arch = rocjitsu::IsaFamily::Cdna5;
  const uint32_t nop = build_s_nop(0, arch).value();
  const uint32_t end = build_s_endpgm(arch).value();
  const std::array<uint32_t, 1> set_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp);
  const std::array<uint32_t, 1> set_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  const std::array<uint32_t, 2> access =
      cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2, .data0 = 3});
  std::vector<uint32_t> words(128, nop);
  words[0] = set_zero[0];
  words[1] = build_s_call_b64(30, 126, arch).value();
  words[2] = end;
  words[64] = set_one[0];
  words[65] = build_s_call_b64(30, 62, arch).value();
  words[66] = end;
  words.push_back(access[0]);
  words.push_back(access[1]);
  words.push_back(build_s_setpc_b64(30, arch).value());
  const std::vector<uint8_t> image = two_owner_image(words, 256, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX1250, error);
  ASSERT_TRUE(ownership.succeeded()) << error;
  for (size_t kernel = 0; kernel < 2; ++kernel) {
    const std::optional<KernelOwnerFlow> flow = ownership.value().owner_flow(kernel);
    ASSERT_TRUE(flow);
    EXPECT_EQ(flow->target, ROCJITSU_CODE_TARGET_GFX1250);
    EXPECT_EQ(flow->wave_size, 32u);
    ASSERT_EQ(flow->roots.size(), 1u);
    const Instruction *instruction = ownership.value().instruction_at(kernel, 512);
    ASSERT_NE(instruction, nullptr);
    ASSERT_NE(instruction->src_operand(0), nullptr);
    const IsaTargetBinding *target = default_isa_target_registry().find_target(flow->target);
    ASSERT_NE(target, nullptr);
    const Gfx1250VgprMsbAnalysis banks(flow->blocks, flow->roots.front(), flow->call_return_edges,
                                       flow->text, flow->roots.subspan(1),
                                       target->capabilities.setreg_vgpr_msb_fixup);
    EXPECT_EQ(banks.bank_before(*instruction, instruction->src_operand(0)->vgpr_msb_role()),
              kernel);
  }
  EXPECT_FALSE(ownership.value().owner_flow(2));
  EXPECT_EQ(ownership.value().instruction_at(0, 256), nullptr);
}

TEST(KernelOwnershipTest, MultipleCallSitesRetainEachProvedReturnAndJoinBank) {
  constexpr rocjitsu::IsaFamily arch = rocjitsu::IsaFamily::Cdna5;
  const uint32_t nop = build_s_nop(0, arch).value();
  const uint32_t end = build_s_endpgm(arch).value();
  const std::array<uint32_t, 1> branch = cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 3});
  const std::array<uint32_t, 1> set_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp);
  const std::array<uint32_t, 1> set_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  const std::array<uint32_t, 2> access =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 2, .vdst = 5});
  std::vector<uint32_t> words(128, nop);
  words[0] = branch[0];
  words[1] = set_zero[0];
  words[2] = build_s_call_b64(30, 125, arch).value();
  words[3] = end;
  words[4] = set_one[0];
  words[5] = build_s_call_b64(30, 122, arch).value();
  words[6] = end;
  words[64] = end;
  words.push_back(access[0]);
  words.push_back(access[1]);
  words.push_back(build_s_setpc_b64(30, arch).value());
  const std::vector<uint8_t> image = two_owner_image(words, 256, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX1250, error);
  ASSERT_TRUE(ownership.succeeded()) << error;
  const std::optional<KernelOwnerFlow> flow = ownership.value().owner_flow(0);
  ASSERT_TRUE(flow);
  ASSERT_EQ(flow->roots.size(), 1u);
  const auto has_edge = [&](uint64_t from, uint64_t to) {
    return std::ranges::any_of(flow->call_return_edges, [&](const ScopedCfgEdge &edge) {
      return edge.from->start_offset() == from && edge.to->start_offset() == to;
    });
  };
  EXPECT_TRUE(has_edge(4, 512));
  EXPECT_TRUE(has_edge(16, 512));
  EXPECT_TRUE(has_edge(512, 12));
  EXPECT_TRUE(has_edge(512, 24));
  const Instruction *instruction = ownership.value().instruction_at(0, 512);
  ASSERT_NE(instruction, nullptr);
  const IsaTargetBinding *target = default_isa_target_registry().find_target(flow->target);
  ASSERT_NE(target, nullptr);
  const Gfx1250VgprMsbAnalysis banks(flow->blocks, flow->roots.front(), flow->call_return_edges,
                                     flow->text, flow->roots.subspan(1),
                                     target->capabilities.setreg_vgpr_msb_fixup);
  EXPECT_EQ(banks.bank_before(*instruction, instruction->src_operand(0)->vgpr_msb_role()),
            std::nullopt);
}

TEST(KernelOwnershipTest, CalleeBankWriteMakesContinuationBankConservativelyUnknown) {
  constexpr rocjitsu::IsaFamily arch = rocjitsu::IsaFamily::Cdna5;
  const uint32_t nop = build_s_nop(0, arch).value();
  const uint32_t end = build_s_endpgm(arch).value();
  const std::array<uint32_t, 1> set_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp);
  const std::array<uint32_t, 1> set_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  const std::array<uint32_t, 2> access =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 2, .vdst = 5});
  std::vector<uint32_t> words(128, nop);
  words[0] = set_zero[0];
  words[1] = build_s_call_b64(30, 126, arch).value();
  words[2] = access[0];
  words[3] = access[1];
  words[4] = end;
  words[64] = end;
  words.push_back(set_one[0]);
  words.push_back(build_s_setpc_b64(30, arch).value());
  const std::vector<uint8_t> image = two_owner_image(words, 256, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const std::shared_ptr<const AmdGpuCodeObject> source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(source, ROCJITSU_CODE_TARGET_GFX1250, error);
  ASSERT_TRUE(ownership.succeeded()) << error;
  const std::optional<KernelOwnerFlow> flow = ownership.value().owner_flow(0);
  ASSERT_TRUE(flow);
  const Instruction *instruction = ownership.value().instruction_at(0, 8);
  ASSERT_NE(instruction, nullptr);
  const IsaTargetBinding *target = default_isa_target_registry().find_target(flow->target);
  ASSERT_NE(target, nullptr);
  const Gfx1250VgprMsbAnalysis banks(flow->blocks, flow->roots.front(), flow->call_return_edges,
                                     flow->text, flow->roots.subspan(1),
                                     target->capabilities.setreg_vgpr_msb_fixup);
  EXPECT_EQ(banks.bank_before(*instruction, instruction->src_operand(0)->vgpr_msb_role()),
            std::nullopt);
}

TEST_F(KernelScopeTest, ScopeStopsAtAnotherKernelEntry) {
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryA},
      .address_taken_entries = {},
  };
  // A branches straight into B's entry; that edge must not pull B's body in.
  EXPECT_EQ(start_offsets(scope_from(kEntryA, spec)),
            (std::vector<uint64_t>{kEntryA, kMidA, kTailA}));
}

TEST_F(KernelScopeTest, EachKernelGetsOnlyItsOwnBlocks) {
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryB},
      .address_taken_entries = {},
  };
  // B is entered from A, so it is not predecessorless, yet its scope is still
  // just its own body -- the walk starts at the entry, it does not run backward.
  EXPECT_EQ(start_offsets(scope_from(kEntryB, spec)),
            (std::vector<uint64_t>{kEntryB, kMidB, kTailB}));
}

TEST_F(KernelScopeTest, OwnEntriesOverrideTheKernelEntryStop) {
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryA, kEntryB},
      .address_taken_entries = {},
  };
  // This is the shape a kernarg-preload firmware entry or an adopted root takes:
  // an entry that belongs to this scope despite being in kernel_entries.
  EXPECT_EQ(start_offsets(scope_from(kEntryA, spec)),
            (std::vector<uint64_t>{kEntryA, kMidA, kTailA, kEntryB, kMidB, kTailB}));
}

TEST_F(KernelScopeTest, ScopeIsOrderedBySourceOffset) {
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryA, kEntryB},
      .address_taken_entries = {},
  };
  // Emission relies on source order so fallthrough is preserved.
  const std::vector<uint64_t> offsets = start_offsets(scope_from(kEntryA, spec));
  EXPECT_TRUE(std::ranges::is_sorted(offsets));
}

TEST_F(KernelScopeTest, UnreachedEntryYieldsOnlyItself) {
  // An own_entry with no decoded block is ignored rather than faulting.
  const KernelScopeSpec spec{
      .kernel_entries = {kEntryA, kEntryB},
      .own_entries = {kEntryB, 0x1000},
      .address_taken_entries = {},
  };
  EXPECT_EQ(start_offsets(scope_from(kEntryB, spec)),
            (std::vector<uint64_t>{kEntryB, kMidB, kTailB}));
}

TEST_F(KernelScopeTest, BlockForOffsetResolvesInteriorOffsets) {
  // A tail spans [0x08, 0x10): the s_endpgm at 0x0c is interior to it.
  const BasicBlock *tail = block_for_offset(offset_index_, kTailA);
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->start_offset(), kTailA);
  EXPECT_EQ(block_for_offset(offset_index_, kTailA + 4), tail);

  EXPECT_EQ(block_for_offset(offset_index_, 0x1000), nullptr);
}

TEST(BasicBlockWindow, CfgRejectsTruncatedInstructionWithSourceOffset) {
  TestCodeObject object({0xbf800000, 0xbe8000ff}); // s_nop; s_mov_b32 with no literal.
  auto decoder = Decoder::create(kArch);
  ASSERT_NE(decoder, nullptr);
  util::StringDiagnostic diagnostic;
  auto result = BasicBlock::build(object, *decoder, kArch, diagnostic.emitter());
  EXPECT_TRUE(result.failed());
  EXPECT_EQ(diagnostic.message(), "truncated instruction encoding at .text byte offset 4");
}

TEST(BasicBlockWindow, CfgPreservesFourWordInstructionAtSectionEnd) {
  TestCodeObject object({0xcc3a0200, 0x42020d04, 0xcc336008, 0x04223110});
  auto decoder = Decoder::create(rocjitsu::IsaFamily::Cdna5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(object, *decoder, rocjitsu::IsaFamily::Cdna5);
  ASSERT_EQ(blocks.size(), 1u);
  const Instruction &instruction = *blocks.front()->instructions().begin();
  EXPECT_EQ(instruction.size(), 16);
  ASSERT_NE(instruction.raw_encoding(), nullptr);
  EXPECT_EQ(instruction.raw_encoding()[3], 0x04223110u);
}

} // namespace
} // namespace rocjitsu
