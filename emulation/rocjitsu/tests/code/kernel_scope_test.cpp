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
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
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
