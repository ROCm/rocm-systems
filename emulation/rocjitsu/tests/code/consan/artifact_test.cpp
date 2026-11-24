// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdhsa_metadata.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/isa/register_set.h"
#include "util/bit.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iterator>

namespace rocjitsu::consan {
namespace {

// This constructs only structural bindings over a real three-entry ELF. It is
// deliberately not an AOT result or evidence of probe execution: the verifier
// has no decoder and cannot certify the producer's instrumentation algorithm.
class ConSanWholeImageStructure : public ::testing::Test {
protected:
  void SetUp() override {
    std::ifstream file(CONSAN_MULTI_ENTRY_FIXTURE, std::ios::binary);
    original_ = {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(original_.empty());
    prepared_ = original_;
    Elf64_Ehdr header = {};
    std::memcpy(&header, original_.data(), sizeof(header));
    const std::optional<std::vector<Elf64_Shdr>> sections =
        read_elf_table<Elf64_Shdr>(original_, header.e_shoff, header.e_shnum, header.e_shentsize);
    ASSERT_TRUE(sections);
    for (const Elf64_Shdr &section : *sections)
      if ((section.sh_flags & (SHF_EXECINSTR | SHF_ALLOC)) == (SHF_EXECINSTR | SHF_ALLOC))
        text_ = section;
    ASSERT_GT(text_.sh_size, 0u);
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(original_, text_.sh_offset, text_.sh_size);
    ASSERT_EQ(kernels.size(), 3u);
    facts_.original = {.sha256 = digest(original_), .bytes = original_.size()};
    facts_.prepared = facts_.original;
    facts_.target = 1250;
    facts_.wave_size = 32;
    facts_.code_object_abi = 5;
    facts_.original_text_bytes = text_.sh_size;
    facts_.prepared_text_bytes = text_.sh_size;
    for (const KernelDescriptorInfo &kernel : kernels) {
      const FailureOr<MetadataResources> metadata = inspect_metadata(original_, kernel);
      ASSERT_TRUE(metadata.succeeded());
      ImageEntryFacts entry;
      entry.key = kernel.descriptor_file_offset;
      entry.original_sgprs = metadata.value().sgprs;
      entry.original_vgprs = metadata.value().vgprs;
      entry.binding = {.name = kernel.kernel_name,
                       .original_descriptor_offset = entry.key,
                       .prepared_descriptor_offset = entry.key,
                       .original_entry_offset = kernel.entry_text_offset,
                       .prepared_entry_offset = kernel.entry_text_offset,
                       .original_kernarg_bytes = kernel.descriptor.kernarg_size,
                       .kernarg_alignment = 8,
                       .original_group_bytes = kernel.descriptor.group_segment_fixed_size,
                       .prepared_group_bytes = kernel.descriptor.group_segment_fixed_size,
                       .prepared_sgprs = metadata.value().sgprs,
                       .prepared_vgprs = metadata.value().vgprs,
                       .maximum_workgroup_size = metadata.value().maximum_workgroup_size,
                       .workgroup_id_scalar =
                           RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc2,
                                              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X)
                               ? kWorkgroupIdTtmp9Selector
                               : UINT32_MAX};
      if (kernel.kernel_name == "mk_a") {
        entry.disposition = EntryDisposition::Selected;
        Manifest manifest;
        manifest.original = facts_.original;
        manifest.prepared = facts_.prepared;
        manifest.options = mvp_options_digest();
        manifest.target = 1250;
        manifest.wave_size = 32;
        manifest.code_object_abi = 5;
        manifest.original_text_bytes = text_.sh_size;
        manifest.prepared_text_bytes = text_.sh_size;
        manifest.entry = entry.binding;
        manifest.sites = {{.analysis = AnalysisKind::LdsRaceDetection,
                           .access = AccessKind::Write,
                           .width = 4,
                           .original_offset = kernel.entry_text_offset + 52,
                           .original_size = 8,
                           .prepared_guest_offset = kernel.entry_text_offset + 52,
                           .prepared_guest_size = 8,
                           .probe_offset = kernel.entry_text_offset + 28,
                           .probe_size = 4}};
        entry.selected = std::move(manifest);
      } else {
        uint64_t end = text_.sh_size;
        for (const KernelDescriptorInfo &other : kernels)
          if (other.entry_text_offset > kernel.entry_text_offset)
            end = std::min(end, other.entry_text_offset);
        entry.native_code = {{.original_offset = kernel.entry_text_offset,
                              .prepared_offset = kernel.entry_text_offset,
                              .bytes = end - kernel.entry_text_offset}};
      }
      facts_.entries.push_back(std::move(entry));
    }
    std::ranges::sort(facts_.entries, {}, &ImageEntryFacts::key);
    const std::vector<KernelDescriptorInfo>::const_iterator selected_kernel =
        std::ranges::find(kernels, std::string("mk_a"), &KernelDescriptorInfo::kernel_name);
    ASSERT_NE(selected_kernel, kernels.end());
    amdhsa::kernel_descriptor_t descriptor = selected_kernel->descriptor;
    const std::optional<KernargExtensionLayout> wrapper =
        make_kernarg_extension_layout(descriptor.kernarg_size, kContextPointerPayloads);
    ASSERT_TRUE(wrapper);
    descriptor.kernarg_size = wrapper->wrapper_size;
    ASSERT_TRUE(amdhsa::update_resource_metadata(prepared_, *selected_kernel, descriptor,
                                                 std::nullopt, std::nullopt)
                    .succeeded());
    std::memcpy(prepared_.data() + selected_kernel->descriptor_file_offset, &descriptor,
                sizeof(descriptor));
    rehash();
    const std::array<EntryKey, 1> selected = {facts_.entries.front().key};
    facts_.selection_profile = selection_profile_digest(selected);
  }
  void rehash() {
    facts_.prepared.sha256 = digest(prepared_);
    facts_.entries.front().selected->prepared = facts_.prepared;
  }
  std::vector<uint8_t> original_;
  std::vector<uint8_t> prepared_;
  ImageFacts facts_;
  Elf64_Shdr text_ = {};
  util::StringDiagnostic error_;
};

TEST_F(ConSanWholeImageStructure, ChecksAllThreeDescriptorsAndDistinctMetadataRows) {
  ASSERT_TRUE(verify_artifact_images(facts_, original_, prepared_, error_.emitter()).succeeded())
      << error_.message();
  EXPECT_EQ(facts_.entries[0].binding.original_kernarg_bytes, 16u);
  EXPECT_EQ(facts_.entries[1].binding.original_kernarg_bytes, 32u);
  EXPECT_EQ(facts_.entries[2].binding.original_kernarg_bytes, 24u);
  EXPECT_EQ(facts_.entries[1].disposition, EntryDisposition::Native);
  EXPECT_EQ(facts_.entries[2].binding.original_group_bytes, 64u);
}

TEST_F(ConSanWholeImageStructure, RecomputedHashCannotHideNativeBodyOrDescriptorMutation) {
  const std::vector<uint8_t> saved = prepared_;
  const ImageEntryFacts &native = facts_.entries.back();
  prepared_[text_.sh_offset + native.binding.prepared_entry_offset] ^= 1;
  rehash();
  EXPECT_TRUE(verify_artifact_images(facts_, original_, prepared_, error_.emitter()).failed());
  EXPECT_NE(error_.message().find("native entry body"), std::string::npos);
  prepared_ = saved;
  prepared_[native.binding.prepared_descriptor_offset + 49] ^= 0x80;
  rehash();
  EXPECT_TRUE(verify_artifact_images(facts_, original_, prepared_, error_.emitter()).failed());
  EXPECT_NE(error_.message().find("native descriptor"), std::string::npos);
}

class ConSanArtifact : public ::testing::Test {
protected:
  void SetUp() override {
    std::ifstream file(CONSAN_ORIGINAL_FIXTURE, std::ios::binary);
    original_ = {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(original_.empty());
    std::ifstream facts_file(std::filesystem::path(CONSAN_ARTIFACT_FIXTURE) / "facts.bin",
                             std::ios::binary);
    const std::vector<uint8_t> facts_bytes = {std::istreambuf_iterator<char>(facts_file),
                                              std::istreambuf_iterator<char>()};
    const FailureOr<ImageFacts> facts = decode_image_facts(facts_bytes, error_.emitter());
    ASSERT_TRUE(facts.succeeded()) << error_.message();
    profile_ = facts.value().selection_profile;
    FailureOr<LoadedArtifact> loaded =
        load_artifact(CONSAN_ARTIFACT_FIXTURE, original_, "gfx1100", profile_, error_.emitter());
    ASSERT_TRUE(loaded.succeeded()) << error_.message();
    artifact_ = std::move(loaded.value());
  }
  Manifest &manifest() { return *artifact_.facts.entries.front().selected; }
  void rehash() {
    manifest().prepared.sha256 = digest(artifact_.prepared);
    artifact_.facts.prepared = manifest().prepared;
  }
  std::vector<uint8_t> original_;
  LoadedArtifact artifact_;
  util::StringDiagnostic error_;
  Digest profile_{};
};

TEST_F(ConSanArtifact, SeparatePreparationProcessProducesLoadableOwningFactsAndImages) {
  EXPECT_TRUE(verify_artifact_images(manifest(), original_, artifact_.prepared, error_.emitter())
                  .succeeded())
      << error_.message();
  EXPECT_EQ(manifest().original.sha256, digest(original_));
  EXPECT_EQ(manifest().prepared.sha256, digest(artifact_.prepared));
  EXPECT_EQ(manifest().sites.size(), 2u);
  EXPECT_GT(manifest().entry.prepared_sgprs, 0u);
  EXPECT_LE(manifest().entry.prepared_sgprs, REGISTER_SET_ALLOCATABLE_SGPRS);
  EXPECT_GT(manifest().entry.prepared_vgprs, 0u);
  EXPECT_LE(manifest().entry.prepared_vgprs, 256u);
}

TEST_F(ConSanArtifact, ReboundWorkgroupInputCannotBypassWholeImageDescriptorVerification) {
  ImageEntryFacts &entry = artifact_.facts.entries.front();
  entry.binding.workgroup_id_scalar =
      entry.binding.workgroup_id_scalar == UINT32_MAX ? 15 : UINT32_MAX;
  entry.selected->entry = entry.binding;
  ASSERT_TRUE(validate_image_facts(artifact_.facts, error_.emitter()).succeeded())
      << error_.message();
  EXPECT_TRUE(
      verify_artifact_images(artifact_.facts, original_, artifact_.prepared, error_.emitter())
          .failed());
  EXPECT_NE(error_.message().find("workgroup-X input"), std::string::npos);
}

TEST_F(ConSanArtifact, MismatchedOriginalAndAgentFailWithoutACompilerFallback) {
  EXPECT_TRUE(
      load_artifact(CONSAN_ARTIFACT_FIXTURE, original_, "gfx1201", profile_, error_.emitter())
          .failed());
  EXPECT_NE(error_.message().find("concrete agent differs"), std::string::npos);
  original_.back() ^= 1;
  EXPECT_TRUE(
      load_artifact(CONSAN_ARTIFACT_FIXTURE, original_, "gfx1100", profile_, error_.emitter())
          .failed());
  EXPECT_EQ(error_.message(), "image or configured selection identity mismatch");
  EXPECT_TRUE(load_artifact(std::filesystem::path(CONSAN_ARTIFACT_FIXTURE) / "missing", original_,
                            "gfx1100", profile_, error_.emitter())
                  .failed());
}

TEST_F(ConSanArtifact, IndependentlyConfiguredWrongSelectionIsRejected) {
  Digest wrong = profile_;
  wrong[0] ^= 1;
  EXPECT_TRUE(load_artifact(CONSAN_ARTIFACT_FIXTURE, original_, "gfx1100", wrong, error_.emitter())
                  .failed());
  EXPECT_EQ(error_.message(), "configured selection profile differs from artifact");
  EXPECT_NE(error_.message().find("selection profile"), std::string::npos);
}

TEST_F(ConSanArtifact, WholeImageFactsCannotRebindAnEntryOrHideTheInventory) {
  const ImageFacts saved = artifact_.facts;
  artifact_.facts.entries.front().binding.name = "rebound";
  EXPECT_TRUE(
      verify_artifact_images(artifact_.facts, original_, artifact_.prepared, error_.emitter())
          .failed());
  artifact_.facts = saved;
  artifact_.facts.entries.clear();
  EXPECT_TRUE(
      verify_artifact_images(artifact_.facts, original_, artifact_.prepared, error_.emitter())
          .failed());
}

TEST_F(ConSanArtifact, DescriptorDiscoveryDeduplicationCannotHideAnExtraAlias) {
  std::vector<uint8_t> &image = artifact_.prepared;
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  std::optional<std::vector<Elf64_Shdr>> sections =
      read_elf_table<Elf64_Shdr>(image, header.e_shoff, header.e_shnum, header.e_shentsize);
  ASSERT_TRUE(sections);
  bool added = false;
  for (Elf64_Shdr &section : *sections) {
    if (section.sh_type != SHT_SYMTAB)
      continue;
    std::optional<std::vector<Elf64_Sym>> symbols = read_elf_table<Elf64_Sym>(
        image, section.sh_offset, section.sh_size / sizeof(Elf64_Sym), section.sh_entsize);
    ASSERT_TRUE(symbols);
    Elf64_Shdr &strings = (*sections)[section.sh_link];
    const std::vector<uint8_t> saved_strings(image.begin() + strings.sh_offset,
                                             image.begin() + strings.sh_offset + strings.sh_size);
    Elf64_Sym alias = {};
    for (const Elf64_Sym &symbol : *symbols)
      if (symbol.st_size == 64 && elf_symbol_type(symbol.st_info) == kElfSymbolTypeObject &&
          elf_symbol_bind(symbol.st_info) == kElfSymbolBindGlobal)
        alias = symbol;
    ASSERT_EQ(alias.st_size, 64u);
    alias.st_name = saved_strings.size();
    constexpr std::string_view kAlias = "unexpected_alias.kd";
    strings.sh_offset = image.size();
    image.insert(image.end(), saved_strings.begin(), saved_strings.end());
    image.insert(image.end(), kAlias.begin(), kAlias.end());
    image.push_back(0);
    strings.sh_size = saved_strings.size() + kAlias.size() + 1;
    symbols->push_back(alias);
    section.sh_offset = image.size();
    section.sh_size = symbols->size() * sizeof(Elf64_Sym);
    image.resize(image.size() + section.sh_size);
    std::memcpy(image.data() + section.sh_offset, symbols->data(), section.sh_size);
    added = true;
    break;
  }
  ASSERT_TRUE(added);
  header.e_shoff = image.size();
  image.resize(image.size() + sections->size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + header.e_shoff, sections->data(),
              sections->size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data(), &header, sizeof(header));
  manifest().prepared.bytes = image.size();
  rehash();
  EXPECT_TRUE(verify_artifact_images(manifest(), original_, image, error_.emitter()).failed());
  EXPECT_NE(error_.message().find("alias"), std::string::npos);
}

TEST_F(ConSanArtifact, ChecksummedArtifactCannotLieAboutItsMetadataResources) {
  ++manifest().entry.prepared_sgprs;
  ASSERT_TRUE(match_images(manifest(), original_, artifact_.prepared).succeeded());
  EXPECT_TRUE(
      verify_artifact_images(manifest(), original_, artifact_.prepared, error_.emitter()).failed());
  EXPECT_NE(error_.message().find("metadata resources"), std::string::npos);
}

TEST_F(ConSanArtifact, RecomputedImageDigestCannotHideAnInvalidDescriptorOrElfExtent) {
  std::vector<uint8_t> &image = artifact_.prepared;
  const size_t descriptor = manifest().entry.prepared_descriptor_offset;
  // The first descriptor dword is the LDS extent. A valid hash cannot make
  // this disagree with its owning facts, even before HSA sees the image.
  image[descriptor] ^= 16;
  rehash();
  ASSERT_TRUE(match_images(manifest(), original_, image).succeeded());
  EXPECT_TRUE(verify_artifact_images(manifest(), original_, image, error_.emitter()).failed());
  EXPECT_NE(error_.message().find("descriptor"), std::string::npos);
  image[descriptor] ^= 16;
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_phoff = UINT64_MAX;
  std::memcpy(image.data(), &header, sizeof(header));
  rehash();
  EXPECT_TRUE(verify_artifact_images(manifest(), original_, image, error_.emitter()).failed());
  EXPECT_NE(error_.message().find("tables"), std::string::npos);
}

TEST_F(ConSanArtifact, UnalignedElfTablesRemainReadableWithoutRelaxingEntrySizes) {
  std::vector<uint8_t> &image = artifact_.prepared;
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  const std::optional<std::vector<Elf64_Shdr>> sections =
      read_elf_table<Elf64_Shdr>(image, header.e_shoff, header.e_shnum, header.e_shentsize);
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  ASSERT_TRUE(sections && segments);
  while (image.size() % 8 != 1)
    image.push_back(0);
  header.e_shoff = image.size();
  image.resize(image.size() + sections->size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + header.e_shoff, sections->data(),
              sections->size() * sizeof(Elf64_Shdr));
  ASSERT_EQ(image.size() % 8, 1u);
  header.e_phoff = image.size();
  image.resize(image.size() + segments->size() * sizeof(Elf64_Phdr));
  std::memcpy(image.data() + header.e_phoff, segments->data(),
              segments->size() * sizeof(Elf64_Phdr));
  std::memcpy(image.data(), &header, sizeof(header));
  manifest().prepared.bytes = image.size();
  rehash();
  ASSERT_TRUE(verify_artifact_images(manifest(), original_, image, error_.emitter()).succeeded())
      << error_.message();
  for (const bool section_table : {true, false}) {
    Elf64_Ehdr invalid = header;
    if (section_table)
      invalid.e_shentsize += 8;
    else
      invalid.e_phentsize += 8;
    std::memcpy(image.data(), &invalid, sizeof(invalid));
    rehash();
    EXPECT_TRUE(verify_artifact_images(manifest(), original_, image, error_.emitter()).failed());
    EXPECT_EQ(error_.message(), "artifact ELF target, ABI or tables are inconsistent");
  }
}

TEST_F(ConSanArtifact, ChecksummedEntryCoverageCannotSubstituteForTheWholeExecutableText) {
  const std::vector<uint8_t> saved = artifact_.prepared;
  for (const uint64_t retained_after_entry : {4u, 8u, 128u}) {
    artifact_.prepared = saved;
    std::vector<uint8_t> &image = artifact_.prepared;
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    bool shortened = false;
    for (size_t index = 0; index < header.e_phnum; ++index) {
      const size_t offset = header.e_phoff + index * sizeof(Elf64_Phdr);
      Elf64_Phdr segment{};
      std::memcpy(&segment, image.data() + offset, sizeof(segment));
      if (segment.p_type != PT_LOAD || (segment.p_flags & PF_X) == 0)
        continue;
      segment.p_filesz = manifest().entry.prepared_entry_offset + retained_after_entry;
      ASSERT_LT(segment.p_filesz, segment.p_memsz);
      // Keep the remaining memory extent: zero-filled executable memory is
      // not a substitute for the required instruction bytes in the file.
      std::memcpy(image.data() + offset, &segment, sizeof(segment));
      shortened = true;
    }
    ASSERT_TRUE(shortened);
    rehash();
    ASSERT_TRUE(match_images(manifest(), original_, image).succeeded());
    EXPECT_TRUE(verify_artifact_images(manifest(), original_, image, error_.emitter()).failed())
        << "accepted only " << retained_after_entry << " bytes after entry";
    EXPECT_NE(error_.message().find("executable .text"), std::string::npos);
  }
}

} // namespace
} // namespace rocjitsu::consan
