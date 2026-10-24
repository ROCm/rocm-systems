// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/consan/image/artifact.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iterator>

namespace rocjitsu::consan {
namespace {

class ConSanArtifact : public ::testing::Test {
protected:
  void SetUp() override {
    std::ifstream file(CONSAN_ORIGINAL_FIXTURE, std::ios::binary);
    original_ = {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(original_.empty());
    FailureOr<LoadedArtifact> loaded =
        load_artifact(CONSAN_ARTIFACT_FIXTURE, original_, "gfx1100", &error_);
    ASSERT_TRUE(loaded.succeeded()) << error_.message;
    artifact_ = std::move(loaded.value());
  }
  void rehash() { artifact_.manifest.prepared.sha256 = digest(artifact_.prepared); }
  std::vector<uint8_t> original_;
  LoadedArtifact artifact_;
  ContractError error_;
};

TEST_F(ConSanArtifact, SeparatePreparationProcessProducesLoadableOwningFactsAndImages) {
  EXPECT_TRUE(verify_artifact_images(artifact_.manifest, original_, artifact_.prepared, &error_)
                  .succeeded())
      << error_.message;
  EXPECT_EQ(artifact_.manifest.original.sha256, digest(original_));
  EXPECT_EQ(artifact_.manifest.prepared.sha256, digest(artifact_.prepared));
  EXPECT_EQ(artifact_.manifest.sites.size(), 2u);
  EXPECT_EQ(artifact_.manifest.entry.prepared_sgprs, 52u);
}

TEST_F(ConSanArtifact, MismatchedOriginalAndAgentFailWithoutACompilerFallback) {
  EXPECT_TRUE(load_artifact(CONSAN_ARTIFACT_FIXTURE, original_, "gfx1201", &error_).failed());
  EXPECT_NE(error_.message.find("concrete agent differs"), std::string::npos);
  original_.back() ^= 1;
  EXPECT_TRUE(load_artifact(CONSAN_ARTIFACT_FIXTURE, original_, "gfx1100", &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::IdentityMismatch);
  EXPECT_TRUE(load_artifact(std::filesystem::path(CONSAN_ARTIFACT_FIXTURE) / "missing", original_,
                            "gfx1100", &error_)
                  .failed());
}

TEST_F(ConSanArtifact, ChecksummedArtifactCannotLieAboutItsMetadataResources) {
  ++artifact_.manifest.entry.prepared_sgprs;
  ASSERT_TRUE(match_images(artifact_.manifest, original_, artifact_.prepared).succeeded());
  EXPECT_TRUE(
      verify_artifact_images(artifact_.manifest, original_, artifact_.prepared, &error_).failed());
  EXPECT_NE(error_.message.find("metadata resources"), std::string::npos);
}

TEST_F(ConSanArtifact, RecomputedImageDigestCannotHideAnInvalidDescriptorOrElfExtent) {
  std::vector<uint8_t> &image = artifact_.prepared;
  const size_t descriptor = artifact_.manifest.entry.prepared_descriptor_offset;
  // The first descriptor dword is the LDS extent. A valid hash cannot make
  // this disagree with its owning facts, even before HSA sees the image.
  image[descriptor] ^= 16;
  rehash();
  ASSERT_TRUE(match_images(artifact_.manifest, original_, image).succeeded());
  EXPECT_TRUE(verify_artifact_images(artifact_.manifest, original_, image, &error_).failed());
  EXPECT_NE(error_.message.find("descriptor"), std::string::npos);
  image[descriptor] ^= 16;
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_phoff = UINT64_MAX;
  std::memcpy(image.data(), &header, sizeof(header));
  rehash();
  EXPECT_TRUE(verify_artifact_images(artifact_.manifest, original_, image, &error_).failed());
  EXPECT_NE(error_.message.find("tables"), std::string::npos);
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
  artifact_.manifest.prepared.bytes = image.size();
  rehash();
  ASSERT_TRUE(verify_artifact_images(artifact_.manifest, original_, image, &error_).succeeded())
      << error_.message;
  for (const bool section_table : {true, false}) {
    Elf64_Ehdr invalid = header;
    if (section_table)
      invalid.e_shentsize += 8;
    else
      invalid.e_phentsize += 8;
    std::memcpy(image.data(), &invalid, sizeof(invalid));
    rehash();
    EXPECT_TRUE(verify_artifact_images(artifact_.manifest, original_, image, &error_).failed());
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidEncoding);
    EXPECT_EQ(error_.message, "artifact ELF target, ABI or tables are inconsistent");
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
      segment.p_filesz = artifact_.manifest.entry.prepared_entry_offset + retained_after_entry;
      ASSERT_LT(segment.p_filesz, segment.p_memsz);
      // Keep the remaining memory extent: zero-filled executable memory is
      // not a substitute for the required instruction bytes in the file.
      std::memcpy(image.data() + offset, &segment, sizeof(segment));
      shortened = true;
    }
    ASSERT_TRUE(shortened);
    rehash();
    ASSERT_TRUE(match_images(artifact_.manifest, original_, image).succeeded());
    EXPECT_TRUE(verify_artifact_images(artifact_.manifest, original_, image, &error_).failed())
        << "accepted only " << retained_after_entry << " bytes after entry";
    EXPECT_NE(error_.message.find("executable .text"), std::string::npos);
  }
}

} // namespace
} // namespace rocjitsu::consan
