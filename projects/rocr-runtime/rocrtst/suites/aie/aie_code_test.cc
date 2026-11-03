/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Exercises AMD::AieCode::Create (core/inc/amd_aie_code.hpp) directly. The parser sources are
// compiled into this binary because core/inc is not installed alongside the runtime, and
// amd::elf::Image (which AieCode depends on) has hidden visibility in libhsa-runtime64.so.

#include <gtest/gtest.h>

#include <elf.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/inc/amd_elf_image.hpp"
#include "core/inc/amd_aie_code.hpp"

namespace {

/**
 * @brief Reads a whole file into memory.
 *
 * @param path file to read
 * @return the file contents, or an empty vector if the file cannot be opened
 */
std::vector<std::uint8_t> ReadFile(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> bytes(size);
  f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  return bytes;
}

using rocr::AMD::aie_kernel_entry;
using rocr::AMD::aie_section_header;
using rocr::AMD::AieCode;

// Same hsaco test_dispatch.cc's HsacoKernelObjectIsPublished loads; built by the same CMake rule.
constexpr const char* kHsaco = "vsadd.hsaco";
// vsmul packed as aie2 and then vsadd as aie2p, in that order, so the aie2p section is not the
// first AIE section in the file.
constexpr const char* kMultiArchHsaco = "multiarch.hsaco";
// The arch vsadd.hsaco was packed for.
constexpr const char* kArch = AIE_HSACO_ARCH;

/**
 * @brief Parses the AIE section for one arch out of an hsaco.
 *
 * @param hsaco hsaco contents
 * @param arch arch whose section to parse
 * @param code receives the parsed code object, or null on failure
 * @return the status of AieCode::Create
 */
hsa_status_t ParseArchSection(const std::vector<std::uint8_t>& hsaco, const char* arch,
                              std::unique_ptr<AieCode>* code) {
  return AieCode::Create(hsaco.data(), hsaco.size(), arch, code);
}

/**
 * @brief Finds an ELF64 section by name.
 *
 * @param image ELF contents
 * @param name section name
 * @return the section's file offset, or 0 if there is none
 */
std::size_t SectionOffset(const std::vector<std::uint8_t>& image, const char* name) {
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  Elf64_Shdr shstrtab{};
  std::memcpy(&shstrtab, image.data() + ehdr.e_shoff + ehdr.e_shstrndx * sizeof(Elf64_Shdr),
              sizeof(shstrtab));
  for (std::size_t i = 1; i < ehdr.e_shnum; ++i) {
    Elf64_Shdr sh{};
    std::memcpy(&sh, image.data() + ehdr.e_shoff + i * sizeof(Elf64_Shdr), sizeof(sh));
    const auto* sh_name =
        reinterpret_cast<const char*>(image.data() + shstrtab.sh_offset + sh.sh_name);
    if (std::strcmp(sh_name, name) == 0) return sh.sh_offset;
  }
  return 0;
}

// The tests below each change one field of a known-good hsaco, so what they feed the parser differs
// from it in exactly the one way named. Field writes go through memcpy: the section is not aligned
// in the file.
class AieCodeMutation : public ::testing::Test {
 protected:
  void SetUp() override {
    hsaco_ = ReadFile(kHsaco);
    if (hsaco_.empty()) GTEST_SKIP() << "hsaco was not built: " << kHsaco;
    section_ = SectionOffset(hsaco_, kArch);
    ASSERT_NE(section_, 0u) << "no " << kArch << " section in " << kHsaco;
    // Unmodified, it parses: whatever a mutation provokes is the mutation's doing.
    std::unique_ptr<AieCode> code;
    ASSERT_EQ(ParseArchSection(hsaco_, kArch, &code), HSA_STATUS_SUCCESS);
  }

  /**
   * @brief The AIE section header of the hsaco under test.
   *
   * @return a copy of the header
   */
  aie_section_header Header() const {
    aie_section_header hdr{};
    std::memcpy(&hdr, hsaco_.data() + section_, sizeof(hdr));
    return hdr;
  }

  /**
   * @brief Overwrites a 32-bit field of the AIE section header.
   *
   * @param field_offset offset of the field within aie_section_header
   * @param value new value
   */
  void SetHeaderField(std::size_t field_offset, std::uint32_t value) {
    std::memcpy(hsaco_.data() + section_ + field_offset, &value, sizeof(value));
  }

  /**
   * @brief Overwrites a 32-bit field of the AIE section's first kernel entry.
   *
   * @param field_offset offset of the field within aie_kernel_entry
   * @param value new value
   */
  void SetEntryField(std::size_t field_offset, std::uint32_t value) {
    std::memcpy(hsaco_.data() + section_ + Header().header_size + field_offset, &value,
                sizeof(value));
  }

  std::vector<std::uint8_t> hsaco_;
  std::size_t section_ = 0;
};

TEST(AieCodeParse, ParsesVsaddHsaco) {
  const auto hsaco = ReadFile(kHsaco);
  if (hsaco.empty()) GTEST_SKIP() << "hsaco was not built: " << kHsaco;

  std::unique_ptr<AieCode> code;
  ASSERT_EQ(ParseArchSection(hsaco, kArch, &code), HSA_STATUS_SUCCESS);
  ASSERT_NE(code, nullptr);
  EXPECT_EQ(code->GetArchSectionName(), kArch);

  const auto names = code->GetKernelNames();
  ASSERT_EQ(names.size(), 1u);

  const auto* info = code->GetKernel(names.front());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->kind, rocr::AMD::AieKernelKind::PdiInsts);
  EXPECT_GT(info->insts_size, 0u);
  EXPECT_GT(info->pdi_size, 0u);
}

TEST(AieCodeParse, RejectsNullAndEmpty) {
  std::unique_ptr<AieCode> code;
  EXPECT_EQ(AieCode::Create(nullptr, 0, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  const std::vector<std::uint8_t> empty;
  EXPECT_EQ(ParseArchSection(empty, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(code, nullptr);
}

TEST(AieCodeParse, RejectsGarbage) {
  const std::vector<std::uint8_t> garbage(512, 0xA5);
  std::unique_ptr<AieCode> code;
  EXPECT_EQ(ParseArchSection(garbage, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

TEST(AieCodeParse, RejectsTruncated) {
  auto hsaco = ReadFile(kHsaco);
  if (hsaco.empty()) GTEST_SKIP() << "hsaco was not built: " << kHsaco;
  hsaco.resize(hsaco.size() / 2);

  std::unique_ptr<AieCode> code;
  EXPECT_NE(ParseArchSection(hsaco, kArch, &code), HSA_STATUS_SUCCESS);
}

TEST(AieCodeParse, ReportsMissingArchAsIncompatible) {
  // A well-formed hsaco without a section for the agent's arch is the wrong code object for the
  // agent, not a malformed one.
  const auto hsaco = ReadFile(kHsaco);
  if (hsaco.empty()) GTEST_SKIP() << "hsaco was not built: " << kHsaco;

  std::unique_ptr<AieCode> code;
  EXPECT_EQ(ParseArchSection(hsaco, "aie4", &code), HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
}

TEST(AieCodeParse, SelectsTheAgentsArchSection) {
  // Both arches are found, including aie2p, which is not the first AIE section in the file.
  const auto hsaco = ReadFile(kMultiArchHsaco);
  if (hsaco.empty()) GTEST_SKIP() << "hsaco was not built: " << kMultiArchHsaco;

  std::unique_ptr<AieCode> aie2;
  ASSERT_EQ(ParseArchSection(hsaco, "aie2", &aie2), HSA_STATUS_SUCCESS);
  EXPECT_EQ(aie2->GetArchSectionName(), "aie2");
  EXPECT_EQ(aie2->GetKernelNames(), std::vector<std::string>{"vsmul"});

  std::unique_ptr<AieCode> aie2p;
  ASSERT_EQ(ParseArchSection(hsaco, "aie2p", &aie2p), HSA_STATUS_SUCCESS);
  EXPECT_EQ(aie2p->GetArchSectionName(), "aie2p");
  EXPECT_EQ(aie2p->GetKernelNames(), std::vector<std::string>{"vsadd"});
}

TEST_F(AieCodeMutation, RejectsHeaderSizeBelowTheHeader) {
  // The kernel table then overlaps the header. Its entries are header bytes, which other checks
  // also refuse, so this asserts the outcome rather than isolating the header_size check.
  SetHeaderField(offsetof(aie_section_header, header_size), sizeof(aie_section_header) - 4);
  std::unique_ptr<AieCode> code;
  EXPECT_EQ(ParseArchSection(hsaco_, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

TEST_F(AieCodeMutation, RejectsStringTableOverlappingTheKernelTable) {
  SetHeaderField(offsetof(aie_section_header, string_table_offset), Header().header_size);
  std::unique_ptr<AieCode> code;
  EXPECT_EQ(ParseArchSection(hsaco_, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

TEST_F(AieCodeMutation, RejectsBlobPoolOverlappingTheStringTable) {
  SetHeaderField(offsetof(aie_section_header, blob_pool_offset), Header().string_table_offset);
  std::unique_ptr<AieCode> code;
  EXPECT_EQ(ParseArchSection(hsaco_, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

TEST_F(AieCodeMutation, RejectsBlobBeforeTheBlobPool) {
  // Points the instruction blob back at the section header, which lies within the section but
  // not within the pool.
  SetEntryField(offsetof(aie_kernel_entry, insts_offset), 0);
  std::unique_ptr<AieCode> code;
  EXPECT_EQ(ParseArchSection(hsaco_, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

TEST_F(AieCodeMutation, RejectsNameOutsideTheStringTable) {
  // Past the end of the table, not at it: an offset at the end is caught by the terminator check
  // alone. One byte further, only the bound keeps the name from being read out of the blob pool.
  SetEntryField(offsetof(aie_kernel_entry, name_offset), Header().string_table_size + 1);
  std::unique_ptr<AieCode> code;
  EXPECT_EQ(ParseArchSection(hsaco_, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

TEST_F(AieCodeMutation, RejectsFullElfEntryWithSeparatePdi) {
  // vsadd's entry carries a PDI, which a full ELF never does: it holds its own.
  SetEntryField(offsetof(aie_kernel_entry, kind),
                static_cast<std::uint32_t>(rocr::AMD::AieKernelKind::FullElf));
  std::unique_ptr<AieCode> code;
  EXPECT_EQ(ParseArchSection(hsaco_, kArch, &code), HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
}

}  // namespace
