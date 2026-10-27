/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <elf.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core/inc/amd_aie_elf.h"

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

constexpr const char* kElf = "kernel_full_elf_vsadd/aie.elf";
// The only arch with a full-ELF format the reader implements.
constexpr const char* kArch = "aie2p";

// ---------------------------------------------------------------------------
// In-memory ELF surgery
//
// The tests below build malformed variants of the real artifact rather than hand-writing an ELF,
// so what they feed the parser differs from a known-good image in exactly the one way named.
// ---------------------------------------------------------------------------

/**
 * @brief The ELF header of an image.
 *
 * @param image ELF contents
 * @return a copy of the ELF header
 */
Elf32_Ehdr Header(const std::vector<std::uint8_t>& image) {
  Elf32_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  return ehdr;
}

/**
 * @brief The header of a section.
 *
 * @param image ELF contents
 * @param index section index
 * @return the section header
 */
Elf32_Shdr SectionHeader(const std::vector<std::uint8_t>& image, std::uint32_t index) {
  const Elf32_Ehdr ehdr = Header(image);
  Elf32_Shdr shdr{};
  std::memcpy(&shdr, image.data() + ehdr.e_shoff + index * sizeof(Elf32_Shdr), sizeof(shdr));
  return shdr;
}

/**
 * @brief Overwrites the header of a section.
 *
 * @param image ELF contents to modify
 * @param index section index
 * @param shdr new section header
 */
void SetSectionHeader(std::vector<std::uint8_t>& image, std::uint32_t index,
                      const Elf32_Shdr& shdr) {
  const Elf32_Ehdr ehdr = Header(image);
  std::memcpy(image.data() + ehdr.e_shoff + index * sizeof(Elf32_Shdr), &shdr, sizeof(shdr));
}

/**
 * @brief Finds a section by name.
 *
 * @param image ELF contents
 * @param name section name
 * @return the section index, or 0 (the NULL section, never a match) if there is none
 */
std::uint32_t FindSection(const std::vector<std::uint8_t>& image, const char* name) {
  const Elf32_Ehdr ehdr = Header(image);
  const Elf32_Shdr shstrtab = SectionHeader(image, ehdr.e_shstrndx);
  for (std::uint32_t i = 1; i < ehdr.e_shnum; ++i) {
    const Elf32_Shdr sh = SectionHeader(image, i);
    const auto* sh_name =
        reinterpret_cast<const char*>(image.data() + shstrtab.sh_offset + sh.sh_name);
    if (std::strcmp(sh_name, name) == 0) return i;
  }
  return 0;
}

/**
 * @brief The contents of a section, looked up by name.
 *
 * @param image ELF contents
 * @param name section name
 * @return a copy of the section contents, or an empty vector if there is no such section
 */
std::vector<std::uint8_t> SectionContents(const std::vector<std::uint8_t>& image,
                                          const char* name) {
  const std::uint32_t index = FindSection(image, name);
  if (index == 0) return {};
  const Elf32_Shdr sh = SectionHeader(image, index);
  return {image.begin() + sh.sh_offset, image.begin() + sh.sh_offset + sh.sh_size};
}

/**
 * @brief Rewrites the symbol and type of a .rela.dyn relocation.
 *
 * The entries are fixed size, so this is an in-place edit that shifts nothing.
 *
 * @param image ELF contents to modify
 * @param index index of the relocation in .rela.dyn
 * @param sym new symbol index
 * @param type new relocation type
 * @return false if there is no .rela.dyn or no relocation at `index`
 */
bool MutateRelocation(std::vector<std::uint8_t>& image, std::uint32_t index, std::uint32_t sym,
                      std::uint32_t type) {
  const std::uint32_t rela = FindSection(image, ".rela.dyn");
  if (rela == 0) return false;
  const Elf32_Shdr sh = SectionHeader(image, rela);
  if ((index + 1) * sizeof(Elf32_Rela) > sh.sh_size) return false;

  const std::size_t entry = sh.sh_offset + index * sizeof(Elf32_Rela);
  const std::uint32_t r_info = (sym << 8) | (type & 0xFF);
  std::memcpy(image.data() + entry + offsetof(Elf32_Rela, r_info), &r_info, sizeof(r_info));
  return true;
}

/**
 * @brief Rewrites the r_offset (the patch site) of a .rela.dyn relocation.
 *
 * @param image ELF contents to modify
 * @param index index of the relocation in .rela.dyn
 * @param offset new patch site
 * @return false if there is no .rela.dyn or no relocation at `index`
 */
bool SetRelocationOffset(std::vector<std::uint8_t>& image, std::uint32_t index,
                         std::uint32_t offset) {
  const std::uint32_t rela = FindSection(image, ".rela.dyn");
  if (rela == 0) return false;
  const Elf32_Shdr sh = SectionHeader(image, rela);
  if ((index + 1) * sizeof(Elf32_Rela) > sh.sh_size) return false;

  const std::size_t entry = sh.sh_offset + index * sizeof(Elf32_Rela);
  std::memcpy(image.data() + entry + offsetof(Elf32_Rela, r_offset), &offset, sizeof(offset));
  return true;
}

/**
 * @brief Parses an ELF for kArch, the arch every test here targets.
 *
 * @param image ELF contents
 * @param kernels receives the parsed kernels, keyed by name
 * @param error receives a description of the failure, if any
 * @return the status of aie_elf::Parse
 */
hsa_status_t ParseImage(const std::vector<std::uint8_t>& image,
                        std::map<std::string, rocr::AMD::aie_elf::Kernel>* kernels,
                        std::string* error) {
  return rocr::AMD::aie_elf::Parse(image.data(), image.size(), kArch, kernels, error);
}

// The relocation type the reader reads as "write a 64-bit address here", used for PDI symbols.
constexpr std::uint32_t kAddress64 = 8;

TEST(AieElfParse, ParsesVectorScalarAdd) {
  const auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(rocr::AMD::aie_elf::Parse(image.data(), image.size(), kArch, &kernels, &error),
            HSA_STATUS_SUCCESS)
      << error;
  ASSERT_FALSE(kernels.empty());

  // Compared against the artifact's own sections rather than fixed sizes: their sizes change with
  // the mlir-aie version that built the ELF, but the reader must copy them out verbatim either way.
  const auto ctrltext = SectionContents(image, ".ctrltext.0");
  const auto pdi = SectionContents(image, ".pdi.1");
  ASSERT_FALSE(ctrltext.empty()) << "artifact has no .ctrltext.0 section";
  ASSERT_FALSE(pdi.empty()) << "artifact has no .pdi.1 section";

  const auto& k = kernels.begin()->second;
  EXPECT_EQ(k.ctrl_code, ctrltext);
  ASSERT_EQ(k.pdis.size(), 1u);
  EXPECT_EQ(k.pdis[0].bytes, pdi);
  ASSERT_EQ(k.pdis[0].patch_offsets.size(), 1u);
  EXPECT_NE(k.pdis[0].patch_offsets[0], 0u);
  EXPECT_GT(k.num_args(), 0u);
}

TEST(AieElfParse, RejectsGarbage) {
  const std::vector<std::uint8_t> garbage(512, 0xA5);
  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  EXPECT_NE(rocr::AMD::aie_elf::Parse(garbage.data(), garbage.size(), kArch, &kernels, &error),
            HSA_STATUS_SUCCESS);
  EXPECT_FALSE(error.empty());
}

TEST(AieElfParse, RejectsTruncated) {
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";
  image.resize(image.size() / 2);

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  EXPECT_NE(rocr::AMD::aie_elf::Parse(image.data(), image.size(), kArch, &kernels, &error),
            HSA_STATUS_SUCCESS);
}

/**
 * @brief The r_offset (the patch site) of a .rela.dyn relocation.
 *
 * @param image ELF contents
 * @param index index of the relocation in .rela.dyn
 * @return the patch site, or UINT32_MAX if there is no .rela.dyn or no relocation at `index`
 */
std::uint32_t RelocationOffset(const std::vector<std::uint8_t>& image, std::uint32_t index) {
  const std::uint32_t rela = FindSection(image, ".rela.dyn");
  if (rela == 0) return UINT32_MAX;
  const Elf32_Shdr sh = SectionHeader(image, rela);
  if ((index + 1) * sizeof(Elf32_Rela) > sh.sh_size) return UINT32_MAX;
  std::uint32_t offset = 0;
  std::memcpy(
      &offset,
      image.data() + sh.sh_offset + index * sizeof(Elf32_Rela) + offsetof(Elf32_Rela, r_offset),
      sizeof(offset));
  return offset;
}

/**
 * @brief Renames ".note.xrt.UID", a section the reader ignores, in place.
 *
 * @param image ELF contents to modify
 * @param name new name; must not be longer than the old one, so no other .shstrtab offset moves
 * @return the renamed section's index, or 0 if the artifact has no such section
 */
std::uint32_t RenameSpareSection(std::vector<std::uint8_t>& image, const char* name) {
  const std::uint32_t spare = FindSection(image, ".note.xrt.UID");
  if (spare == 0 || std::strlen(name) > std::strlen(".note.xrt.UID")) return 0;
  const Elf32_Shdr shstrtab = SectionHeader(image, Header(image).e_shstrndx);
  std::memcpy(image.data() + shstrtab.sh_offset + SectionHeader(image, spare).sh_name, name,
              std::strlen(name) + 1);
  return spare;
}

TEST(AieElfParse, ParsesTwoPatchSitesForOnePdi) {
  // A control code may load the same PDI more than once (MLIR-AIR's loads its empty PDI before and
  // after the design runs). Both sites must be kept, since a dropped one leaves that load_pdi
  // pointing at the ELF's placeholder, and the PDI itself held once. Turn the first argument
  // relocation into a second relocation of .pdi.1 (symbol 1, as the ELF's relocation 0 already is).
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  // Rewriting r_info changes the scheme only for an ELF that keeps it there rather than in the
  // addend (ABI version 1); asserting on the resulting sites, not just success, catches that.
  ASSERT_TRUE(MutateRelocation(image, /*index=*/1, /*sym=*/1, kAddress64));
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;
  const auto& k = kernels.begin()->second;
  ASSERT_EQ(k.pdis.size(), 1u);
  EXPECT_EQ(k.pdis[0].bytes, SectionContents(image, ".pdi.1"));
  EXPECT_EQ(k.pdis[0].patch_offsets,
            (std::vector<std::uint32_t>{RelocationOffset(image, 0), RelocationOffset(image, 1)}));
}

TEST(AieElfParse, ParsesTwoPdisInOneKernel) {
  // A control code may load two different PDIs. The artifact has one PDI section, so build a
  // second: rename a section the reader ignores to ".pdi.2" and point a PDI relocation at a symbol
  // of that name. Moving .dynstr to the end of the image to grow it is safe because nothing but its
  // own section header refers to its file offset.
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  ASSERT_NE(RenameSpareSection(image, ".pdi.2"), 0u) << "no spare section to rename";

  // Move and grow .dynstr so it can hold the new symbol name.
  const std::uint32_t dynstr_index = FindSection(image, ".dynstr");
  ASSERT_NE(dynstr_index, 0u);
  Elf32_Shdr dynstr = SectionHeader(image, dynstr_index);
  std::vector<std::uint8_t> strings(image.begin() + dynstr.sh_offset,
                                    image.begin() + dynstr.sh_offset + dynstr.sh_size);
  const std::uint32_t new_name_offset = static_cast<std::uint32_t>(strings.size());
  strings.insert(strings.end(), std::begin(".pdi.2"), std::end(".pdi.2"));
  dynstr.sh_offset = static_cast<Elf32_Off>(image.size());
  dynstr.sh_size = static_cast<Elf32_Word>(strings.size());
  image.insert(image.end(), strings.begin(), strings.end());
  SetSectionHeader(image, dynstr_index, dynstr);

  // Rename dynamic symbol 3 (the second kernel argument) to ".pdi.2". It already lives in the
  // control-code section, which is what the reader requires of a patched symbol.
  const std::uint32_t dynsym_index = FindSection(image, ".dynsym");
  ASSERT_NE(dynsym_index, 0u);
  const Elf32_Shdr dynsym = SectionHeader(image, dynsym_index);
  ASSERT_GE(dynsym.sh_size, 4 * sizeof(Elf32_Sym));
  std::memcpy(
      image.data() + dynsym.sh_offset + 3 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_name),
      &new_name_offset, sizeof(new_name_offset));

  // Point the second argument relocation at it, as a PDI address.
  ASSERT_TRUE(MutateRelocation(image, /*index=*/2, /*sym=*/3, kAddress64));

  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;
  const auto& k = kernels.begin()->second;
  ASSERT_EQ(k.pdis.size(), 2u);
  EXPECT_EQ(k.pdis[0].bytes, SectionContents(image, ".pdi.1"));
  EXPECT_EQ(k.pdis[0].patch_offsets, std::vector<std::uint32_t>{RelocationOffset(image, 0)});
  EXPECT_EQ(k.pdis[1].bytes, SectionContents(image, ".pdi.2"));
  EXPECT_EQ(k.pdis[1].patch_offsets, std::vector<std::uint32_t>{RelocationOffset(image, 2)});
}

TEST(AieElfParse, KeepsIdenticalPdiSectionsSeparate) {
  // Two sites can name sections whose bytes happen to be equal (MLIR-AIR's control code loads the
  // same empty PDI both before and after the design, as two separate sections). Each site's section
  // must still get its own entry and so its own device placement: a kernel that shared one
  // placement across two identical-content sections dispatched correctly once and incorrectly on
  // every dispatch after.
  //
  // Same setup as ParsesTwoPdisInOneKernel, except the renamed section is also pointed at .pdi.1's
  // own bytes (same offset and size), so its content is identical to .pdi.1's by construction
  // rather than merely by the spare section's original content happening to differ.
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  const std::uint32_t alias_index = RenameSpareSection(image, ".pdi.2");
  ASSERT_NE(alias_index, 0u) << "no spare section to rename";
  const std::uint32_t pdi1_index = FindSection(image, ".pdi.1");
  ASSERT_NE(pdi1_index, 0u);
  Elf32_Shdr alias = SectionHeader(image, alias_index);
  const Elf32_Shdr pdi1 = SectionHeader(image, pdi1_index);
  alias.sh_offset = pdi1.sh_offset;
  alias.sh_size = pdi1.sh_size;
  SetSectionHeader(image, alias_index, alias);
  ASSERT_EQ(SectionContents(image, ".pdi.2"), SectionContents(image, ".pdi.1"))
      << "the alias must hold the same bytes, or this test proves nothing";

  // Move and grow .dynstr so it can hold the new symbol name.
  const std::uint32_t dynstr_index = FindSection(image, ".dynstr");
  ASSERT_NE(dynstr_index, 0u);
  Elf32_Shdr dynstr = SectionHeader(image, dynstr_index);
  std::vector<std::uint8_t> strings(image.begin() + dynstr.sh_offset,
                                    image.begin() + dynstr.sh_offset + dynstr.sh_size);
  const std::uint32_t new_name_offset = static_cast<std::uint32_t>(strings.size());
  strings.insert(strings.end(), std::begin(".pdi.2"), std::end(".pdi.2"));
  dynstr.sh_offset = static_cast<Elf32_Off>(image.size());
  dynstr.sh_size = static_cast<Elf32_Word>(strings.size());
  image.insert(image.end(), strings.begin(), strings.end());
  SetSectionHeader(image, dynstr_index, dynstr);

  // Rename dynamic symbol 3 (the second kernel argument) to ".pdi.2". It already lives in the
  // control-code section, which is what the reader requires of a patched symbol.
  const std::uint32_t dynsym_index = FindSection(image, ".dynsym");
  ASSERT_NE(dynsym_index, 0u);
  const Elf32_Shdr dynsym = SectionHeader(image, dynsym_index);
  ASSERT_GE(dynsym.sh_size, 4 * sizeof(Elf32_Sym));
  std::memcpy(
      image.data() + dynsym.sh_offset + 3 * sizeof(Elf32_Sym) + offsetof(Elf32_Sym, st_name),
      &new_name_offset, sizeof(new_name_offset));

  // Point the second argument relocation at it, as a PDI address.
  ASSERT_TRUE(MutateRelocation(image, /*index=*/2, /*sym=*/3, kAddress64));

  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;
  const auto& k = kernels.begin()->second;
  ASSERT_EQ(k.pdis.size(), 2u);
  EXPECT_EQ(k.pdis[0].bytes, SectionContents(image, ".pdi.1"));
  EXPECT_EQ(k.pdis[0].patch_offsets, std::vector<std::uint32_t>{RelocationOffset(image, 0)});
  EXPECT_EQ(k.pdis[1].bytes, SectionContents(image, ".pdi.2"));
  EXPECT_EQ(k.pdis[1].patch_offsets, std::vector<std::uint32_t>{RelocationOffset(image, 2)});
  EXPECT_EQ(k.pdis[0].bytes, k.pdis[1].bytes);
}

TEST(AieElfParse, RejectsAmbiguousPdiName) {
  // PDI sections are not group members, so a relocation names its PDI by section name alone. A
  // name shared by sections with different contents has no right answer, so it is refused rather
  // than resolved by section order. Give a spare section the name ".pdi.1".
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  ASSERT_NE(RenameSpareSection(image, ".pdi.1"), 0u) << "no spare section to rename";
  EXPECT_NE(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS)
      << "an ambiguous PDI name was accepted";
  EXPECT_NE(error.find("ambiguous PDI"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfParse, AcceptsRepeatedPdiNameWithSameContents) {
  // A multi-kernel MLIR-AIR ELF repeats .pdi.1.. for each kernel, with the same bytes. That is
  // unambiguous and has to parse. Give a spare section the name ".pdi.1" and the original's bytes.
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  const std::uint32_t pdi = FindSection(image, ".pdi.1");
  ASSERT_NE(pdi, 0u);
  const std::uint32_t spare = RenameSpareSection(image, ".pdi.1");
  ASSERT_NE(spare, 0u) << "no spare section to rename";
  Elf32_Shdr spare_sh = SectionHeader(image, spare);
  const Elf32_Shdr pdi_sh = SectionHeader(image, pdi);
  spare_sh.sh_offset = pdi_sh.sh_offset;
  spare_sh.sh_size = pdi_sh.sh_size;
  SetSectionHeader(image, spare, spare_sh);

  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;
  const auto& k = kernels.begin()->second;
  ASSERT_EQ(k.pdis.size(), 1u);
  EXPECT_EQ(k.pdis[0].bytes, SectionContents(image, ".pdi.1"));
}

TEST(AieElfParse, RejectsArgumentPatchSiteOutOfRange) {
  // An argument patch site is where the driver folds a buffer address into the control code, three
  // dwords wide. The driver sizes its buffer from the control code, so the reader must refuse a
  // site that would make the patch write past the end.
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;
  const std::size_t ctrl_code_size = kernels.begin()->second.ctrl_code.size();

  // The last dword of the control code: in range on its own, but the two the patch also writes
  // are not.
  auto past_the_end = image;
  ASSERT_TRUE(SetRelocationOffset(past_the_end, /*index=*/1,
                                  static_cast<std::uint32_t>(ctrl_code_size - 4)));
  EXPECT_NE(ParseImage(past_the_end, &kernels, &error), HSA_STATUS_SUCCESS)
      << "an out-of-range argument patch site was accepted";
  EXPECT_NE(error.find("argument patch site"), std::string::npos)
      << "rejected for the wrong reason: " << error;

  // The scheme reads dwords, so a misaligned site is refused too.
  auto misaligned = image;
  ASSERT_TRUE(SetRelocationOffset(misaligned, /*index=*/1, 1));
  EXPECT_NE(ParseImage(misaligned, &kernels, &error), HSA_STATUS_SUCCESS)
      << "a misaligned argument patch site was accepted";
  EXPECT_NE(error.find("argument patch site"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfParse, RejectsArchWithoutFullElf) {
  // aie2 has no full-ELF format, so an aie2 hsaco section cannot carry one.
  const auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  EXPECT_NE(rocr::AMD::aie_elf::Parse(image.data(), image.size(), "aie2", &kernels, &error),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(error.find("no full-ELF format"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfParse, RejectsOsAbiOfAnotherArch) {
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  image[EI_OSABI] = 64;  // aie2ps
  EXPECT_NE(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS);
  EXPECT_NE(error.find("not an aie2p AIE ELF"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfParse, RejectsNonComdatGroup) {
  // A plain group has no known meaning in a full ELF; mlir-aie's packer refuses it too.
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  const std::uint32_t group = FindSection(image, ".group.0");
  ASSERT_NE(group, 0u) << "artifact has no .group.0 section";
  const std::uint32_t no_flags = 0;
  std::memcpy(image.data() + SectionHeader(image, group).sh_offset, &no_flags, sizeof(no_flags));

  EXPECT_NE(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS);
  EXPECT_NE(error.find("not a COMDAT group"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfParse, FindsStringTablesThroughSymbolTableLinks) {
  // The string tables are the ones the symbol tables link to, whatever their names. Renamed in
  // place (same length, so no other .shstrtab offset moves); the result must not change.
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> expected;
  std::string error;
  ASSERT_EQ(ParseImage(image, &expected, &error), HSA_STATUS_SUCCESS) << error;

  const Elf32_Shdr shstrtab = SectionHeader(image, Header(image).e_shstrndx);
  for (const char* name : {".strtab", ".dynstr"}) {
    const std::uint32_t index = FindSection(image, name);
    ASSERT_NE(index, 0u) << "artifact has no " << name << " section";
    // Overwrite the last character: ".strtaX", ".dynstX".
    image[shstrtab.sh_offset + SectionHeader(image, index).sh_name + std::strlen(name) - 1] = 'X';
    ASSERT_EQ(FindSection(image, name), 0u) << name << " is still found by name";
  }

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;
  ASSERT_EQ(kernels.size(), expected.size());
  for (const auto& [name, k] : expected) {
    const auto it = kernels.find(name);
    ASSERT_NE(it, kernels.end()) << "kernel " << name << " not found";
    EXPECT_EQ(it->second.ctrl_code, k.ctrl_code);
    ASSERT_EQ(it->second.pdis.size(), k.pdis.size());
    for (std::size_t p = 0; p < k.pdis.size(); ++p) {
      EXPECT_EQ(it->second.pdis[p].bytes, k.pdis[p].bytes);
      EXPECT_EQ(it->second.pdis[p].patch_offsets, k.pdis[p].patch_offsets);
    }
    EXPECT_EQ(it->second.num_args(), k.num_args());
  }
}

// The scratchpad design: like kElf, but its control code also takes the address of the control
// scratchpad, which the ELF names but does not carry.
constexpr const char* kParamElf = "kernel_full_elf_vsparam/aie.elf";
constexpr const char* kScratchpad = "scratch-pad-ctrl";

/**
 * @brief Finds the .rela.dyn relocations whose symbol has a given name.
 *
 * @param image ELF contents
 * @param name symbol name
 * @param st_size receives the symbol's st_size, if any relocation names it
 * @return the indices of those relocations in .rela.dyn
 */
std::vector<std::uint32_t> RelocationsNaming(const std::vector<std::uint8_t>& image,
                                             const std::string& name, std::uint32_t* st_size) {
  std::vector<std::uint32_t> found;
  const std::uint32_t rela = FindSection(image, ".rela.dyn");
  const std::uint32_t dynsym = FindSection(image, ".dynsym");
  const std::uint32_t dynstr = FindSection(image, ".dynstr");
  if (rela == 0 || dynsym == 0 || dynstr == 0) return found;
  const Elf32_Shdr rela_sh = SectionHeader(image, rela);
  const Elf32_Shdr sym_sh = SectionHeader(image, dynsym);
  const Elf32_Shdr str_sh = SectionHeader(image, dynstr);
  for (std::uint32_t r = 0; r < rela_sh.sh_size / sizeof(Elf32_Rela); ++r) {
    Elf32_Rela rel{};
    std::memcpy(&rel, image.data() + rela_sh.sh_offset + r * sizeof(Elf32_Rela), sizeof(rel));
    Elf32_Sym sym{};
    std::memcpy(&sym, image.data() + sym_sh.sh_offset + ELF32_R_SYM(rel.r_info) * sizeof(Elf32_Sym),
                sizeof(sym));
    const auto* sym_name =
        reinterpret_cast<const char*>(image.data() + str_sh.sh_offset + sym.st_name);
    if (name != sym_name) continue;
    found.push_back(r);
    *st_size = sym.st_size;
  }
  return found;
}

TEST(AieElfParse, ParsesScratchpadAsExternalSymbol) {
  // The scratchpad is neither an argument nor a PDI. The reader records its sites and size for the
  // loader to resolve.
  const auto image = ReadFile(kParamElf);
  if (image.empty()) GTEST_SKIP() << "scratchpad full-ELF artifact not built";

  std::uint32_t st_size = 0;
  const auto relocs = RelocationsNaming(image, kScratchpad, &st_size);
  ASSERT_FALSE(relocs.empty()) << "artifact has no " << kScratchpad << " relocation";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;
  const auto& k = kernels.begin()->second;

  ASSERT_EQ(k.external_symbols.size(), 1u);
  const auto it = k.external_symbols.find(kScratchpad);
  ASSERT_NE(it, k.external_symbols.end());
  EXPECT_EQ(it->second.size, st_size);
  ASSERT_EQ(it->second.sites.size(), relocs.size());
  for (std::size_t i = 0; i < relocs.size(); ++i) {
    EXPECT_EQ(it->second.sites[i].offset, RelocationOffset(image, relocs[i]));
  }
  // Its sites are not PDI sites, and the argument count is unchanged.
  for (const auto& pdi : k.pdis) {
    for (const auto& site : it->second.sites) {
      EXPECT_EQ(std::count(pdi.patch_offsets.begin(), pdi.patch_offsets.end(), site.offset), 0);
    }
  }
  EXPECT_EQ(k.num_args(), 2u);
}

TEST(AieElfParse, RejectsExternalSymbolWithAnotherScheme) {
  // Only Address64 is defined for an external symbol.
  auto image = ReadFile(kParamElf);
  if (image.empty()) GTEST_SKIP() << "scratchpad full-ELF artifact not built";

  std::uint32_t st_size = 0;
  const auto relocs = RelocationsNaming(image, kScratchpad, &st_size);
  ASSERT_FALSE(relocs.empty());
  const std::uint32_t rela = FindSection(image, ".rela.dyn");
  Elf32_Rela rel{};
  std::memcpy(&rel,
              image.data() + SectionHeader(image, rela).sh_offset + relocs[0] * sizeof(Elf32_Rela),
              sizeof(rel));
  // Same symbol, ShimDma48 instead of Address64.
  ASSERT_TRUE(MutateRelocation(image, relocs[0], ELF32_R_SYM(rel.r_info), 5));

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  EXPECT_NE(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS)
      << "an external symbol with a ShimDma48 patch was accepted";
  EXPECT_NE(error.find("external symbol"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfParse, RejectsExternalSymbolPatchSiteOutOfRange) {
  // The 8-byte store must lie inside the control code.
  auto image = ReadFile(kParamElf);
  if (image.empty()) GTEST_SKIP() << "scratchpad full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;
  const std::size_t ctrl_code_size = kernels.begin()->second.ctrl_code.size();

  std::uint32_t st_size = 0;
  const auto relocs = RelocationsNaming(image, kScratchpad, &st_size);
  ASSERT_FALSE(relocs.empty());
  ASSERT_TRUE(
      SetRelocationOffset(image, relocs[0], static_cast<std::uint32_t>(ctrl_code_size - 4)));
  EXPECT_NE(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS)
      << "an out-of-range external symbol patch site was accepted";
  EXPECT_NE(error.find("external symbol patch site"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfPatch, ShimDma48IsAdditive) {
  // Two applications must not equal one: this is why the driver patches a pristine copy.
  std::uint32_t once[3] = {0, 0x1000, 0};
  std::uint32_t twice[3] = {0, 0x1000, 0};
  rocr::AMD::aie_elf::PatchShimDma48(once, 0x2000);
  rocr::AMD::aie_elf::PatchShimDma48(twice, 0x2000);
  rocr::AMD::aie_elf::PatchShimDma48(twice, 0x2000);
  EXPECT_NE(once[1], twice[1]);
}

}  // namespace
