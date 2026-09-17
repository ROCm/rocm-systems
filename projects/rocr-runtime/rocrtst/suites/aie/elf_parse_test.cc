/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <cstring>
#include <elf.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core/inc/amd_aie_elf.h"

namespace {

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

// ---------------------------------------------------------------------------
// In-memory ELF surgery
//
// The tests below build malformed variants of the real artifact rather than hand-writing an ELF,
// so what they feed the parser differs from a known-good image in exactly the one way named.
// ---------------------------------------------------------------------------

Elf32_Ehdr Header(const std::vector<std::uint8_t>& image) {
  Elf32_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  return ehdr;
}

Elf32_Shdr SectionHeader(const std::vector<std::uint8_t>& image, std::uint32_t index) {
  const Elf32_Ehdr ehdr = Header(image);
  Elf32_Shdr shdr{};
  std::memcpy(&shdr, image.data() + ehdr.e_shoff + index * sizeof(Elf32_Shdr), sizeof(shdr));
  return shdr;
}

void SetSectionHeader(std::vector<std::uint8_t>& image, std::uint32_t index,
                      const Elf32_Shdr& shdr) {
  const Elf32_Ehdr ehdr = Header(image);
  std::memcpy(image.data() + ehdr.e_shoff + index * sizeof(Elf32_Shdr), &shdr, sizeof(shdr));
}

// Index of the section named `name`, or 0 (the NULL section, never a match) if there is none.
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

// Rewrite the relocation at `index` in .rela.dyn to reference symbol `sym` with relocation type
// `type`. The entries are fixed size, so this is an in-place edit that shifts nothing.
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

// Rewrite the r_offset (the patch site) of the relocation at `index`.
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

hsa_status_t ParseImage(const std::vector<std::uint8_t>& image,
                        std::map<std::string, rocr::AMD::aie_elf::Kernel>* kernels,
                        std::string* error) {
  return rocr::AMD::aie_elf::Parse(image.data(), image.size(), kernels, error);
}

// The relocation type the reader reads as "write a 64-bit address here", used for PDI symbols.
constexpr std::uint32_t kAddress64 = 8;

TEST(AieElfParse, ParsesVectorScalarAdd) {
  const auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(rocr::AMD::aie_elf::Parse(image.data(), image.size(), &kernels, &error),
            HSA_STATUS_SUCCESS) << error;
  ASSERT_FALSE(kernels.empty());

  const auto& k = kernels.begin()->second;
  EXPECT_EQ(k.ctrl_code.size(), 316u);   // .ctrltext.0, measured on this branch
  EXPECT_EQ(k.pdi.size(), 2992u);        // .pdi.1
  EXPECT_TRUE(k.has_pdi_patch);
  EXPECT_NE(k.pdi_patch_offset, 0u);
  EXPECT_GT(k.num_args(), 0u);
}

TEST(AieElfParse, RejectsGarbage) {
  const std::vector<std::uint8_t> garbage(512, 0xA5);
  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  EXPECT_NE(rocr::AMD::aie_elf::Parse(garbage.data(), garbage.size(), &kernels, &error),
            HSA_STATUS_SUCCESS);
  EXPECT_FALSE(error.empty());
}

TEST(AieElfParse, RejectsTruncated) {
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";
  image.resize(image.size() / 2);

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  EXPECT_NE(rocr::AMD::aie_elf::Parse(image.data(), image.size(), &kernels, &error),
            HSA_STATUS_SUCCESS);
}

TEST(AieElfParse, RejectsSecondPdiPatchSite) {
  // A Kernel carries one PDI patch offset. If an ELF asked for two, keeping only one would leave
  // the other load_pdi pointing at a placeholder and the dispatch would run against a bogus PDI
  // address -- so the reader has to refuse rather than pick one. Turn the first argument
  // relocation into a second PDI relocation to provoke it: symbol 1 is .pdi.1, matching the
  // relocation the ELF already has at index 0.
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  // Unmodified, it parses: whatever the mutation below provokes is the mutation's doing.
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  ASSERT_TRUE(MutateRelocation(image, /*index=*/1, /*sym=*/1, kAddress64));
  // Assert on the reason, not just that it failed: this ELF encodes the patch scheme in r_info,
  // but an ABI-version-1 ELF encodes it in the addend, where rewriting r_info alone would leave
  // the scheme unchanged and trip a different check.
  EXPECT_NE(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS)
      << "a second PDI patch site was accepted";
  EXPECT_NE(error.find("more than one PDI patch site"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfParse, RejectsMoreThanOnePdiPerKernel) {
  // A kernel loads one PDI. Two would mean the Kernel's single pdi field silently dropped one.
  // Reaching this needs two *distinct* PDI sections, which the artifact does not have -- with one
  // section the duplicate-patch-site check above fires first -- so build the second one: rename a
  // section the reader ignores to ".pdi.2" and point a second PDI relocation at a symbol of that
  // name. The new symbol name does not fit the existing .dynstr, so .dynstr is relocated to the
  // end of the image and grown; nothing refers to its file offset but its own section header.
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS) << error;

  // A section the reader never looks at, renamed in place (the new name is shorter, so no other
  // .shstrtab offset moves).
  const std::uint32_t spare = FindSection(image, ".note.xrt.UID");
  ASSERT_NE(spare, 0u) << "no spare section to rename";
  const Elf32_Ehdr ehdr = Header(image);
  const Elf32_Shdr shstrtab = SectionHeader(image, ehdr.e_shstrndx);
  const Elf32_Shdr spare_sh = SectionHeader(image, spare);
  std::memcpy(image.data() + shstrtab.sh_offset + spare_sh.sh_name, ".pdi.2", sizeof(".pdi.2"));
  ASSERT_EQ(FindSection(image, ".pdi.2"), spare);

  // Relocate and grow .dynstr so it can hold the new symbol name.
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
  std::memcpy(image.data() + dynsym.sh_offset + 3 * sizeof(Elf32_Sym) +
                  offsetof(Elf32_Sym, st_name),
              &new_name_offset, sizeof(new_name_offset));

  // Point the second argument relocation at it, as a PDI address.
  ASSERT_TRUE(MutateRelocation(image, /*index=*/2, /*sym=*/3, kAddress64));

  EXPECT_NE(ParseImage(image, &kernels, &error), HSA_STATUS_SUCCESS)
      << "a second PDI was accepted";
  EXPECT_NE(error.find("more than one PDI per kernel"), std::string::npos)
      << "rejected for the wrong reason: " << error;
}

TEST(AieElfParse, RejectsArgumentPatchSiteOutOfRange) {
  // An argument patch site is where the driver folds a buffer address into the control code, three
  // dwords wide. An offset past the end would make it write past the buffer it allocated from
  // ctrl_code_size, so the reader has to bound it -- the PDI patch site has been bounded all
  // along, the argument sites were not.
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

TEST(AieElfPatch, ShimDma48IsAdditive) {
  // Two applications must not equal one: this is why §4 copies from pristine.
  std::uint32_t once[3] = {0, 0x1000, 0};
  std::uint32_t twice[3] = {0, 0x1000, 0};
  rocr::AMD::aie_elf::PatchShimDma48(once, 0x2000);
  rocr::AMD::aie_elf::PatchShimDma48(twice, 0x2000);
  rocr::AMD::aie_elf::PatchShimDma48(twice, 0x2000);
  EXPECT_NE(once[1], twice[1]);
}

}  // namespace
