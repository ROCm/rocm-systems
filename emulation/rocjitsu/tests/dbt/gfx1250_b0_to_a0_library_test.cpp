// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0_library_test.cpp
/// @brief Tests the fixed-profile gfx1250 B0-to-A0 shared-library API.

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

#ifdef GFX1250_B0_TO_A0_FIXTURE
uint64_t source_identity(const std::vector<uint8_t> &bytes) {
  constexpr uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t identity = kOffsetBasis;
  for (uint8_t byte : bytes) {
    identity ^= byte;
    identity *= kPrime;
  }
  return identity;
}

struct FunctionExtent {
  std::string name;
  uint64_t value = 0;
  uint64_t size = 0;
};

/// @brief Read the local `STT_FUNC` extents from `.symtab`, sorted by address.
std::vector<FunctionExtent> local_function_extents(const std::vector<uint8_t> &image) {
  using namespace rocjitsu;
  std::vector<FunctionExtent> extents;
  if (image.size() < sizeof(Elf64_Ehdr))
    return extents;
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  if (ehdr.e_shentsize != sizeof(Elf64_Shdr) ||
      ehdr.e_shoff + static_cast<uint64_t>(ehdr.e_shnum) * sizeof(Elf64_Shdr) > image.size())
    return extents;

  std::vector<Elf64_Shdr> sections(ehdr.e_shnum);
  std::memcpy(sections.data(), image.data() + ehdr.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  for (const Elf64_Shdr &section : sections) {
    if (section.sh_type != SHT_SYMTAB || section.sh_entsize != sizeof(Elf64_Sym) ||
        section.sh_link >= sections.size())
      continue;
    const Elf64_Shdr &strings = sections[section.sh_link];
    for (uint64_t offset = 0; offset + sizeof(Elf64_Sym) <= section.sh_size;
         offset += sizeof(Elf64_Sym)) {
      Elf64_Sym symbol{};
      std::memcpy(&symbol, image.data() + section.sh_offset + offset, sizeof(symbol));
      if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeFunc ||
          elf_symbol_bind(symbol.st_info) != kElfSymbolBindLocal || symbol.st_size == 0)
        continue;
      const auto *name =
          reinterpret_cast<const char *>(image.data() + strings.sh_offset + symbol.st_name);
      extents.push_back({.name = name, .value = symbol.st_value, .size = symbol.st_size});
    }
  }
  std::ranges::sort(extents, {}, &FunctionExtent::value);
  return extents;
}
#endif

TEST(Gfx1250B0ToA0Library, RejectsInvalidArgumentsAndClearsOutputs) {
  auto *output = reinterpret_cast<uint8_t *>(0x1);
  size_t output_size = 1;
  rj_gfx1250_b0_to_a0_translation_info_t info{1, 1};
  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate_with_info(nullptr, 0, &output, &output_size, &info),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  EXPECT_EQ(info.source_code_object_id, 0u);
  EXPECT_EQ(info.changed_instruction_count, 0u);

  constexpr std::array<uint8_t, 64> kNotElf = {'N', 'O', 'T', 'E', 'L', 'F'};
  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate_with_info(kNotElf.data(), kNotElf.size(), &output,
                                                    &output_size, &info),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  EXPECT_NE(info.source_code_object_id, 0u);
  EXPECT_EQ(info.changed_instruction_count, 0u);

  rj_gfx1250_b0_to_a0_free(nullptr);
}

#ifdef GFX1250_B0_TO_A0_FIXTURE
TEST(Gfx1250B0ToA0Library, TranslatesRealGfx1250CodeObject) {
  std::ifstream input(GFX1250_B0_TO_A0_FIXTURE, std::ios::binary);
  ASSERT_TRUE(input) << GFX1250_B0_TO_A0_FIXTURE;
  const std::vector<uint8_t> source((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
  ASSERT_GE(source.size(), 4u);

  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  ASSERT_EQ(rj_gfx1250_b0_to_a0_translate_with_info(source.data(), source.size(), &output,
                                                    &output_size, &info),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(info.source_code_object_id, source_identity(source));
  EXPECT_GT(info.changed_instruction_count, 0u);
  constexpr std::array<uint8_t, 4> kElfMagic = {0x7f, 'E', 'L', 'F'};
  ASSERT_GE(output_size, kElfMagic.size());
  EXPECT_TRUE(std::equal(kElfMagic.begin(), kElfMagic.end(), output));

  rj_gfx1250_b0_to_a0_free(output);
}
#endif

#ifdef GFX1250_DEVICE_FUNCTION_FIXTURE
/// A code object whose kernels call out-of-line device functions must translate, and a second pass
/// must reproduce it byte for byte.
///
/// @details The fixture is the only artifact carrying non-kernel `STT_FUNC` bodies: clang reaches
/// them with `s_get_pc_i64` + `s_add_nc_u64` + `s_swap_pc_i64` rather than a direct call, and emits
/// one `LOCAL` copy per caller. Byte-identical re-translation is what proves the function-coverage
/// refusal does not fire on rocjitsu's own output, whose inter-body padding is `s_nop` rather than
/// the zeros the decoder skips.
TEST(Gfx1250B0ToA0Library, TranslatesAndReproducesDeviceFunctionCallers) {
  std::ifstream input(GFX1250_DEVICE_FUNCTION_FIXTURE, std::ios::binary);
  ASSERT_TRUE(input) << GFX1250_DEVICE_FUNCTION_FIXTURE;
  const std::vector<uint8_t> source((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
  ASSERT_GE(source.size(), 4u);

  // A future toolchain that inlines the helpers away would leave this test passing while covering
  // nothing at all, so require the bodies the fixture exists to carry. The names are mangled and
  // the helpers are `noinline`, so their absence means the shape under test is gone.
  for (const std::string_view symbol :
       {"_Z11device_leafff", "_Z14device_nonleafff", "_Z18device_tail_callerff"}) {
    ASSERT_NE(std::search(source.begin(), source.end(), symbol.begin(), symbol.end()), source.end())
        << "fixture no longer carries an out-of-line device function named " << symbol;
  }

  uint8_t *first = nullptr;
  size_t first_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  ASSERT_EQ(rj_gfx1250_b0_to_a0_translate_with_info(source.data(), source.size(), &first,
                                                    &first_size, &info),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(first, nullptr);
  const std::vector<uint8_t> first_pass(first, first + first_size);
  rj_gfx1250_b0_to_a0_free(first);

  uint8_t *second = nullptr;
  size_t second_size = 0;
  ASSERT_EQ(
      rj_gfx1250_b0_to_a0_translate(first_pass.data(), first_pass.size(), &second, &second_size),
      ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(second, nullptr);
  const std::vector<uint8_t> second_pass(second, second + second_size);
  rj_gfx1250_b0_to_a0_free(second);

  EXPECT_EQ(second_pass, first_pass)
      << "re-translating a device-function object must not drop or duplicate a callee body";

  // A symbol's value and its extent are answered from opposite ends of the same source offset, and
  // those two offsets coincide wherever one callee abuts the next -- which this fixture does twice,
  // once per clone of the leaf. Pinning the relationships rather than the numbers keeps the check
  // meaningful if a translation rule later grows one of the bodies.
  const auto source_extents = local_function_extents(source);
  const auto target_extents = local_function_extents(first_pass);
  ASSERT_GE(source_extents.size(), 2u) << "fixture must carry more than one out-of-line callee";
  ASSERT_EQ(target_extents.size(), source_extents.size())
      << "translation must neither drop nor invent a callee body";

  for (size_t i = 0; i + 1 < target_extents.size(); ++i) {
    EXPECT_EQ(target_extents[i].name, source_extents[i].name)
        << "callee bodies must keep their relative order";
    EXPECT_LE(target_extents[i].value + target_extents[i].size, target_extents[i + 1].value)
        << target_extents[i].name << " overlaps " << target_extents[i + 1].name;
    const bool abutted_in_source =
        source_extents[i].value + source_extents[i].size == source_extents[i + 1].value;
    if (abutted_in_source) {
      EXPECT_EQ(target_extents[i].value + target_extents[i].size, target_extents[i + 1].value)
          << "an extent measured through the following body's start would span the gap between "
          << target_extents[i].name << " and " << target_extents[i + 1].name;
    }
  }
}
#endif

} // namespace
