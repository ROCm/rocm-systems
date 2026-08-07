// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0_library_test.cpp
/// @brief Tests the fixed-profile gfx1250 B0-to-A0 shared-library API.

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "support/gfx1250_test_code_object.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CapturedDiagnostic {
  std::string severity;
  std::string kind;
  bool has_guest_offset = false;
  uint64_t guest_offset = 0;
  std::string mnemonic;
  std::string message;
  bool required_work = false;
};

void capture_diagnostic(const rj_gfx1250_b0_to_a0_diagnostic_t *diagnostic, void *user_data) {
  auto *captured = static_cast<std::vector<CapturedDiagnostic> *>(user_data);
  captured->push_back(
      {diagnostic->severity != nullptr ? diagnostic->severity : "",
       diagnostic->kind != nullptr ? diagnostic->kind : "", diagnostic->has_guest_offset != 0,
       diagnostic->guest_offset, diagnostic->mnemonic != nullptr ? diagnostic->mnemonic : "",
       diagnostic->message != nullptr ? diagnostic->message : "", diagnostic->required_work != 0});
}

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

std::optional<std::span<const uint8_t>> elf_section(std::span<const uint8_t> image,
                                                    std::string_view wanted_name) {
  if (image.size() < sizeof(rocjitsu::Elf64_Ehdr))
    return std::nullopt;
  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  const uint64_t section_bytes =
      static_cast<uint64_t>(header.e_shnum) * sizeof(rocjitsu::Elf64_Shdr);
  if (header.e_shentsize != sizeof(rocjitsu::Elf64_Shdr) || header.e_shoff > image.size() ||
      section_bytes > image.size() - header.e_shoff || header.e_shstrndx >= header.e_shnum)
    return std::nullopt;
  auto section = [&](size_t index) {
    rocjitsu::Elf64_Shdr value{};
    std::memcpy(&value, image.data() + header.e_shoff + index * sizeof(value), sizeof(value));
    return value;
  };
  const rocjitsu::Elf64_Shdr names = section(header.e_shstrndx);
  if (names.sh_offset > image.size() || names.sh_size > image.size() - names.sh_offset)
    return std::nullopt;
  for (size_t index = 0; index < header.e_shnum; ++index) {
    const rocjitsu::Elf64_Shdr candidate = section(index);
    if (candidate.sh_name >= names.sh_size || candidate.sh_offset > image.size() ||
        candidate.sh_size > image.size() - candidate.sh_offset)
      continue;
    const char *name =
        reinterpret_cast<const char *>(image.data() + names.sh_offset + candidate.sh_name);
    const size_t available = names.sh_size - candidate.sh_name;
    const void *terminator = std::memchr(name, 0, available);
    if (terminator != nullptr && std::string_view(name) == wanted_name)
      return image.subspan(candidate.sh_offset, candidate.sh_size);
  }
  return std::nullopt;
}
#endif

TEST(Gfx1250B0ToA0Library, RejectsInvalidArgumentsAndClearsOutputs) {
  auto *output = reinterpret_cast<uint8_t *>(0x1);
  size_t output_size = 1;
  rj_gfx1250_b0_to_a0_translation_info_t info{1, 1};
  EXPECT_EQ(
      rj_gfx1250_b0_to_a0_translate(nullptr, 0, &output, &output_size, &info, nullptr, nullptr),
      ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  EXPECT_EQ(info.source_code_object_id, 0u);
  EXPECT_EQ(info.changed_instruction_count, 0u);

  constexpr std::array<uint8_t, 64> kNotElf = {'N', 'O', 'T', 'E', 'L', 'F'};
  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate(kNotElf.data(), kNotElf.size(), &output, &output_size,
                                          &info, nullptr, nullptr),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  EXPECT_NE(info.source_code_object_id, 0u);
  EXPECT_EQ(info.changed_instruction_count, 0u);

  rj_gfx1250_b0_to_a0_free(nullptr);
}

TEST(Gfx1250B0ToA0Library, ReportsInvalidCodeObjectDiagnostic) {
  constexpr std::array<uint8_t, 64> kNotElf = {'N', 'O', 'T', 'E', 'L', 'F'};
  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  std::vector<CapturedDiagnostic> diagnostics;

  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate(kNotElf.data(), kNotElf.size(), &output, &output_size,
                                          &info, capture_diagnostic, &diagnostics),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  ASSERT_FALSE(diagnostics.empty());
  const auto matching = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return item.severity == "error" && item.kind == "input-invalid-code-object" &&
           item.message.find("valid gfx1250") != std::string::npos;
  });
  EXPECT_NE(matching, diagnostics.end());
}

TEST(Gfx1250B0ToA0Library, ReportsTranslatorDiagnosticsAndRequiredWork) {
  constexpr auto conversion = rocjitsu::gfx1250::build_vds(
      rocjitsu::gfx1250::kDsStore2addrStride64B64Vds,
      {.offset0 = 255, .offset1 = 255, .addr = 20, .data0 = 30, .data1 = 40});
  constexpr uint32_t kEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text = {conversion[0], conversion[1], kEndpgm};
  const auto source = rocjitsu::test_support::make_gfx1250_code_object(text);
  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  std::vector<CapturedDiagnostic> diagnostics;

  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate(source.data(), source.size(), &output, &output_size,
                                          &info, capture_diagnostic, &diagnostics),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  ASSERT_GE(diagnostics.size(), 2u);

  const auto primary = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return !item.required_work && item.severity == "error" &&
           item.kind == "translator-expand-failed";
  });
  ASSERT_NE(primary, diagnostics.end());
  EXPECT_TRUE(primary->has_guest_offset);
  EXPECT_EQ(primary->guest_offset, 0u);
  EXPECT_EQ(primary->mnemonic, "ds_store_2addr_stride64_b64");

  const auto required = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return item.required_work && item.kind == "translator-expand-failed";
  });
  ASSERT_NE(required, diagnostics.end());
  EXPECT_TRUE(required->has_guest_offset);
  EXPECT_EQ(required->guest_offset, 0u);
  EXPECT_EQ(required->mnemonic, "ds_store_2addr_stride64_b64");
  EXPECT_NE(required->message.find("scratch-address lowering"), std::string::npos);
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
  ASSERT_EQ(rj_gfx1250_b0_to_a0_translate(source.data(), source.size(), &output, &output_size,
                                          &info, nullptr, nullptr),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(info.source_code_object_id, source_identity(source));
  EXPECT_GT(info.changed_instruction_count, 0u);
  constexpr std::array<uint8_t, 4> kElfMagic = {0x7f, 'E', 'L', 'F'};
  ASSERT_GE(output_size, kElfMagic.size());
  EXPECT_TRUE(std::equal(kElfMagic.begin(), kElfMagic.end(), output));

  const std::span<const uint8_t> translated(output, output_size);
  const auto source_addresses = elf_section(source, ".debug_addr");
  const auto target_addresses = elf_section(translated, ".debug_addr");
  const auto source_lines = elf_section(source, ".debug_line");
  const auto target_lines = elf_section(translated, ".debug_line");
  const auto source_locations = elf_section(source, ".debug_loclists");
  const auto target_locations = elf_section(translated, ".debug_loclists");
  const auto source_frames = elf_section(source, ".debug_frame");
  const auto target_frames = elf_section(translated, ".debug_frame");
  ASSERT_TRUE(source_addresses);
  ASSERT_TRUE(target_addresses);
  ASSERT_TRUE(source_lines);
  ASSERT_TRUE(target_lines);
  ASSERT_TRUE(source_locations);
  ASSERT_TRUE(target_locations);
  ASSERT_TRUE(source_frames);
  ASSERT_TRUE(target_frames);
  EXPECT_FALSE(std::ranges::equal(*source_addresses, *target_addresses));
  EXPECT_FALSE(std::ranges::equal(*source_lines, *target_lines));
  EXPECT_FALSE(std::ranges::equal(*source_locations, *target_locations));
  EXPECT_FALSE(std::ranges::equal(*source_frames, *target_frames));

  rj_gfx1250_b0_to_a0_free(output);
}
#endif

} // namespace
