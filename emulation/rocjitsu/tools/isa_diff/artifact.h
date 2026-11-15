// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file artifact.h
/// @brief Binary-mode artifact ingestion: AMDGPU code objects and raw instruction words.
///
/// @details This is the frontend seam. P0 supports two binary inputs:
///   - an AMDGPU ELF code object (HSACO) plus a kernel symbol name;
///   - a raw little-endian instruction stream (one kernel starting at offset 0).
/// The `.s` frontend through LLVM MC is a separate plugin task. Whatever produces
/// an Artifact, the executed object is the original machine code: bytes, instruction
/// boundaries, literals and PCs are kept verbatim.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu::isa_diff {

struct Artifact {
  std::string label;  ///< Path or synthetic label.
  std::string kind;   ///< "code_object" or "raw_words".
  std::string sha256; ///< Of the complete input bytes.
  std::string target = "gfx950";
  std::vector<uint8_t> text; ///< Whole .text section (raw words: the stream itself).
  uint64_t text_vaddr = 0;   ///< Address of text[0]; executed PC = text_vaddr + offset.
  uint64_t kernel_begin = 0; ///< Section-relative kernel entry.
  uint64_t kernel_end = 0;   ///< Section-relative exclusive end of the kernel symbol.
  std::string kernel_name;
  /// 64-byte kernel descriptor (empty for raw words).
  std::vector<uint8_t> descriptor;
  /// SHA-256 of the code object's note (AMDHSA metadata) sections; empty for raw words.
  std::string metadata_sha256;
};

/// @brief Load a code object (ELF magic) or a raw word stream.
/// @param kernel Kernel symbol name; may be empty when the object has exactly one kernel.
/// @returns false and sets @p error on failure.
[[nodiscard]] bool load_artifact(const std::string &path, const std::string &kernel, Artifact &out,
                                 std::string &error);

/// @brief Wrap an in-memory raw word stream (tests, replay).
[[nodiscard]] Artifact artifact_from_words(std::span<const uint32_t> words, std::string label,
                                           uint64_t text_vaddr = 0);

/// @brief Wrap an in-memory code object image.
[[nodiscard]] bool artifact_from_code_object(std::span<const uint8_t> image,
                                             const std::string &label, const std::string &kernel,
                                             Artifact &out, std::string &error);

/// @brief Lowercase hex SHA-256 of a byte span.
[[nodiscard]] std::string sha256_hex(std::span<const uint8_t> bytes);
[[nodiscard]] std::string sha256_hex(std::string_view text);

/// @brief Named descriptor fields that differ, ignoring the layout-only entry offset.
/// @param layout_only_change Set when only kernel_code_entry_byte_offset differs.
[[nodiscard]] std::vector<std::string>
descriptor_semantic_differences(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b,
                                bool &layout_only_change);

} // namespace rocjitsu::isa_diff
