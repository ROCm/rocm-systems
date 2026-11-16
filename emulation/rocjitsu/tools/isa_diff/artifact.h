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
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::isa_diff {

/// A comment-only inline-asm block of a `.s` input (`;;#ASMSTART` ... `;;#ASMEND` around
/// comment lines). It assembles to nothing, so the LLVM plugin marks its position with a
/// symbol and records its text in a non-loaded section of the object; the code object
/// carries both (see llvm/llvm_frontend.h).
struct InlineAsmBlock {
  uint64_t offset = 0; ///< Section-relative: the block runs before the instruction here.
  std::string text;    ///< The comment lines, newline-separated, whitespace-trimmed.
  /// A label separates the block from the next instruction, so the block ends the
  /// preceding code instead of starting the labelled block.
  bool before_label = false;
  bool operator==(const InlineAsmBlock &) const = default;
};

/// Symbol-name prefix and section of the inline-asm markers written by the LLVM plugin.
inline constexpr std::string_view kInlineAsmMarkerPrefix = "__rj_isa_diff_asm.";
inline constexpr std::string_view kInlineAsmSection = ".rj_isa_diff_inline_asm";

/// MODE at entry of a callable function without other information: LLVM's default
/// function mode (round to nearest, f32/f16/f64 denormals kept, DX10_CLAMP and IEEE on).
inline constexpr uint32_t kDefaultFunctionMode = (3u << 4) | (3u << 6) | (1u << 8) | (1u << 9);

/// One relocation that patches bytes of .text (static .rela.text kept by --emit-relocs or
/// a relocatable object, or a dynamic relocation whose target lies in .text).
struct TextRelocation {
  uint64_t offset = 0; ///< Section-relative byte offset into .text.
  uint32_t type = 0;   ///< R_AMDGPU_* type number.
  std::string symbol;  ///< Target symbol name, empty when none.
  int64_t addend = 0;
  bool operator==(const TextRelocation &) const = default;
};
[[nodiscard]] std::string describe_relocation(const TextRelocation &r);

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
  /// Relocations patching .text, sorted by offset (handoff 4: kept, never normalised away).
  std::vector<TextRelocation> text_relocations;
  /// True when the AMDHSA metadata note parsed and this kernel's entry was found.
  bool metadata_parsed = false;
  /// This kernel's metadata map flattened to key -> value (".args[0].size" -> "8").
  std::map<std::string, std::string> kernel_metadata;
  /// Top-level metadata fields other than amdhsa.kernels ("amdhsa.target", ...).
  std::map<std::string, std::string> global_metadata;
  uint64_t load_ns = 0; ///< Time spent reading and parsing the input (handoff 11).
  /// A callable function (an STT_FUNC symbol without a kernel descriptor): entered by a
  /// call and left by `s_setpc_b64 s[30:31]`, under the AMDGPU C calling convention.
  bool callable = false;
  /// Callable functions: VGPRs v0..v{n-1} that carry the return value (at most 32).
  unsigned return_vgprs = 32;
  /// Callable functions: MODE at entry, which the caller provides.
  uint32_t function_mode = kDefaultFunctionMode;
  /// Comment-only inline-asm blocks inside the kernel/function range, in address order.
  std::vector<InlineAsmBlock> inline_asm;
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
/// MODE register at dispatch for this kernel, from its descriptor's compute_pgm_rsrc1
/// (gfx9 layout: round [3:0], denorm [7:4], DX10_CLAMP 8, IEEE 9, DEBUG 11, FP16_OVFL 23),
/// as RocJITsu's command processor sets it. Without a descriptor (raw words) the
/// assembler defaults are used: round-to-nearest, f32 denormals flushed, f16/f64
/// denormals kept, DX10_CLAMP and IEEE on.
[[nodiscard]] uint32_t initial_mode(const std::vector<uint8_t> &descriptor);
/// MODE at entry: the descriptor's for a kernel, function_mode for a callable function.
[[nodiscard]] uint32_t initial_mode(const Artifact &artifact);

[[nodiscard]] std::string sha256_hex(std::span<const uint8_t> bytes);
[[nodiscard]] std::string sha256_hex(std::string_view text);

/// @brief Named descriptor fields that differ, ignoring the layout-only entry offset.
/// @param layout_only_change Set when only kernel_code_entry_byte_offset differs.
[[nodiscard]] std::vector<std::string>
descriptor_semantic_differences(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b,
                                bool &layout_only_change);

} // namespace rocjitsu::isa_diff
