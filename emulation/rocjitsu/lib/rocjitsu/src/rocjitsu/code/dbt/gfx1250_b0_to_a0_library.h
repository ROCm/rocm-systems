// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {

/// @brief Provenance collected while translating one exact code-object ELF.
struct Gfx1250B0ToA0TranslationInfo {
  /// 64-bit FNV-1a identity of the complete source ELF byte sequence.
  uint64_t source_code_object_id = 0;
  /// Number of source instructions whose translated encoding changed.
  size_t changed_instruction_count = 0;
};

/// @brief Borrowed diagnostic strings, valid only during the synchronous callback.
/// @details Required-work items are emitted separately with the same kind and
/// source location as their primary diagnostic.
struct Gfx1250B0ToA0DiagnosticView {
  const char *severity;
  const char *kind;
  bool has_guest_offset;
  uint64_t guest_offset;
  const char *mnemonic;
  const char *message;
  bool required_work;
};

/// @brief Synchronous diagnostic sink. Exceptions terminate the calling process.
using Gfx1250B0ToA0DiagnosticCallback = void (*)(const Gfx1250B0ToA0DiagnosticView *diagnostic,
                                                 void *user_data);

/// @brief Translate a standalone gfx1250 B0 code object for execution on gfx1250 A0.
/// @details This private interface is shared by the hot-swap hook and the
/// pretranslation tool. Both outputs are cleared before validation; nonempty
/// input receives a source identity before parsing. Diagnostics include warnings
/// on success and required-work items. Internal allocation failures terminate.
/// @pre source must not refer to storage owned by translated.
/// @returns SUCCESS with owned output bytes, INVALID_ARGUMENT for empty input,
/// INVALID_CODE_OBJECT for rejected input, or ERROR for an internal failure.
[[nodiscard]] RJ_API_EXPORT rj_status_t
translate_gfx1250_b0_to_a0(std::span<const uint8_t> source, std::vector<uint8_t> &translated,
                           Gfx1250B0ToA0TranslationInfo &info,
                           Gfx1250B0ToA0DiagnosticCallback diagnostic_callback = nullptr,
                           void *user_data = nullptr) noexcept;

} // namespace rocjitsu
