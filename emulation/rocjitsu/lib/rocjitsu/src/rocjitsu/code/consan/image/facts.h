// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"

#include <optional>

namespace rocjitsu::consan {

// The original descriptor's file offset, scoped by the exact original image.
using EntryKey = uint64_t;
inline constexpr uint32_t kImageEntryLimit = 1024;

enum class EntryDisposition : uint32_t { Selected = 1, Native = 2 };

class ImageEntryFacts {
public:
  EntryKey key = 0;
  EntryDisposition disposition = EntryDisposition::Native;
  EntryFacts binding;
  uint32_t original_sgprs = 0;
  uint32_t original_vgprs = 0;
  // Each selected entry keeps the existing guest-local entry ID zero.
  std::optional<Manifest> selected;
  bool operator==(const ImageEntryFacts &) const = default;
};

/// @brief Owning facts for one coherent prepared image and explicit selection.
/// @details Entries are strictly ordered by original descriptor offset. Common
/// fields also describe native rows, which deliberately have no detector facts.
/// Selected entries must share one qualified scope; mixed scopes are rejected.
class ImageFacts {
public:
  ImageIdentity original;
  ImageIdentity prepared;
  Digest selection_profile{};
  uint32_t target = 0;
  uint32_t wave_size = 0;
  uint32_t code_object_abi = 0;
  uint64_t original_text_bytes = 0;
  uint64_t prepared_text_bytes = 0;
  std::vector<ImageEntryFacts> entries;
  bool operator==(const ImageFacts &) const = default;
};

/// @brief Hash the canonical selected-key sequence with the detector options.
/// @details Input keys must be strictly ascending; validation enforces this.
[[nodiscard]] Digest
selection_profile_digest(std::span<const EntryKey> selected_keys,
                         ScopeKind scope = ScopeKind::NativeLdsNoSynchronization);

/// @brief Return the common selected scope; mixed or missing scopes are invalid.
[[nodiscard]] std::optional<ScopeKind> selected_scope(const ImageFacts &image);

[[nodiscard]] const ImageEntryFacts *find_image_entry(const ImageFacts &image, EntryKey key);
[[nodiscard]] const Manifest *selected_manifest(const ImageFacts &image, EntryKey key);

/// @brief Wrap existing single-entry facts with explicitly supplied original counts.
[[nodiscard]] ImageFacts single_entry_image_facts(const Manifest &manifest, uint32_t original_sgprs,
                                                  uint32_t original_vgprs);

[[nodiscard]] Result validate_image_facts(const ImageFacts &image, ContractError *error = nullptr,
                                          const ContractLimits &limits = {});
[[nodiscard]] FailureOr<std::vector<uint8_t>> encode_image_facts(const ImageFacts &image,
                                                                 ContractError *error = nullptr,
                                                                 const ContractLimits &limits = {});
[[nodiscard]] FailureOr<ImageFacts> decode_image_facts(std::span<const uint8_t> bytes,
                                                       ContractError *error = nullptr,
                                                       const ContractLimits &limits = {});

/// @brief Exact common identities plus the independently configured selection.
[[nodiscard]] Result match_image_facts(const ImageFacts &image, std::span<const uint8_t> original,
                                       std::span<const uint8_t> prepared,
                                       const Digest &expected_selection_profile,
                                       ContractError *error = nullptr,
                                       const ContractLimits &limits = {});

} // namespace rocjitsu::consan
