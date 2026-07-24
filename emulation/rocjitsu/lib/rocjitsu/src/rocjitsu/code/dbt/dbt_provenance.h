// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_provenance.h
/// @brief Provenance note stamped on a DBT translation output for idempotence.

#pragma once

#include "rocjitsu/code/dbt/processor_revision.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {

/// @brief Non-allocated ELF section carrying the DBT translation provenance note.
///
/// @details The binary translator stamps this section into every successful
/// output so a later load can recognize an already-translated object and avoid
/// re-translating it (idempotence). It is intentionally distinct from any
/// producer-side "input authorship" marker: this note attests what the
/// translator *did*, not what the compiler *authored*.
inline constexpr std::string_view kDbtProvenanceSectionName = ".rocjitsu.dbt_provenance";

/// @brief Decoded contents of a @ref kDbtProvenanceSectionName payload.
struct DbtProvenance {
  ProcessorRevision input_revision = ProcessorRevision::Unspecified;  ///< Source silicon revision.
  ProcessorRevision output_revision = ProcessorRevision::Unspecified; ///< Target silicon revision.
};

/// @brief Serialize a provenance note for @ref kDbtProvenanceSectionName.
///
/// @details Returns an empty vector if @p provenance names an Unspecified
/// revision (there is nothing meaningful to attest, and callers must not stamp a
/// blank note).
[[nodiscard]] std::vector<uint8_t> serialize_dbt_provenance(const DbtProvenance &provenance);

/// @brief Parse a @ref kDbtProvenanceSectionName payload.
///
/// @details Returns std::nullopt for a malformed note (bad magic/version/size or
/// an unrecognized revision encoding). A well-formed note always decodes to two
/// recognized revisions; the caller decides whether the *direction* is the one
/// it expects.
[[nodiscard]] std::optional<DbtProvenance> parse_dbt_provenance(std::span<const uint8_t> bytes);

} // namespace rocjitsu
