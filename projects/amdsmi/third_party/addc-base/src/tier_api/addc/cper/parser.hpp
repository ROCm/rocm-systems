// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <span>

namespace addc::cper
{

/// Native producer entrypoint. It intentionally returns the established
/// compatibility JSON rather than typed analysis data.
[[nodiscard]] nlohmann::json parse_native_json(
    std::span<const uint8_t> buf,
    const addc::DecodeOptions& options = {},
    ParseFailure* failure_out = nullptr);

/// Backend-neutral canonicalization seam. Wave 3 preserves the native value
/// exactly; a future backend must normalize through this same function.
[[nodiscard]] nlohmann::json normalize_parser_json(
    const nlohmann::json& producer_json);

/// Parse a raw CPER binary buffer into a JSON IR object compatible with
/// libcper's externally observable output schema (header, sectionDescriptors,
/// sections).
///
/// Returns a JSON null value if the record cannot be parsed.
[[nodiscard]] nlohmann::json cper_to_ir(
    std::span<const uint8_t> buf);

/// Parse with an explicit recovery policy. Tolerant mode is the default in the
/// compatibility overload above; strict mode rejects every condition that
/// would otherwise require a recorded repair.
[[nodiscard]] nlohmann::json cper_to_ir(
    std::span<const uint8_t> buf, const addc::DecodeOptions& options);

/// Detailed parser form used by native product boundaries.  failure_out is
/// populated only when parsing fails.
[[nodiscard]] nlohmann::json cper_to_ir(
    std::span<const uint8_t> buf, const addc::DecodeOptions& options,
    ParseFailure* failure_out);

/// Quick validity check: returns true iff buf starts with a valid CPER
/// signature, with behavior compatible with libcper consumers.
[[nodiscard]] bool cper_is_valid(std::span<const uint8_t> buf) noexcept;

} // namespace addc::cper
