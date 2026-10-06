// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/export.h"

#include "addc/common.hpp"
#include "addc/json.hpp"
#include "addc/options.hpp"
#include "addc/report_summary.hpp"
#include "addc/schema_version.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace addc
{

/// Source-level C++ convenience API version. Binary consumers should use the
/// stable C ABI and addc_api_version().
inline constexpr uint32_t kApiVersion = 2u;

[[nodiscard]] ADDC_API std::optional<ordered_json_value> analyze_cper_json(
    std::span<const uint8_t> bytes, std::string_view filename = "",
    std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<ordered_json_value> analyze_cper_json(
    std::span<const uint8_t> bytes, std::string_view filename,
    const DecodeOptions& options, std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<std::string> analyze_cper(
    std::span<const uint8_t> bytes, std::string_view filename = "",
    std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<std::string> analyze_cper(
    std::span<const uint8_t> bytes, std::string_view filename,
    const DecodeOptions& options, std::string* error_out = nullptr);

/// All *_file functions open the exact path supplied by the caller. They do
/// not authorize or confine filesystem access. Callers that receive paths from
/// an untrusted source must enforce their own allowed-root policy first, or
/// read the authorized file themselves and use the byte/span overloads above.
[[nodiscard]] ADDC_API std::optional<ordered_json_value>
    analyze_cper_file_json(std::string_view path,
                           std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<ordered_json_value>
    analyze_cper_file_json(std::string_view path, const DecodeOptions& options,
                           std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<std::string> analyze_cper_file(
    std::string_view path, std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<std::string> analyze_cper_file(
    std::string_view path, const DecodeOptions& options,
    std::string* error_out = nullptr);

/// Decode a CPER and return its AFID/FRU summary as typed C++ data.
/// One entry is returned for every AFID emitted by a decoded event, including
/// kUnclassifiedAfid when the event cannot be classified more specifically.
[[nodiscard]] ADDC_API std::optional<CperErrorSummary>
    get_error_summary_entries(std::span<const uint8_t> bytes,
                              std::string_view filename = "",
                              std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<CperErrorSummary>
    get_error_summary_file_entries(std::string_view path,
                                   std::string* error_out = nullptr);

/// JSON-array form of get_error_summary_entries().
[[nodiscard]] ADDC_API std::optional<ordered_json_value>
    get_error_summary_json(std::span<const uint8_t> bytes,
                           std::string_view filename = "",
                           std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<ordered_json_value>
    get_error_summary_file_json(std::string_view path,
                                std::string* error_out = nullptr);

/// Pretty-printed JSON-array form of get_error_summary_entries().
[[nodiscard]] ADDC_API std::optional<std::string> get_error_summary(
    std::span<const uint8_t> bytes, std::string_view filename = "",
    std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<std::string> get_error_summary_file(
    std::string_view path, std::string* error_out = nullptr);

// JSON object: {"feature":"mca","available":true,
//               "projects":[{"name":"..."}, ...]}.
// addc-base reports its supported project names only.
[[nodiscard]] ADDC_API std::string mca_supported_projects_json();

[[nodiscard]] ADDC_API std::optional<json_value> parse_cper(
    std::span<const uint8_t> bytes);

[[nodiscard]] ADDC_API std::optional<json_value> parse_cper(
    std::span<const uint8_t> bytes, const DecodeOptions& options);

[[nodiscard]] ADDC_API std::optional<json_value> parse_cper_file(
    std::string_view path, std::string* error_out = nullptr);

[[nodiscard]] ADDC_API std::optional<json_value> parse_cper_file(
    std::string_view path, const DecodeOptions& options,
    std::string* error_out = nullptr);

} // namespace addc
