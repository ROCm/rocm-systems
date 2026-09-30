// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/lib.hpp"

#include "addc/base_report.hpp"
#include "addc/pipeline/output.hpp"
#include "addc/product.hpp"
#include "../core/runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

namespace addc
{

namespace
{

constexpr std::string_view TOOL_VERSION = ADDC_VERSION;

std::optional<std::vector<uint8_t>> readBinaryFile(std::string_view path,
                                                   std::string* error_out)
{
    // The file API deliberately opens the caller-selected path; the public
    // contract requires privileged callers to enforce their own allowed root.
    // codeql[cpp/path-injection]
    std::ifstream file{std::string{path}, std::ios::binary | std::ios::ate};
    if (!file)
    {
        if (error_out != nullptr)
        {
            *error_out = std::string{"Failed to open file: "} +
                         std::string{path};
        }
        return std::nullopt;
    }

    const auto pos = file.tellg();
    if (pos < 0)
    {
        if (error_out != nullptr)
        {
            *error_out = std::string{"Failed to determine size: "} +
                         std::string{path};
        }
        return std::nullopt;
    }
    const auto size = static_cast<std::size_t>(pos);
    constexpr std::size_t kMaxFileSize =
        static_cast<std::size_t>(256U * 1024U * 1024U);
    if (size > kMaxFileSize)
    {
        if (error_out != nullptr)
        {
            *error_out = std::string{"File too large: "} + std::string{path};
        }
        return std::nullopt;
    }

    file.seekg(0);
    std::vector<uint8_t> bytes(size);
    if (!file.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(size)))
    {
        if (error_out != nullptr)
        {
            *error_out = std::string{"Failed to read file: "} +
                         std::string{path};
        }
        return std::nullopt;
    }
    return bytes;
}

detail::OperationResult<base::ResolvedReport> decodeReport(
    std::span<const uint8_t> bytes, std::string_view filename,
    const DecodeOptions& options)
{
    auto decoded =
        detail::build_base_runtime(options).decode(bytes, filename, TOOL_VERSION);
    if (!decoded)
    {
        return detail::OperationResult<base::ResolvedReport>::failure(
            decoded.error ? std::move(*decoded.error)
                          : detail::OperationError{
                                detail::OperationCode::InternalError,
                                "CPER analysis failed", "runtime", std::nullopt});
    }
    return detail::OperationResult<base::ResolvedReport>::success(
        base::resolve_report(std::move(*decoded.value)));
}

} // anonymous namespace

std::optional<nlohmann::ordered_json> analyze_cper_json(
    std::span<const uint8_t> bytes, std::string_view filename,
    std::string* error_out)
{
    return analyze_cper_json(bytes, filename, DecodeOptions{}, error_out);
}

std::optional<nlohmann::ordered_json> analyze_cper_json(
    std::span<const uint8_t> bytes, std::string_view filename,
    const DecodeOptions& options, std::string* error_out)
{
    auto result = decodeReport(bytes, filename, options);
    if (!result)
    {
        if ((error_out != nullptr) && result.error)
        {
            *error_out = result.error->message;
        }
        return std::nullopt;
    }

    return base::to_json(*result.value);
}

std::optional<std::string> analyze_cper(std::span<const uint8_t> bytes,
                                        std::string_view filename,
                                        std::string* error_out)
{
    return analyze_cper(bytes, filename, DecodeOptions{}, error_out);
}

std::optional<std::string> analyze_cper(
    std::span<const uint8_t> bytes, std::string_view filename,
    const DecodeOptions& options, std::string* error_out)
{
    const auto j = analyze_cper_json(bytes, filename, options, error_out);
    if (!j)
    {
        return std::nullopt;
    }
    return j->dump(2);
}

std::optional<nlohmann::ordered_json> analyze_cper_file_json(
    std::string_view path, std::string* error_out)
{
    return analyze_cper_file_json(path, DecodeOptions{}, error_out);
}

std::optional<nlohmann::ordered_json> analyze_cper_file_json(
    std::string_view path, const DecodeOptions& options, std::string* error_out)
{
    const auto bytes = readBinaryFile(path, error_out);
    if (!bytes)
    {
        return std::nullopt;
    }

    const auto fname = std::filesystem::path{path}.filename().string();
    return analyze_cper_json(std::span<const uint8_t>{*bytes}, fname, options,
                             error_out);
}

std::optional<std::string> analyze_cper_file(std::string_view path,
                                             std::string* error_out)
{
    return analyze_cper_file(path, DecodeOptions{}, error_out);
}

std::optional<std::string> analyze_cper_file(
    std::string_view path, const DecodeOptions& options, std::string* error_out)
{
    const auto j = analyze_cper_file_json(path, options, error_out);
    if (!j)
    {
        return std::nullopt;
    }
    return j->dump(2);
}

std::optional<CperErrorSummary> get_error_summary_entries(
    std::span<const uint8_t> bytes, std::string_view filename,
    std::string* error_out)
{
    auto report = decodeReport(bytes, filename, DecodeOptions{});
    if (!report)
    {
        if ((error_out != nullptr) && report.error)
        {
            *error_out = report.error->message;
        }
        return std::nullopt;
    }
    return base::summarize(*report.value);
}

std::optional<CperErrorSummary> get_error_summary_file_entries(
    std::string_view path, std::string* error_out)
{
    const auto bytes = readBinaryFile(path, error_out);
    if (!bytes)
    {
        return std::nullopt;
    }
    return get_error_summary_entries(
        *bytes, std::filesystem::path{path}.filename().string(), error_out);
}

std::optional<nlohmann::ordered_json> get_error_summary_json(
    std::span<const uint8_t> bytes, std::string_view filename,
    std::string* error_out)
{
    auto summary = get_error_summary_entries(bytes, filename, error_out);
    if (!summary)
    {
        return std::nullopt;
    }

    return base::summary_to_json(*summary);
}

std::optional<nlohmann::ordered_json> get_error_summary_file_json(
    std::string_view path, std::string* error_out)
{
    auto summary = get_error_summary_file_entries(path, error_out);
    if (!summary)
    {
        return std::nullopt;
    }

    return base::summary_to_json(*summary);
}

std::optional<std::string> get_error_summary(std::span<const uint8_t> bytes,
                                             std::string_view filename,
                                             std::string* error_out)
{
    auto summary = get_error_summary_json(bytes, filename, error_out);
    if (!summary)
    {
        return std::nullopt;
    }

    return summary->dump(2);
}

std::optional<std::string> get_error_summary_file(std::string_view path,
                                                  std::string* error_out)
{
    auto summary = get_error_summary_file_json(path, error_out);
    if (!summary)
    {
        return std::nullopt;
    }

    return summary->dump(2);
}

std::string mca_supported_projects_json()
{
    std::vector<std::string_view> names;
    for (const auto& definition : product::compiled_products())
    {
        if (definition.capabilities.mca)
        {
            names.push_back(definition.identity.canonical_name);
        }
    }
    std::sort(names.begin(), names.end());
    nlohmann::ordered_json out;
    out["feature"] = "mca";
    out["available"] = true;
    out["projects"] = nlohmann::ordered_json::array();
    for (const auto& name : names)
    {
        out["projects"].push_back({{"name", name}});
    }
    return out.dump();
}

std::optional<nlohmann::json> parse_cper(std::span<const uint8_t> bytes)
{
    return parse_cper(bytes, DecodeOptions{});
}

std::optional<nlohmann::json> parse_cper(std::span<const uint8_t> bytes,
                                         const DecodeOptions& options)
{
    auto result = detail::build_base_runtime(options).parse(bytes);
    if (!result)
    {
        return std::nullopt;
    }
    return std::move(*result.value);
}

std::optional<nlohmann::json> parse_cper_file(std::string_view path,
                                              std::string* error_out)
{
    return parse_cper_file(path, DecodeOptions{}, error_out);
}

std::optional<nlohmann::json> parse_cper_file(
    std::string_view path, const DecodeOptions& options, std::string* error_out)
{
    const auto bytes = readBinaryFile(path, error_out);
    if (!bytes)
    {
        return std::nullopt;
    }
    auto result = parse_cper(std::span<const uint8_t>{*bytes}, options);
    if (!result && (error_out != nullptr))
    {
        *error_out = "Failed to parse CPER record";
    }
    return result;
}

} // namespace addc
