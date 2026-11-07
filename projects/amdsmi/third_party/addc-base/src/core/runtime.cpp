// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "runtime.hpp"

#include "addc/cper/parser.hpp"
#include "addc/detail/format.hpp"
#include "addc/pipeline/pipeline.hpp"

namespace addc::detail
{

OperationResult<nlohmann::json> BaseRuntime::parse(
    std::span<const uint8_t> bytes) const
{
    cper::ParseFailure failure;
    auto ir = cper::normalize_parser_json(
        cper::parse_native_json(bytes, options, &failure));
    if (ir.is_null())
    {
        std::string message{cper::parse_error_message(failure.code)};
        if (failure.section_index)
        {
            message += addc::format(" (section {})", *failure.section_index);
        }
        return OperationResult<nlohmann::json>::failure({
            .code = OperationCode::ParseError,
            .message = std::move(message),
            .source = "cper",
            .offset = failure.offset,
        });
    }
    return OperationResult<nlohmann::json>::success(std::move(ir));
}

OperationResult<pipeline::PipelineOutput> BaseRuntime::analyze(
    cper::CperRecord record, std::string_view filename,
    std::string_view tool_version) const
{
    std::string message;
    std::optional<cper::ParseFailure> parse_failure;
    auto output = pipeline::run_pipeline(
        std::move(record), options, filename, tool_version, &message,
        &parse_failure);
    if (!output)
    {
        return OperationResult<pipeline::PipelineOutput>::failure({
            .code = parse_failure ? OperationCode::ParseError
                                  : OperationCode::DecodeError,
            .message =
                message.empty() ? "CPER decode failed" : std::move(message),
            .source = parse_failure ? "cper" : "pipeline",
            .offset = parse_failure
                          ? std::optional<std::size_t>{parse_failure->offset}
                          : std::nullopt,
        });
    }
    return OperationResult<pipeline::PipelineOutput>::success(
        std::move(*output));
}

OperationResult<pipeline::PipelineOutput> BaseRuntime::decode(
    std::span<const uint8_t> bytes, std::string_view filename,
    std::string_view tool_version) const
{
    cper::ParseFailure parse_failure;
    auto record =
        cper::CperRecord::parse_detailed(bytes, options, &parse_failure);
    if (!record)
    {
        std::string message{cper::parse_error_message(parse_failure.code)};
        if (parse_failure.section_index)
        {
            message +=
                addc::format(" (section {})", *parse_failure.section_index);
        }
        return OperationResult<pipeline::PipelineOutput>::failure({
            .code = OperationCode::ParseError,
            .message = std::move(message),
            .source = "cper",
            .offset = parse_failure.offset,
        });
    }
    return analyze(std::move(*record), filename, tool_version);
}

BaseRuntime build_base_runtime(DecodeOptions options)
{
    return BaseRuntime{.options = options};
}

} // namespace addc::detail
