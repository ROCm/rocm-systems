// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"
#include "addc/options.hpp"
#include "addc/pipeline/output.hpp"
#include "result.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace addc::detail
{

/// The one complete decoder composition used by every Base facade. This is a
/// private implementation type: Base consumers select parser behavior but
/// cannot replace individual decoders.
struct BaseRuntime
{
    DecodeOptions options;

    [[nodiscard]] OperationResult<nlohmann::json> parse(
        std::span<const uint8_t> bytes) const;

    [[nodiscard]] OperationResult<pipeline::PipelineOutput> analyze(
        cper::CperRecord record, std::string_view filename,
        std::string_view tool_version) const;

    [[nodiscard]] OperationResult<pipeline::PipelineOutput> decode(
        std::span<const uint8_t> bytes, std::string_view filename,
        std::string_view tool_version) const;
};

[[nodiscard]] BaseRuntime build_base_runtime(DecodeOptions options = {});

} // namespace addc::detail
