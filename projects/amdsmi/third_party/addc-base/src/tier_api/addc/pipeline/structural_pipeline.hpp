// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/structural.hpp"
#include "addc/pipeline/decode_context.hpp"

namespace addc::pipeline
{

/// Shared structural orchestration primitives for fixed tier compositions.
/// They expose values and normal functions only; section dispatch remains
/// owned explicitly by each tier.
[[nodiscard]] std::optional<cper::StructuralRecord> prepare_structural_record(
    cper::CperRecord& record, const addc::DecodeOptions& options,
    std::optional<cper::ParseFailure>* parse_failure_out = nullptr,
    std::string* error_out = nullptr);

[[nodiscard]] PipelineOutput make_pipeline_output(
    const nlohmann::json& header, std::string_view filename,
    std::string_view tool_version);

[[nodiscard]] DecodeContext make_decode_context(
    const nlohmann::json& header, PipelineOutput& output,
    addc::mca::McaDecodeFunction mca_decode =
        addc::mca::try_decode_project_event,
    addc::decoder::DbgLogDecodeFunction dbglog_decode =
        addc::decoder::decode_base_dbglog,
    addc::decoder::WdtDecodeFunction wdt_decode =
        addc::decoder::decode_base_wdt);

void append_parser_recovery(PipelineOutput& output,
                            const nlohmann::json& recovery);

} // namespace addc::pipeline
