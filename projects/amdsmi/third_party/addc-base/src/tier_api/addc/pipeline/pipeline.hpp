// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"
#include "addc/decoder/dbglog/decoder.hpp"
#include "addc/decoder/wdt/decoder.hpp"
#include "addc/mca/decoder.hpp"
#include "addc/pipeline/output.hpp"

#include <nlohmann/json.hpp>

#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace addc::pipeline
{

/// Decode an already parsed CPER IR with one complete decoder composition.
/// The Base facade builds the composition; the pipeline never
/// supplies or guesses missing registries.
[[nodiscard]] std::optional<PipelineOutput> run_pipeline(
    const nlohmann::json& ir,
    std::string_view filename, std::string_view tool_version,
    std::string* error_out = nullptr,
    addc::mca::McaDecodeFunction mca_decode =
        addc::mca::try_decode_project_event,
    addc::decoder::DbgLogDecodeFunction dbglog_decode =
        addc::decoder::decode_base_dbglog,
    addc::decoder::WdtDecodeFunction wdt_decode =
        addc::decoder::decode_base_wdt);

/// Decode a typed raw CPER record without routing the analysis path through
/// the public JSON IR. The record is taken by value because section parsers may
/// append tolerant-recovery entries while materializing typed section data.
[[nodiscard]] std::optional<PipelineOutput> run_pipeline(
    addc::cper::CperRecord record, const addc::DecodeOptions& options,
    std::string_view filename, std::string_view tool_version,
    std::string* error_out = nullptr,
    std::optional<addc::cper::ParseFailure>* parse_failure_out = nullptr,
    addc::mca::McaDecodeFunction mca_decode =
        addc::mca::try_decode_project_event,
    addc::decoder::DbgLogDecodeFunction dbglog_decode =
        addc::decoder::decode_base_dbglog,
    addc::decoder::WdtDecodeFunction wdt_decode =
        addc::decoder::decode_base_wdt);

} // namespace addc::pipeline
