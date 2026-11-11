// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/pipeline/output.hpp"
#include "addc/report_summary.hpp"

#include <nlohmann/json_fwd.hpp>

namespace addc::base
{

// The single resolved Base report consumed by every Base API edge. The wrapper
// makes the resolve boundary explicit without exposing it as installed ABI.
struct ResolvedReport
{
    pipeline::PipelineOutput output;
};

[[nodiscard]] ResolvedReport resolve_report(
    pipeline::PipelineOutput output);
[[nodiscard]] nlohmann::ordered_json to_json(const ResolvedReport& report);
[[nodiscard]] CperErrorSummary summarize_events(
    const pipeline::PipelineOutput& output);
[[nodiscard]] CperErrorSummary summarize(const ResolvedReport& report);
[[nodiscard]] nlohmann::ordered_json summary_to_json(
    const CperErrorSummary& summary);

} // namespace addc::base
