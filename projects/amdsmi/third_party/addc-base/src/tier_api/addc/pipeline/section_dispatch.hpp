// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/structural.hpp"
#include "addc/pipeline/decode_context.hpp"

namespace addc::pipeline
{

struct SectionDecodeResult
{
    bool handled = false;
    std::vector<PipelineEvent> events;
};

/// Base's fixed compile-time section composition. The payload alternative is
/// the dispatch key; there is no mutable registration or priority ordering.
[[nodiscard]] SectionDecodeResult decode_base_section(
    const SectionDescriptor& descriptor, const cper::SectionPayload& payload,
    const DecodeContext& context, std::string* error_out = nullptr);

} // namespace addc::pipeline
