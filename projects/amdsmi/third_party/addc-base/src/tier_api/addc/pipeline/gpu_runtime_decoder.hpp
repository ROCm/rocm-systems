// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/pipeline/decode_context.hpp"
#include "addc/pipeline/section_data.hpp"

namespace addc::pipeline
{

[[nodiscard]] std::vector<PipelineEvent> decode_gpu_runtime(
    const SectionDescriptor& descriptor, const AmdGpuRuntimeSection& section,
    const DecodeContext& context, std::string* error_out = nullptr);

} // namespace addc::pipeline
