// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/mi450_scaleup_decoder.hpp"

#include "addc-common/base64.hpp"
#include "addc/common.hpp"
#include "gpu_decode_helpers.hpp"

namespace addc::pipeline
{

namespace
{

[[nodiscard]] std::string_view contextTypeName(uint16_t context_type)
{
    switch (context_type)
    {
        case 1U:
            return "UALoE Event";
        case 2U:
            return "UALoE Telemetry";
        default:
            return "Unknown";
    }
}

[[nodiscard]] std::string_view eventClass(uint16_t context_type)
{
    switch (context_type)
    {
        case 1U:
            return "scaleup_event";
        case 2U:
            return "scaleup_telemetry";
        default:
            return "scaleup_context";
    }
}

} // namespace

std::vector<PipelineEvent> decode_mi450_scaleup(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AmdMi450ScaleUpSection& s, const DecodeContext& context,
    [[maybe_unused]] std::string* error_out)
{
    const bool has_ei = !s.error_info_structures.empty();
    const auto* ei = has_ei ? s.error_info_structures.data() : nullptr;

    std::vector<PipelineEvent> events;
    for (std::size_t ci = 0U; ci < s.contexts.size(); ++ci)
    {
        const ScaleUpContext& sc = s.contexts[ci];

        PipelineEvent ev;
        ev.metadata = gpu::make_gpu_metadata(
            s, context, ci, contextTypeName(sc.context_type),
            eventClass(sc.context_type), events.size());
        ev.platform = gpu::make_gpu_platform(s, has_ei, ei);
        // Base reports the structure only; the buffer is not interpreted.
        ev.event_report = {
            {"relative_to_err", nullptr},
            {"timestamp", nullptr},
            {"details", nullptr},
        };
        ev.data_array["raw_data"] = addc::detail::base64_encode(sc.raw_data);
        ev.firmware = gpu::make_mi450_firmware_json(s);
        ev.validation["struct"] = true;
        ev.validation["bank"] = false;
        ev.validation["corrupted"] = nullptr;
        ev.validation["possible_non_mca_fatal"] = nullptr;
        ev.validation["mtl_raw_data"] = false;
        ev.analysis["afid"] = addc::afid_list(addc::kAfidSentinel);
        events.push_back(std::move(ev));
    }
    return events;
}

} // namespace addc::pipeline
