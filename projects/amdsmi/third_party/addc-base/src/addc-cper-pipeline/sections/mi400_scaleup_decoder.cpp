// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/mi400_scaleup_decoder.hpp"

#include "addc-common/base64.hpp"
#include "addc/common.hpp"
#include "gpu_decode_helpers.hpp"

namespace addc::pipeline
{

namespace
{

constexpr uint16_t kUaloeEvent = 1U;
constexpr uint16_t kUaloeTelemetry = 2U;

} // namespace

std::vector<PipelineEvent> decode_mi400_scaleup(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AmdMi400ScaleUpSection& s, const DecodeContext& context,
    [[maybe_unused]] std::string* error_out)
{
    const bool has_ei = !s.error_info_structures.empty();
    const auto* ei = has_ei ? s.error_info_structures.data() : nullptr;

    std::vector<PipelineEvent> events;
    for (std::size_t ci = 0U; ci < s.contexts.size(); ++ci)
    {
        const ScaleUpContext& sc = s.contexts[ci];
        if (sc.context_type != kUaloeEvent &&
            sc.context_type != kUaloeTelemetry)
        {
            continue;
        }
        const bool is_event = sc.context_type == kUaloeEvent;

        PipelineEvent ev;
        ev.metadata = gpu::make_gpu_metadata(
            s, context, ci, is_event ? "UALoE Event" : "UALoE Telemetry",
            is_event ? "scaleup_event" : "scaleup_telemetry", events.size());
        ev.platform = gpu::make_gpu_platform(s, has_ei, ei);
        if (auto& gpu = ev.platform["gpu"]; gpu.is_object())
        {
            gpu.erase("redfish_event_log_timestamp");
        }
        // Base reports the structure only; the buffer is not interpreted.
        ev.event_report = {
            {"relative_to_err", nullptr},
            {"timestamp", nullptr},
            {"details", nullptr},
        };
        ev.data_array["raw_data"] = addc::detail::base64_encode(sc.raw_data);
        ev.firmware = gpu::make_mi400_firmware_json(s);
        if (ev.firmware.is_object() || !s.event_data_format_version.empty())
        {
            ev.firmware["event_data_format_version"] =
                s.event_data_format_version.empty()
                    ? nlohmann::ordered_json(nullptr)
                    : nlohmann::ordered_json(s.event_data_format_version);
        }
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
