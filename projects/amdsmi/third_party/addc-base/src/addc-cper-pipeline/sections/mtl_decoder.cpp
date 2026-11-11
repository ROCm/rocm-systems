// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/mtl_decoder.hpp"

#include "addc/common.hpp"
#include "addc/detail/format.hpp"
#include "addc/pipeline/section_data.hpp"

namespace addc::pipeline
{

std::vector<PipelineEvent> decode_mtl(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AmdEpycMtlSection& s, const DecodeContext& context,
    [[maybe_unused]] std::string* error_out)
{

    PipelineEvent ev;
    ev.metadata["section_index"] = context.section_index;
    ev.metadata["context_structure_index"] = 0;
    ev.metadata["context_type"] = "MTL";
    ev.metadata["event_class"] = "mpx_tracelog";
    ev.metadata["event_index"] = 0;
    ev.metadata["timestamp"] = context.timestamp;
    ev.metadata["addc_version"] = {{"name", s.platform_name},
                                   {"value", s.addc_version}};

    ev.platform["fru"] = s.fru;
    ev.platform["fru_id"] = s.fru_id;
    ev.platform["cpu"] = nullptr;
    ev.platform["gpu"] = nullptr;

    nlohmann::ordered_json log_header_j;
    log_header_j["logging_enabled"] =
        addc::format("0x{:08x}", s.logging_enabled);
    log_header_j["tail_offset"] = addc::format("0x{:x}", s.tail_offset);
    log_header_j["entries"] = std::to_string(s.entries);
    log_header_j["log_version"] = addc::format("0x{:08x}", s.log_version);
    log_header_j["usec_timestamp"] = addc::format("0x{:08x}", s.usec_timestamp);

    ev.event_report["log_header"] = std::move(log_header_j);

    ev.data_array = nullptr;
    ev.analysis["afid"] = addc::afid_list(addc::kAfidSentinel);

    return {std::move(ev)};
}

} // namespace addc::pipeline
