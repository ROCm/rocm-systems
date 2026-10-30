// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/pcie_decoder.hpp"

#include "addc/common.hpp"
#include "addc/decoder/pcie/decoder.hpp"
#include "addc/detail/format.hpp"
#include "addc/pipeline/section_data.hpp"

namespace addc::pipeline
{

namespace
{

[[nodiscard]] nlohmann::ordered_json makeMetadata(
    const PcieSection& s, const DecodeContext& context, std::size_t event_index)
{
    nlohmann::ordered_json m;
    m["section_index"] = context.section_index;
    m["context_structure_index"] = 0;
    m["context_type"] = "PCIe";
    m["event_class"] = "pcie";
    m["event_index"] = event_index;
    m["timestamp"] = context.timestamp;
    m["addc_version"] = {{"name", s.platform_name}, {"value", s.addc_version}};
    return m;
}

[[nodiscard]] nlohmann::ordered_json makePlatform(const PcieSection& s)
{
    nlohmann::ordered_json p;
    p["fru"] = s.fru;
    p["fru_id"] = s.fru_id;
    p["cpu"] = nullptr;
    p["gpu"] = nullptr;
    return p;
}

} // namespace

std::vector<PipelineEvent> decode_pcie_section(
    const SectionDescriptor& descriptor, const PcieSection& s,
    const DecodeContext& context, std::string* /*error_out*/)
{
    auto decoded =
        addc::decoder::decode_pcie(s.raw_body, descriptor.revision_major);

    PipelineEvent ev;
    ev.metadata = makeMetadata(s, context, 0U);
    ev.platform = makePlatform(s);
    ev.event_report = std::move(decoded.event_report);
    ev.data_array = std::move(decoded.data_array);
    ev.validation = std::move(decoded.validation);
    ev.analysis["afid"] = std::move(decoded.afid);

    std::vector<PipelineEvent> events;
    events.push_back(std::move(ev));
    return events;
}

} // namespace addc::pipeline
