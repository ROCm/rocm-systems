// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/gpu_mi400_runtime_decoder.hpp"

#include "addc/mca/decoder.hpp"
#include "addc/mca/policy.hpp"
#include "addc/mca/registers.hpp"
#include "addc/pipeline/section_data.hpp"
#include "gpu_decode_helpers.hpp"

namespace addc::pipeline
{

std::vector<PipelineEvent> decode_gpu_mi400_runtime(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AmdGpuMi400RuntimeSection& s, const DecodeContext& context,
    std::string* error_out)
{

    if (context.decode_mca == nullptr)
    {
        if (error_out != nullptr)
        {
            *error_out =
                "GpuMi400RuntimeDecoder: no MCA operation in DecodeContext";
        }
        return {};
    }

    std::vector<PipelineEvent> events;
    const bool has_ei = !s.error_info_structures.empty();
    const auto* ei = has_ei ? s.error_info_structures.data() : nullptr;
    for (std::size_t ci = 0U; ci < s.contexts.size(); ++ci)
    {
        const GpuMi400RuntimeContext& mc = s.contexts[ci];
        if (!mc.bank)
        {
            continue;
        }
        const McaBankDump& bank = *mc.bank;

        addc::mca::McaRegisters regs{bank.status, bank.ipid,  bank.synd,
                                     bank.addr,   bank.misc0, bank.misc1};
        if (!regs.status.val)
        {
            continue;
        }

        auto decoded = context.decode_mca(
            s.platform, addc::mca::BankCapture{regs, bank.transaddr});
        if (!decoded)
        {
            continue;
        }

        if (addc::mca::is_bad_page_retirement(
                s.platform, s.descriptor_flags.error_threshold_exceeded, regs))
        {
            decoded->decoded.error_type = "BadPageRetirement";
            decoded->decoded.description =
                "Bad page retirement threshold exceeded";
            decoded->decoded.severity = "Fatal";
            decoded->decoded.instance = std::nullopt;
            decoded->decoded.afid = {47873};
            decoded->event_report = addc::mca::to_json(decoded->decoded);
        }

        PipelineEvent ev;
        ev.metadata = gpu::make_gpu_metadata(s, context, ci, "Runtime MCA",
                                             "mca", events.size());
        ev.platform = gpu::make_gpu_platform(s, has_ei, ei);
        ev.platform["gpu"]["check_info"] =
            has_ei ? gpu::make_mi400_check_info(*ei)
                   : nlohmann::ordered_json(nullptr);
        ev.event_report = decoded->event_report;
        ev.data_array = to_data_array(bank);
        ev.validation = decoded->validation;
        ev.analysis["afid"] = decoded->decoded.afid;
        ev.firmware = gpu::make_mi400_firmware_json(s);
        ev.ppr = decoded->ppr;

        events.push_back(std::move(ev));
    }

    return events;
}

} // namespace addc::pipeline
