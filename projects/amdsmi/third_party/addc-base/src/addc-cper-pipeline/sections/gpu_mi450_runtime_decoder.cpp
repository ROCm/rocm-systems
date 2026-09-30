// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/gpu_mi450_runtime_decoder.hpp"

#include "addc/mca/decoder.hpp"
#include "addc/mca/registers.hpp"
#include "addc/pipeline/section_data.hpp"
#include "gpu_decode_helpers.hpp"

namespace addc::pipeline
{

std::vector<PipelineEvent> decode_gpu_mi450_runtime(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AmdGpuMi450RuntimeSection& s, const DecodeContext& context,
    std::string* error_out)
{

    if (context.decode_mca == nullptr)
    {
        if (error_out != nullptr)
        {
            *error_out =
                "GpuMi450RuntimeDecoder: no MCA operation in DecodeContext";
        }
        return {};
    }

    std::vector<PipelineEvent> events;
    const bool has_ei = !s.error_info_structures.empty();
    const auto* ei = has_ei ? s.error_info_structures.data() : nullptr;
    for (std::size_t ci = 0U; ci < s.contexts.size(); ++ci)
    {
        const GpuMi450RuntimeContext& mc = s.contexts[ci];
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

        const auto decoded = context.decode_mca(
            s.platform, addc::mca::BankCapture{regs, bank.transaddr});
        if (!decoded)
        {
            continue;
        }

        PipelineEvent ev;
        ev.metadata = gpu::make_gpu_metadata(s, context, ci, "Runtime MCA",
                                             "mca", events.size());
        ev.platform = gpu::make_gpu_platform(s, has_ei, ei);
        ev.event_report = decoded->event_report;
        ev.data_array = to_data_array(bank);
        ev.validation = decoded->validation;
        ev.analysis["afid"] = decoded->decoded.afid;
        ev.firmware = gpu::make_firmware_json(s);
        ev.ppr = decoded->ppr;

        events.push_back(std::move(ev));
    }

    return events;
}

} // namespace addc::pipeline
