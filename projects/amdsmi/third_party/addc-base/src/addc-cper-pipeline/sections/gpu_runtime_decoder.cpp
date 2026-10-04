// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "addc/pipeline/gpu_runtime_decoder.hpp"

#include "addc/detail/format.hpp"
#include "addc/mca/decoder.hpp"
#include "addc/mca/policy.hpp"
#include "addc/mca/registers.hpp"
#include "addc/pipeline/section_data.hpp"
#include "gpu_decode_helpers.hpp"

namespace addc::pipeline
{

std::vector<PipelineEvent> decode_gpu_runtime(
    [[maybe_unused]] const SectionDescriptor& descriptor,
    const AmdGpuRuntimeSection& s, const DecodeContext& context,
    std::string* error_out)
{

    if (context.decode_mca == nullptr)
    {
        if (error_out != nullptr)
        {
            *error_out = "GpuRuntimeDecoder: no MCA operation in DecodeContext";
        }
        return {};
    }

    std::vector<PipelineEvent> events;

    for (std::size_t ci = 0U; ci < s.contexts.size(); ++ci)
    {
        const GpuContextStructure& gs = s.contexts[ci];

        if (gs.context_type == GpuContextType::Mca ||
            gs.context_type == GpuContextType::Crashdump)
        {
            auto bank_opt = gpu::decode_runtime_mca_bank(gs.raw_bytes);
            if (!bank_opt)
            {
                continue;
            }

            const McaBankDump& bank = *bank_opt;
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
                    s.platform, s.descriptor_flags.error_threshold_exceeded,
                    regs))
            {
                decoded->decoded.error_type = "BadPageRetirement";
                decoded->decoded.description =
                    "Bad page retirement threshold exceeded";
                decoded->decoded.severity = "Fatal";
                decoded->decoded.instance = std::nullopt;
                decoded->decoded.afid = {19};
                decoded->event_report = addc::mca::to_json(decoded->decoded);
            }

            const bool has_ei = !s.error_info_structures.empty();
            const auto* ei = has_ei ? s.error_info_structures.data() : nullptr;

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
        else if (gs.context_type == GpuContextType::BootMessage)
        {
            auto messages = gpu::decode_boot_messages(gs.raw_bytes);
            if (messages.empty())
            {
                continue;
            }

            const auto boot_result = gpu::compute_boot_afid(messages);

            auto boot_reports =
                gpu::make_base_boot_reports(messages, boot_result.has_a4);
            if (boot_reports.empty())
            {
                continue;
            }

            const bool has_ei = !s.error_info_structures.empty();
            const auto* ei = has_ei ? s.error_info_structures.data() : nullptr;

            PipelineEvent ev;
            ev.metadata = gpu::make_gpu_metadata(s, context, ci, "Runtime Boot",
                                                 "boot_log", events.size());
            ev.platform = gpu::make_gpu_platform(s, has_ei, ei);
            ev.event_report = std::move(boot_reports);
            ev.data_array = gpu::make_boot_data_array(messages);
            ev.validation = gpu::make_boot_validation();
            ev.analysis["afid"] = addc::afid_list(boot_result.afid);
            ev.firmware = gpu::make_firmware_json(s);

            events.push_back(std::move(ev));
        }
    }

    return events;
}

} // namespace addc::pipeline
