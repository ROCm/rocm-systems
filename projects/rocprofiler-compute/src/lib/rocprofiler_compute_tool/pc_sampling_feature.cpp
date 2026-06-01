// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_feature.h"

#include "code_object_translator.h"
#include "code_object_writer.h"
#include "ps_file_writer.h"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/pc_sampling.h>

#include <iostream>

using namespace rocprofiler_compute_tool;

namespace
{
constexpr size_t k_pc_buffer_size      = 4 * 1024 * 1024;
constexpr size_t k_pc_buffer_watermark = k_pc_buffer_size / 2;

rocprofiler_pc_sampling_method_t to_sdk_method(PcSamplingMode mode)
{
    switch (mode)
    {
    case PcSamplingMode::Stochastic:
        return ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC;
    case PcSamplingMode::HostTrap:
        return ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;
    case PcSamplingMode::Disabled:
    default:
        return ROCPROFILER_PC_SAMPLING_METHOD_NONE;
    }
}

// Threaded through the SDK C callbacks to capture the chosen agent + config.
struct configure_ctx_t
{
    rocprofiler_pc_sampling_method_t wanted_method = ROCPROFILER_PC_SAMPLING_METHOD_NONE;
    bool                             found         = false;
    rocprofiler_agent_id_t           agent_id{0};
    rocprofiler_pc_sampling_unit_t   unit{ROCPROFILER_PC_SAMPLING_UNIT_NONE};
    uint64_t                         interval = 0;
};

rocprofiler_status_t pc_sampling_config_cb(const rocprofiler_pc_sampling_configuration_t* configs,
                                           size_t num_config,
                                           void*  user_data)
{
    auto* cctx = static_cast<configure_ctx_t*>(user_data);
    for (size_t i = 0; i < num_config && !cctx->found; ++i)
    {
        if (configs[i].method == cctx->wanted_method)
        {
            cctx->found = true;
            cctx->unit  = configs[i].unit;
            // Choose a supported interval from the queried config. min_interval
            // is the highest-frequency (smallest-period) value the agent allows.
            cctx->interval = configs[i].min_interval;
        }
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t available_agents_cb(rocprofiler_agent_version_t version,
                                         const void**                agents,
                                         size_t                      num_agents,
                                         void*                       user_data)
{
    if (version != ROCPROFILER_AGENT_INFO_VERSION_0)
        return ROCPROFILER_STATUS_SUCCESS;

    auto* both = static_cast<std::pair<SdkWrapper*, configure_ctx_t*>*>(user_data);
    auto* sdk  = both->first;
    auto* cctx = both->second;

    for (size_t i = 0; i < num_agents && !cctx->found; ++i)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
        if (agent == nullptr || agent->type != ROCPROFILER_AGENT_TYPE_GPU)
            continue;

        configure_ctx_t per_agent;
        per_agent.wanted_method = cctx->wanted_method;
        sdk->query_pc_sampling_agent_configurations(agent->id, pc_sampling_config_cb, &per_agent);
        if (per_agent.found)
        {
            cctx->found    = true;
            cctx->agent_id = agent->id;
            cctx->unit     = per_agent.unit;
            cctx->interval = per_agent.interval;
        }
    }
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace

PcSamplingMode rocprofiler_compute_tool::parse_pc_sampling_mode(const std::string& mode)
{
    if (mode == "stochastic")
        return PcSamplingMode::Stochastic;
    if (mode == "host_trap")
        return PcSamplingMode::HostTrap;
    return PcSamplingMode::Disabled;
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode mode, std::filesystem::path output_path)
    : pc_sampling_feature_t(mode,
                            std::move(output_path),
                            pc_sampling_collector_t::create(
                                std::make_shared<code_object_translator_impl_t>()))
{
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode               mode,
                                             std::filesystem::path        output_path,
                                             pc_sampling_collector_t::ptr collector)
    : m_enabled(true)
    , m_mode(mode)
    , m_output_path(std::move(output_path))
    , m_collector(std::move(collector))
{
    // Share one translator (and therefore one decode of each code object)
    // between the code-object writer and the PC-sample collector.
    m_record_collector = pc_sample_record_collector_t::create(m_collector->translator());
}

void pc_sampling_feature_t::on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    // A single shared translator backs both collectors, so one load call decodes
    // each code object exactly once.
    m_collector->on_code_object_load(info);
}

void pc_sampling_feature_t::finalize()
{
    code_object_writer_json_t writer;
    m_collector->write(writer);
    writer.flush(m_output_path);
}

void pc_sampling_feature_t::configure(SdkWrapper&                     sdk,
                                      rocprofiler_buffer_tracing_cb_t buffer_cb,
                                      void*                           buffer_cb_data)
{
    if (!m_enabled)
        return;

    const auto method = to_sdk_method(m_mode);
    if (method == ROCPROFILER_PC_SAMPLING_METHOD_NONE)
    {
        std::clog << "[rocprofiler-compute] PC sampling mode disabled; skipping configure\n";
        return;
    }

    configure_ctx_t cctx;
    cctx.wanted_method = method;

    std::pair<SdkWrapper*, configure_ctx_t*> agents_data{&sdk, &cctx};
    sdk.query_available_agents(available_agents_cb, sizeof(rocprofiler_agent_v0_t), &agents_data);

    if (!cctx.found)
    {
        std::clog << "[rocprofiler-compute] No GPU agent reported a matching PC sampling "
                     "configuration; skipping\n";
        return;
    }

    sdk.create_buffer(m_pc_context,
                      k_pc_buffer_size,
                      k_pc_buffer_watermark,
                      ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                      buffer_cb,
                      buffer_cb_data,
                      &m_pc_buffer);

    const auto status = sdk.configure_pc_sampling_service(m_pc_context,
                                                          cctx.agent_id,
                                                          method,
                                                          cctx.unit,
                                                          cctx.interval,
                                                          m_pc_buffer,
                                                          0);
    if (status != ROCPROFILER_STATUS_SUCCESS)
    {
        std::clog << "[rocprofiler-compute] configure_pc_sampling_service failed (status=" << status
                  << "); skipping\n";
        m_pc_buffer = rocprofiler_buffer_id_t{0};
    }
}

void pc_sampling_feature_t::start(SdkWrapper& sdk)
{
    if (m_pc_buffer.handle != 0)
        sdk.start_context(m_pc_context);
}

void pc_sampling_feature_t::on_sample_records(rocprofiler_record_header_t** headers, size_t count)
{
    if (!m_record_collector)
        return;

    // Decode the whole batch locally, then append in one locked call to avoid
    // taking the collector mutex once per record on this hot callback path.
    std::vector<pc_sampling_record_t> batch;
    batch.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        auto* header = headers[i];
        if (header == nullptr || header->category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
            continue;

        switch (header->kind)
        {
        case ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE:
        {
            const auto* p = static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t*>(header->payload);
            pc_sampling_record_t rec{};
            rec.code_object_id     = p->pc.code_object_id;
            rec.code_object_offset = p->pc.code_object_offset;
            rec.dispatch_id        = p->dispatch_id;
            // Host-trap records carry no wave_issued field; leave it unset.
            rec.is_stochastic = false;
            batch.push_back(std::move(rec));
            break;
        }
        case ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE:
        {
            const auto* p = static_cast<rocprofiler_pc_sampling_record_stochastic_v0_t*>(header->payload);
            pc_sampling_record_t rec{};
            rec.code_object_id     = p->pc.code_object_id;
            rec.code_object_offset = p->pc.code_object_offset;
            rec.dispatch_id        = p->dispatch_id;
            const bool wave_issued = (p->wave_issued != 0);
            rec.wave_issued        = wave_issued;
            rec.is_stochastic      = true;
            if (!wave_issued)
            {
                const char* reason = rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
                    static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                        p->snapshot.reason_not_issued));
                if (reason != nullptr)
                    rec.stall_reason = std::string{reason};
            }
            batch.push_back(std::move(rec));
            break;
        }
        default:
            std::clog << "[rocprofiler-compute] Skipped unrecognized PC sampling record kind: "
                      << header->kind << "\n";
            break;
        }
    }

    if (!batch.empty())
        m_record_collector->add_records(batch);
}

void pc_sampling_feature_t::stop_and_flush(SdkWrapper& sdk)
{
    if (m_pc_buffer.handle == 0)
        return;

    // Stop sampling, then flush so every buffered record below the watermark is
    // delivered to on_sample_records before finalize_samples serializes them.
    sdk.stop_context(m_pc_context);
    sdk.flush_buffer(m_pc_buffer);
}

void pc_sampling_feature_t::finalize_samples()
{
    if (!m_enabled || m_ps_output_path.empty() || !m_record_collector)
        return;

    ps_file_writer_json_t writer;
    m_record_collector->write(writer);
    writer.flush(m_ps_output_path);
}
