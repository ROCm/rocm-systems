// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_feature.h"

#include "code_object_writer.h"
#include "pc_record_writer.h"
#include "sdk_callbacks.h"
#include "sdk_wrapper.h"
#include "source_snapshot.h"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/pc_sampling.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>

using namespace rocprofiler_compute_tool;

namespace
{
// Map the tool's PC sampling mode to the SDK method enum.
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

// Map a unit string to the SDK unit enum. Returns NONE for an unrecognized or
// empty string so the caller can fall back to the agent's advertised unit.
rocprofiler_pc_sampling_unit_t to_sdk_unit(const std::string& unit)
{
    if (unit == "instructions")
        return ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS;
    if (unit == "cycles")
        return ROCPROFILER_PC_SAMPLING_UNIT_CYCLES;
    if (unit == "time")
        return ROCPROFILER_PC_SAMPLING_UNIT_TIME;
    return ROCPROFILER_PC_SAMPLING_UNIT_NONE;
}

const char* safe_name(const char* name)
{
    return name != nullptr ? name : "";
}

// A code_object_writer_t that captures the de-duplicated instruction/comment
// string table (keyed by code_object_id + offset), kernel symbols, and the code
// object id list during the collector's disassembly walk. Reused by finalize()
// so no second translator traversal is required.
class ps_metadata_capture_writer_t : public code_object_writer_t
{
public:
    void start_code_obj(size_t obj_id) override
    {
        m_current_obj_id = obj_id;
        m_code_objects.push_back(nlohmann::json::object({{"id", obj_id}}));
    }

    void end_code_obj() override {}

    void start_symbol(const symbol_t& symbol) override
    {
        m_kernel_symbols.push_back(pc_kernel_symbol_t{m_current_obj_id, symbol.name});
    }

    void end_symbol() override {}

    void write_instruction(const instruction_t& inst) override
    {
        const int index = static_cast<int>(m_strings.pc_sample_instructions.size());
        m_strings.pc_sample_instructions.push_back(inst.name);
        m_strings.pc_sample_comments.push_back(inst.comment);
        m_strings.offset_to_index[{static_cast<uint64_t>(m_current_obj_id), inst.code_obj_offset}] = index;
    }

    std::string get_result() override { return std::string{}; }

    void flush(const std::filesystem::path&) override {}

    pc_sample_strings_t& strings() { return m_strings; }

    std::vector<pc_kernel_symbol_t>& kernel_symbols() { return m_kernel_symbols; }

    nlohmann::json::array_t& code_objects() { return m_code_objects; }

private:
    size_t                          m_current_obj_id = 0;
    pc_sample_strings_t             m_strings{};
    std::vector<pc_kernel_symbol_t> m_kernel_symbols{};
    nlohmann::json::array_t         m_code_objects = nlohmann::json::array();
};

// Forwards every writer callback to two underlying writers so a single
// translator traversal feeds both the code_obj_info JSON and the PC sampling
// metadata capture, instead of walking the translator twice.
class tee_code_object_writer_t : public code_object_writer_t
{
public:
    tee_code_object_writer_t(code_object_writer_t& a, code_object_writer_t& b)
        : m_a(a)
        , m_b(b)
    {
    }

    void start_code_obj(size_t obj_id) override
    {
        m_a.start_code_obj(obj_id);
        m_b.start_code_obj(obj_id);
    }

    void end_code_obj() override
    {
        m_a.end_code_obj();
        m_b.end_code_obj();
    }

    void start_symbol(const symbol_t& symbol) override
    {
        m_a.start_symbol(symbol);
        m_b.start_symbol(symbol);
    }

    void end_symbol() override
    {
        m_a.end_symbol();
        m_b.end_symbol();
    }

    void write_instruction(const instruction_t& inst) override
    {
        m_a.write_instruction(inst);
        m_b.write_instruction(inst);
    }

    std::string get_result() override { return std::string{}; }

    void flush(const std::filesystem::path&) override {}

private:
    code_object_writer_t& m_a;
    code_object_writer_t& m_b;
};

// Collected agent + the PC sampling config chosen for it.
struct agent_config_match_t
{
    rocprofiler_pc_sampling_configuration_t config{};
    bool                                    matched = false;
};
}  // namespace

PcSamplingMode rocprofiler_compute_tool::parse_pc_sampling_mode(const std::string& mode)
{
    if (mode == "stochastic")
        return PcSamplingMode::Stochastic;
    if (mode == "host_trap")
        return PcSamplingMode::HostTrap;
    return PcSamplingMode::Disabled;
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode        mode,
                                             uint64_t              interval,
                                             std::string           unit,
                                             std::filesystem::path code_obj_info_path,
                                             std::filesystem::path ps_file_path)
    : pc_sampling_feature_t(mode,
                            interval,
                            std::move(unit),
                            std::move(code_obj_info_path),
                            std::move(ps_file_path),
                            pc_sampling_collector_t::create())
{
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode               mode,
                                             uint64_t                     interval,
                                             std::string                  unit,
                                             std::filesystem::path        code_obj_info_path,
                                             std::filesystem::path        ps_file_path,
                                             pc_sampling_collector_t::ptr collector)
    : m_enabled(true)
    , m_mode(mode)
    , m_interval(interval)
    , m_unit(std::move(unit))
    , m_output_path(std::move(code_obj_info_path))
    , m_ps_file_path(std::move(ps_file_path))
    , m_collector(std::move(collector))
    , m_record_store(std::make_shared<pc_record_store_t>())
{
}

void pc_sampling_feature_t::on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    m_collector->on_code_object_load(info);
}

bool pc_sampling_feature_t::configure(rocprofiler_context_id_t ctx,
                                      SdkWrapper&              sdk,
                                      void*                    buffer_callback_user_data)
{
    // No SDK service is configured for the disabled mode; drain nothing.
    if (m_mode == PcSamplingMode::Disabled)
        return false;

    // The SDK seam throws on any non-success status (e.g. PC sampling is not
    // implemented on this runtime/agent, returning
    // ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED). This runs from tool_init,
    // whose frames are reached through the SDK's C callbacks, so an escaping
    // exception would reach a C ABI boundary and terminate the profiled
    // application. Treat any failure as "PC sampling unavailable": log, leave
    // the buffer unconfigured, and report false so the caller degrades
    // gracefully (the application still runs; no PC samples are collected).
    try
    {
        return try_configure(ctx, sdk, buffer_callback_user_data);
    }
    catch (const std::exception& e)
    {
        std::clog << "[rocprofiler-compute] WARNING: PC sampling could not be "
                     "configured on this system; continuing without PC samples ("
                  << e.what() << ")\n";
        return false;
    }
}

bool pc_sampling_feature_t::try_configure(rocprofiler_context_id_t ctx,
                                          SdkWrapper&              sdk,
                                          void*                    buffer_callback_user_data)
{
    const auto requested_method = to_sdk_method(m_mode);
    const auto requested_unit   = to_sdk_unit(m_unit);

    // Step 1: enumerate GPU agents.
    std::vector<rocprofiler_agent_id_t> gpu_agents;
    sdk.iterate_agents(
        [](rocprofiler_agent_version_t, const void** agents, size_t num_agents, void* user_data)
        {
            auto* out = static_cast<std::vector<rocprofiler_agent_id_t>*>(user_data);
            for (size_t i = 0; i < num_agents; ++i)
            {
                const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[i]);
                if (agent != nullptr && agent->type == ROCPROFILER_AGENT_TYPE_GPU)
                    out->push_back(agent->id);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        sizeof(rocprofiler_agent_v0_t),
        static_cast<void*>(&gpu_agents));

    // Step 2: for each GPU agent, find a config matching method (+ unit if set).
    for (const auto& agent_id : gpu_agents)
    {
        struct query_ctx_t
        {
            rocprofiler_pc_sampling_method_t requested_method;
            rocprofiler_pc_sampling_unit_t   requested_unit;
            agent_config_match_t             match;
        } query_ctx{requested_method, requested_unit, agent_config_match_t{}};

        sdk.query_pc_sampling_agent_configurations(
            agent_id,
            [](const rocprofiler_pc_sampling_configuration_t* configs, size_t num_config, void* user_data)
            {
                auto* qc = static_cast<query_ctx_t*>(user_data);
                if (qc->match.matched)
                    return ROCPROFILER_STATUS_SUCCESS;
                for (size_t i = 0; i < num_config; ++i)
                {
                    if (configs[i].method != qc->requested_method)
                        continue;
                    // Accept the agent's advertised unit when none was requested.
                    if (qc->requested_unit != ROCPROFILER_PC_SAMPLING_UNIT_NONE &&
                        configs[i].unit != qc->requested_unit)
                        continue;
                    qc->match.config  = configs[i];
                    qc->match.matched = true;
                    break;
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            static_cast<void*>(&query_ctx));

        if (!query_ctx.match.matched)
            continue;

        // Step 3: create the delivery buffer for PC samples.
        constexpr size_t buffer_size      = 4 * 1024 * 1024;  // 4 MB
        constexpr size_t buffer_watermark = buffer_size / 2;
        sdk.create_buffer(ctx,
                          buffer_size,
                          buffer_watermark,
                          ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                          &pc_sampling_buffer_callback,
                          buffer_callback_user_data,
                          &m_buffer_id);

        // Step 4: configure the service. Honor the INTERVAL_POW2 flag if the
        // chosen config requires it; otherwise pass no flags.
        int flags = 0;
        if ((query_ctx.match.config.flags & ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_INTERVAL_POW2) != 0u)
            flags = ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_INTERVAL_POW2;

        sdk.configure_pc_sampling_service(ctx,
                                          agent_id,
                                          requested_method,
                                          query_ctx.match.config.unit,
                                          m_interval,
                                          m_buffer_id,
                                          flags);
        return true;
    }

    return false;
}

void pc_sampling_feature_t::on_pc_sample_records(rocprofiler_record_header_t** headers, size_t num_headers)
{
    if (headers == nullptr)
        return;

    // Accumulate this delivery's records, then hand each kind to the store in a
    // single locked batch instead of locking once per sample.
    std::vector<pc_sample_record_t> stochastic;
    std::vector<pc_sample_record_t> host_trap;

    for (size_t i = 0; i < num_headers; ++i)
    {
        rocprofiler_record_header_t* header = headers[i];
        if (header == nullptr || header->category != ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
            continue;

        if (header->kind == ROCPROFILER_PC_SAMPLING_RECORD_STOCHASTIC_V0_SAMPLE)
        {
            const auto* sample = static_cast<const rocprofiler_pc_sampling_record_stochastic_v0_t*>(
                header->payload);
            if (sample == nullptr)
                continue;

            pc_sample_record_t record{};
            record.code_object_id     = sample->pc.code_object_id;
            record.code_object_offset = sample->pc.code_object_offset;
            record.wave_issued        = static_cast<int>(sample->wave_issued);
            record.dispatch_id        = sample->dispatch_id;
            record.inst_type          = safe_name(rocprofiler_get_pc_sampling_instruction_type_name(
                static_cast<rocprofiler_pc_sampling_instruction_type_t>(sample->inst_type)));
            record.stall_reason = safe_name(rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
                static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                    sample->snapshot.reason_not_issued)));
            stochastic.push_back(record);
        }
        else if (header->kind == ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE)
        {
            const auto* sample = static_cast<const rocprofiler_pc_sampling_record_host_trap_v0_t*>(
                header->payload);
            if (sample == nullptr)
                continue;

            pc_sample_record_t record{};
            record.code_object_id     = sample->pc.code_object_id;
            record.code_object_offset = sample->pc.code_object_offset;
            record.dispatch_id        = sample->dispatch_id;
            host_trap.push_back(record);
        }
        // Skip unknown / invalid kinds.
    }

    if (!stochastic.empty())
        m_record_store->add_stochastic(std::move(stochastic));
    if (!host_trap.empty())
        m_record_store->add_host_trap(std::move(host_trap));
}

void pc_sampling_feature_t::flush(SdkWrapper& sdk)
{
    // configure() leaves m_buffer_id at its default {0} when it set up no SDK
    // service (disabled mode or no matching agent); nothing to drain then.
    if (m_buffer_id.handle == 0)
        return;
    sdk.flush_buffer(m_buffer_id);
}

void pc_sampling_feature_t::finalize()
{
    // Walk the translator once, feeding both the code_obj_info JSON writer and
    // the PC sampling metadata capture via a tee. Source paths are harvested
    // during the same walk (see collector::write -> record_source_path).
    code_object_writer_json_t    writer;
    ps_metadata_capture_writer_t capture;
    tee_code_object_writer_t     tee{writer, capture};
    m_collector->write(tee);
    writer.flush(m_output_path);

    auto base = m_output_path.parent_path();
    if (base.empty())
        base = ".";
    copy_source_files(m_collector->collect_source_paths(), base / "code_obj_sources");

    // Emit the ps_file results from the captured metadata. Both record arrays
    // are always present (one may be empty); m_record_store is always set by the
    // constructor.
    pc_record_writer_t ps_writer;
    ps_writer.write(*m_record_store, capture.strings(), capture.kernel_symbols(), capture.code_objects());
    ps_writer.flush(m_ps_file_path);
}
