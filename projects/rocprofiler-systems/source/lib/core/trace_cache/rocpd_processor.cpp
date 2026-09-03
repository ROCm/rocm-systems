// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/rocpd_processor.hpp"
#include "agent.hpp"
#include "common/md5sum.hpp"
#include "common/units.hpp"
#include "core/agent_manager.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/node_info.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/rocpd_helpers.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/cpu/sample.hpp"
#include "library/pmc/collectors/gpu/types.hpp"
#include "library/pmc/collectors/nic/sample.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <array>
#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/version.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::trace_cache
{
namespace
{

using rocpd_helpers::make_agent_uid;
using rocpd_helpers::make_event;
using rocpd_helpers::make_trace_env;
using rocpd_helpers::make_trace_env_with_agent;
using rocpd_helpers::make_trace_env_with_agent_queue_stream;
using rocpd_helpers::parse_memory_operation_name;

auto
get_handle_from_code_object(
    const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
#if(ROCPROFILER_VERSION >= 600)
    return code_object.agent_id.handle;
#else
    return code_object.rocp_agent.handle;
#endif
}

std::string
generate_db_output_path(int pid)
{
    auto tag     = std::to_string(pid);
    auto db_name = std::string{ "rocpd" };
    return rocprofsys::get_database_absolute_path(db_name, tag);
}

}  // namespace

void
rocpd_processor_t::handle(const kernel_dispatch_sample& kds)
{
    const auto& n_info    = node_info::get_instance();
    auto        process   = m_metadata->get_process_info();
    const auto& agent_ref = m_agent_manager->get_agent_by_handle(kds.agent_id_handle);

    auto kernel_symbol = m_metadata->get_kernel_symbol(kds.kernel_id);
    if(!kernel_symbol.has_value())
    {
        throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
    }

    auto kernel_name = rocprofsys::utility::demangle(kernel_symbol->kernel_name);

    auto event = make_event(kds.correlation_id_internal, kds.correlation_id_ancestor, 0,
                            trait::name<category::rocm_kernel_dispatch>::value);

    profiler_hub::writer_types::kernel_dispatch_data_t kernel_dispatch;
    kernel_dispatch.event                = event;
    kernel_dispatch.dispatch_id          = kds.dispatch_id;
    kernel_dispatch.start_timestamp      = kds.start_timestamp;
    kernel_dispatch.end_timestamp        = kds.end_timestamp;
    kernel_dispatch.kernel_symbol_id     = kds.kernel_id;
    kernel_dispatch.private_segment_size = kds.private_segment_size;
    kernel_dispatch.group_segment_size   = kds.group_segment_size;
    kernel_dispatch.workgroup_size_x     = kds.workgroup_size_x;
    kernel_dispatch.workgroup_size_y     = kds.workgroup_size_y;
    kernel_dispatch.workgroup_size_z     = kds.workgroup_size_z;
    kernel_dispatch.grid_size_x          = kds.grid_size_x;
    kernel_dispatch.grid_size_y          = kds.grid_size_y;
    kernel_dispatch.grid_size_z          = kds.grid_size_z;
    kernel_dispatch.name                 = kernel_name.c_str();

    auto env = make_trace_env_with_agent_queue_stream(
        n_info.id, process.pid, kds.thread_id, agent_ref, kds.queue_id_handle,
        kds.stream_handle);

    m_writer->insert_kernel_dispatch_data(kernel_dispatch, env);
}

void
rocpd_processor_t::handle(const scratch_memory_sample& sms)
{
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    const auto* name = m_metadata->get_buffer_name_info().at(
        static_cast<rocprofiler_buffer_tracing_kind_t>(sms.kind),
        static_cast<rocprofiler_tracing_operation_t>(sms.operation));

    const auto& agent_ref = m_agent_manager->get_agent_by_handle(sms.agent_id_handle);

    auto [memory_operation, memory_type_val] = parse_memory_operation_name(name);
    auto extdata_json_str = fmt::format("{{\"flags\": {}}}", sms.flags);

    auto event = make_event(sms.correlation_id_internal, sms.correlation_id_ancestor, 0,
                            trait::name<category::rocm_scratch_memory>::value);

    profiler_hub::writer_types::memory_alloc_data_t ma;
    ma.event           = event;
    ma.type            = memory_operation.c_str();
    ma.level           = memory_type_val.c_str();
    ma.start_timestamp = sms.start_timestamp;
    ma.end_timestamp   = sms.end_timestamp;
    ma.address         = 0;
    ma.size            = sms.allocation_size;
    ma.extdata         = extdata_json_str;

    auto env = make_trace_env_with_agent_queue_stream(
        n_info.id, process.pid, sms.thread_id, agent_ref, sms.queue_id_handle,
        sms.stream_handle);

    m_writer->insert_memory_alloc_data(ma, env);
}

void
rocpd_processor_t::handle(const memory_copy_sample& mcs)
{
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    auto name = std::string{ m_metadata->get_buffer_name_info().at(
        static_cast<rocprofiler_buffer_tracing_kind_t>(mcs.kind),
        static_cast<rocprofiler_tracing_operation_t>(mcs.operation)) };

    const auto& dst_agent = m_agent_manager->get_agent_by_handle(mcs.dst_agent_id_handle);
    const auto& src_agent = m_agent_manager->get_agent_by_handle(mcs.src_agent_id_handle);

    auto event = make_event(mcs.correlation_id_internal, mcs.correlation_id_ancestor, 0,
                            trait::name<category::rocm_memory_copy>::value);

    profiler_hub::writer_types::memory_copy_data_t memory_copy;
    memory_copy.event           = event;
    memory_copy.start_timestamp = mcs.start_timestamp;
    memory_copy.end_timestamp   = mcs.end_timestamp;
    memory_copy.dst_agent_id    = make_agent_uid(dst_agent);
    memory_copy.dst_address     = mcs.dst_address_value;
    memory_copy.src_agent_id    = make_agent_uid(src_agent);
    memory_copy.src_address     = mcs.src_address_value;
    memory_copy.size            = mcs.bytes;
    memory_copy.name            = name;
    memory_copy.region_name     = name;

    auto env      = make_trace_env(n_info.id, process.pid, mcs.thread_id);
    env.stream_id = mcs.stream_handle;
    env.queue_id  = 0;

    m_writer->insert_memory_copy_data(memory_copy, env);
}

void
rocpd_processor_t::handle([[maybe_unused]] const memory_allocate_sample& mas)
{
#if(ROCPROFILER_VERSION >= 600)
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    const auto invalid_context = ROCPROFILER_CONTEXT_NONE;
    if(mas.agent_id_handle != invalid_context.handle)
    {
        const auto& agent_ref = m_agent_manager->get_agent_by_handle(mas.agent_id_handle);

        const auto* name = m_metadata->get_buffer_name_info().at(
            static_cast<rocprofiler_buffer_tracing_kind_t>(mas.kind),
            static_cast<rocprofiler_tracing_operation_t>(mas.operation));

        auto [memory_operation, memory_type_val] = parse_memory_operation_name(name);

        auto event = make_event(mas.correlation_id_internal, mas.correlation_id_ancestor,
                                0, trait::name<category::rocm_memory_allocate>::value);

        profiler_hub::writer_types::memory_alloc_data_t ma;
        ma.event           = event;
        ma.type            = memory_operation;
        ma.level           = memory_type_val;
        ma.start_timestamp = mas.start_timestamp;
        ma.end_timestamp   = mas.end_timestamp;
        ma.address         = mas.address_value;
        ma.size            = mas.allocation_size;

        auto env =
            make_trace_env_with_agent(n_info.id, process.pid, mas.thread_id, agent_ref);
        env.stream_id = mas.stream_handle;
        env.queue_id  = 0;

        m_writer->insert_memory_alloc_data(ma, env);
    }
#endif
}

void
rocpd_processor_t::handle(const region_sample& reg_sample)
{
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    auto event =
        make_event(reg_sample.correlation_id_internal, reg_sample.correlation_id_ancestor,
                   0, reg_sample.category.c_str());
    event.call_stack.push_back({});
    // call_stack and line_info are serialized JSON in the old code; in profiler-hub
    // they are structured types. For now pass the raw JSON via extdata.
    event.extdata = reg_sample.call_stack;

    profiler_hub::writer_types::region_data_t region;
    region.event           = event;
    region.start_timestamp = reg_sample.start_timestamp;
    region.end_timestamp   = reg_sample.end_timestamp;
    region.name            = reg_sample.name;

    auto parsed_args = process_arguments_string(reg_sample.args_str);
    for(const auto& arg : parsed_args)
    {
        profiler_hub::writer_types::arg_data_t arg_data;
        arg_data.position = arg.arg_number;
        arg_data.type     = arg.arg_type;
        arg_data.name     = arg.arg_name;
        arg_data.value    = arg.arg_value;
        region.args.push_back(arg_data);
    }

    auto env = make_trace_env(n_info.id, process.pid, reg_sample.thread_id);
    m_writer->insert_region_data(region, env);
}

void
rocpd_processor_t::handle(const backtrace_region_sample& bts)
{
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    auto event = make_event(0, 0, 0, bts.category.c_str());
    event.call_stack.push_back({});
    // call_stack and line_info are serialized JSON in the old code; in profiler-hub
    // they are structured types. For now pass the raw JSON via extdata.
    event.extdata = bts.call_stack;

    profiler_hub::writer_types::region_data_t region;
    region.event           = event;
    region.start_timestamp = bts.start_timestamp;
    region.end_timestamp   = bts.end_timestamp;
    region.name            = bts.name;

    auto env       = make_trace_env(n_info.id, process.pid, bts.thread_id);
    env.track_name = bts.track_name;

    m_writer->insert_region_data(region, env);
}

void
rocpd_processor_t::handle(const in_time_sample& its)
{
    auto event    = make_event(its.stack_id, its.parent_stack_id, its.correlation_id,
                               its.track_name.c_str());
    event.extdata = its.event_metadata;

    profiler_hub::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event = event;
    pmc_data.value = 0.0;

    profiler_hub::writer_types::track_info_t track;
    track.name = its.track_name;

    profiler_hub::writer_types::sample_data_t sample;
    sample.timestamp = its.timestamp_ns;
    sample.track     = track;
    pmc_data.sample  = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
    pmc_uid.name = its.track_name;

    try_insert_pmc_event(pmc_data, pmc_uid, "In-time sample");
}

void
rocpd_processor_t::handle(const pmc_event_with_sample& pmc)
{
    const auto& process_info = m_metadata->get_process_info();

    const agent* agent_ptr = nullptr;
    try
    {
        agent_ptr = &m_agent_manager->get_agent_by_type_index(
            pmc.device_id, static_cast<agent_type>(pmc.device_type));
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("PMC event skipped: agent lookup failed for device_id={}, "
                    "device_type={}: {}",
                    pmc.device_id, pmc.device_type, e.what());
        return;
    }

    const auto& agent_ref = *agent_ptr;
    auto        event = make_event(pmc.stack_id, pmc.parent_stack_id, pmc.correlation_id,
                                   pmc.track_name.c_str());
    event.extdata     = pmc.event_metadata;

    profiler_hub::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event   = event;
    pmc_data.value   = pmc.value;
    pmc_data.extdata = pmc.event_metadata;

    profiler_hub::writer_types::track_info_t track;
    track.name       = pmc.track_name;
    track.node_id    = node_info::get_instance().id;
    track.process_id = process_info.pid;
    track.thread_id  = pmc.system_tid;

    profiler_hub::writer_types::sample_data_t sample;
    sample.timestamp = pmc.timestamp_ns;
    sample.track     = track;
    pmc_data.sample  = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
    pmc_uid.name     = pmc.pmc_info_name;
    pmc_uid.agent_id = make_agent_uid(agent_ref);

    try_insert_pmc_event(pmc_data, pmc_uid, "PMC event");
}

void
rocpd_processor_t::handle([[maybe_unused]] const gpu_pmc_sample& gpu_pmc)
{
    const auto* name         = trait::name<category::amd_smi>::value;
    const auto& process_info = m_metadata->get_process_info();

    const agent* agent_ptr = nullptr;
    try
    {
        agent_ptr =
            &m_agent_manager->get_agent_by_type_index(gpu_pmc.device_id, agent_type::GPU);
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("GPU PMC sample skipped: agent lookup failed for device_id={}: {}",
                    gpu_pmc.device_id, e.what());
        return;
    }

    const auto& agent_ref = *agent_ptr;
    const auto  agent_uid = make_agent_uid(agent_ref);

    auto event = make_event(0, 0, 0, name);

    auto insert_event_and_sample = [&](bool is_enabled, const char* pmc_name,
                                       const char* track_name, double value) {
        if(!is_enabled) return;

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = event;
        pmc_data.value = value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = track_name;
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = gpu_pmc.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = pmc_name;
        pmc_uid.agent_id = agent_uid;

        try_insert_pmc_event(pmc_data, pmc_uid, "GPU PMC sample");
    };

    const auto& m       = gpu_pmc.metric_values;
    const auto& enabled = gpu_pmc.enabled_metric;

    auto insert_scalar = [&](const char* metric_name, const std::string& track,
                             bool is_enabled, double value) {
        insert_event_and_sample(is_enabled, metric_name, track.c_str(), value);
    };

    insert_scalar(trait::name<category::amd_smi_gfx_busy>::value,
                  info::format_track_name<category::amd_smi_gfx_busy>(),
                  enabled.bits.gfx_activity, m.gfx_activity);
    insert_scalar(trait::name<category::amd_smi_umc_busy>::value,
                  info::format_track_name<category::amd_smi_umc_busy>(),
                  enabled.bits.umc_activity, m.umc_activity);
    insert_scalar(trait::name<category::amd_smi_mm_busy>::value,
                  info::format_track_name<category::amd_smi_mm_busy>(),
                  enabled.bits.mm_activity, m.mm_activity);
    insert_scalar(trait::name<category::amd_smi_temp>::value,
                  info::format_track_name<category::amd_smi_temp>(),
                  enabled.bits.hotspot_temperature || enabled.bits.edge_temperature,
                  enabled.bits.hotspot_temperature ? m.hotspot_temperature
                                                   : m.edge_temperature);
    insert_scalar(trait::name<category::amd_smi_power>::value,
                  info::format_track_name<category::amd_smi_power>(),
                  enabled.bits.current_socket_power || enabled.bits.average_socket_power,
                  pmc::collectors::gpu::select_socket_power(enabled, m));
    insert_scalar(trait::name<category::amd_smi_memory_usage>::value,
                  info::format_track_name<category::amd_smi_memory_usage>(),
                  enabled.bits.memory_usage, m.memory_usage / units::megabyte);
    insert_scalar(trait::name<category::amd_smi_sdma_usage>::value,
                  info::format_track_name<category::amd_smi_sdma_usage>(),
                  enabled.bits.sdma_usage, m.sdma_usage);
    insert_scalar(trait::name<category::amd_smi_gfx_clock>::value,
                  info::format_track_name<category::amd_smi_gfx_clock>(),
                  enabled.bits.gfx_clock, m.gfx_clock_mhz);
    insert_scalar(trait::name<category::amd_smi_mem_clock>::value,
                  info::format_track_name<category::amd_smi_mem_clock>(),
                  enabled.bits.mem_clock, m.mem_clock_mhz);

    auto insert_xcp_metrics = [&](bool is_enabled, const auto& get_array,
                                  const auto& format_name) {
        if(!is_enabled) return;
        for(size_t xcp = 0; xcp < m.xcp_stats.size(); ++xcp)
        {
            const auto& arr = get_array(m.xcp_stats[xcp]);
            for(size_t i = 0; i < arr.size(); ++i)
            {
                if(arr[i] == pmc::collectors::gpu::METRIC_VALUE_NOT_SUPPORTED_16)
                {
                    continue;
                }
                auto metric_name =
                    format_name(static_cast<int>(xcp), static_cast<int>(i));
                insert_event_and_sample(true, metric_name.c_str(), metric_name.c_str(),
                                        static_cast<double>(arr[i]));
            }
        }
    };

    insert_xcp_metrics(
        enabled.bits.vcn_busy,
        [](const auto& xcp) -> const auto& { return xcp.vcn_busy; },
        [](int xcp, int engine) {
            return info::format_track_name<category::amd_smi_vcn_activity>(xcp, engine);
        });
    insert_xcp_metrics(
        enabled.bits.jpeg_busy,
        [](const auto& xcp) -> const auto& { return xcp.jpeg_busy; },
        [](int xcp, int engine) {
            return info::format_track_name<category::amd_smi_jpeg_activity>(xcp, engine);
        });

    auto insert_device_level_metrics = [&](const std::string_view base_name,
                                           bool is_enabled, const auto& arr) {
        if(!is_enabled) return;
        for(std::size_t i = 0; i < arr.size(); ++i)
        {
            if(arr[i] == pmc::collectors::gpu::METRIC_VALUE_NOT_SUPPORTED_16) continue;

            auto pmc_name   = fmt::format("{}_{}", base_name, i);
            auto track_name = pmc_name;

            LOG_TRACE("Inserting metric: pmc_name: {}, track_name: {}, value: {}",
                      pmc_name, track_name, arr[i]);
            insert_event_and_sample(true, pmc_name.c_str(), track_name.c_str(),
                                    static_cast<double>(arr[i]));
        }
    };

    insert_device_level_metrics(info::format_track_name<category::amd_smi_vcn_activity>(),
                                enabled.bits.vcn_activity, m.vcn_activity);

    insert_device_level_metrics(
        info::format_track_name<category::amd_smi_jpeg_activity>(),
        enabled.bits.jpeg_activity, m.jpeg_activity);

    insert_scalar(trait::name<category::amd_smi_pcie_link_width>::value,
                  info::format_track_name<category::amd_smi_pcie_link_width>(),
                  enabled.bits.pcie, m.pcie.link.width);
    insert_scalar(trait::name<category::amd_smi_pcie_link_speed>::value,
                  info::format_track_name<category::amd_smi_pcie_link_speed>(),
                  enabled.bits.pcie, m.pcie.link.speed);
    insert_scalar(trait::name<category::amd_smi_pcie_bandwidth_acc>::value,
                  info::format_track_name<category::amd_smi_pcie_bandwidth_acc>(),
                  enabled.bits.pcie, m.pcie.bandwidth.acc);
    insert_scalar(trait::name<category::amd_smi_pcie_bandwidth_inst>::value,
                  info::format_track_name<category::amd_smi_pcie_bandwidth_inst>(),
                  enabled.bits.pcie, m.pcie.bandwidth.inst);

    // XGMI metrics
    insert_scalar(trait::name<category::amd_smi_xgmi_link_width>::value,
                  info::format_track_name<category::amd_smi_xgmi_link_width>(),
                  enabled.bits.xgmi, m.xgmi.link.width);
    insert_scalar(trait::name<category::amd_smi_xgmi_link_speed>::value,
                  info::format_track_name<category::amd_smi_xgmi_link_speed>(),
                  enabled.bits.xgmi, m.xgmi.link.speed);

    // XGMI data accumulators (per-link arrays)
    auto insert_xgmi_link_metrics = [&](const std::string& base_track_name,
                                        bool is_enabled, const auto& arr) {
        if(!is_enabled) return;
        for(size_t i = 0; i < arr.size(); ++i)
        {
            if(arr[i] == pmc::collectors::gpu::METRIC_VALUE_NOT_SUPPORTED_64) continue;

            const std::string pmc_name = base_track_name + "_link" + std::to_string(i);
            const std::string track_name =
                base_track_name + " [Link " + std::to_string(i) + "]";
            insert_event_and_sample(true, pmc_name.c_str(), track_name.c_str(),
                                    static_cast<double>(arr[i]));
        }
    };

    insert_xgmi_link_metrics(trait::name<category::amd_smi_xgmi_read_data>::value,
                             enabled.bits.xgmi, m.xgmi.data_acc.read);
    insert_xgmi_link_metrics(trait::name<category::amd_smi_xgmi_write_data>::value,
                             enabled.bits.xgmi, m.xgmi.data_acc.write);
}

void
rocpd_processor_t::handle([[maybe_unused]] const ainic_pmc_sample& nic_sample)
{
    // Insert NIC RDMA metrics into rocpd database
    const auto* name         = trait::name<category::amd_smi_nic>::value;
    const auto& process_info = m_metadata->get_process_info();

    const agent* agent_ptr = nullptr;
    try
    {
        agent_ptr =
            &m_agent_manager->get_agent_by_id(nic_sample.device_id, agent_type::NIC);
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("NIC PMC sample skipped: agent lookup failed for device_id={}: {}",
                    nic_sample.device_id, e.what());
        return;
    }

    const auto& nic_agent = *agent_ptr;
    const auto  agent_uid = make_agent_uid(nic_agent);

    auto event = make_event(0, 0, 0, name);

    auto insert_event_and_sample = [&](bool is_enabled, const char* pmc_name,
                                       const char* track_name, std::uint64_t value) {
        if(!is_enabled) return;

        LOG_TRACE("Inserting metric: pmc_name: {}, track_name: {}, value: {}", pmc_name,
                  track_name, value);

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = event;
        pmc_data.value = static_cast<double>(value);

        profiler_hub::writer_types::track_info_t track;
        track.name       = track_name;
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = nic_sample.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = pmc_name;
        pmc_uid.agent_id = agent_uid;

        try_insert_pmc_event(pmc_data, pmc_uid, "NIC PMC sample");
    };

    const auto& mtrcs   = nic_sample.metric_values;
    const auto& enabled = nic_sample.enabled_metric;

    insert_event_and_sample(enabled.bits.rx_rdma_ucast_bytes,
                            trait::name<category::amd_smi_nic_rx_ucast_bytes>::value,
                            "ainic_rx_rdma_ucast_bytes", mtrcs.rx_rdma_ucast_bytes);
    insert_event_and_sample(enabled.bits.tx_rdma_ucast_bytes,
                            trait::name<category::amd_smi_nic_tx_ucast_bytes>::value,
                            "ainic_tx_rdma_ucast_bytes", mtrcs.tx_rdma_ucast_bytes);
    insert_event_and_sample(enabled.bits.rx_rdma_ucast_pkts,
                            trait::name<category::amd_smi_nic_rx_ucast_pkts>::value,
                            "ainic_rx_rdma_ucast_pkts", mtrcs.rx_rdma_ucast_pkts);
    insert_event_and_sample(enabled.bits.tx_rdma_ucast_pkts,
                            trait::name<category::amd_smi_nic_tx_ucast_pkts>::value,
                            "ainic_tx_rdma_ucast_pkts", mtrcs.tx_rdma_ucast_pkts);
    insert_event_and_sample(enabled.bits.rx_rdma_cnp_pkts,
                            trait::name<category::amd_smi_nic_rx_cnp_pkts>::value,
                            "ainic_rx_rdma_cnp_pkts", mtrcs.rx_rdma_cnp_pkts);
    insert_event_and_sample(enabled.bits.tx_rdma_cnp_pkts,
                            trait::name<category::amd_smi_nic_tx_cnp_pkts>::value,
                            "ainic_tx_rdma_cnp_pkts", mtrcs.tx_rdma_cnp_pkts);
    insert_event_and_sample(enabled.bits.tx_rdma_ack_timeout,
                            trait::name<category::amd_smi_nic_tx_rdma_ack_timeout>::value,
                            "ainic_tx_rdma_ack_timeout", mtrcs.tx_rdma_ack_timeout);
    insert_event_and_sample(enabled.bits.resp_tx_pkt_seq_err,
                            trait::name<category::amd_smi_nic_resp_tx_pkt_seq_err>::value,
                            "ainic_resp_tx_pkt_seq_err", mtrcs.resp_tx_pkt_seq_err);
    insert_event_and_sample(enabled.bits.req_rx_pkt_seq_err,
                            trait::name<category::amd_smi_nic_req_rx_pkt_seq_err>::value,
                            "ainic_req_rx_pkt_seq_err", mtrcs.req_rx_pkt_seq_err);
    insert_event_and_sample(
        enabled.bits.req_rx_impl_nak_seq_err,
        trait::name<category::amd_smi_nic_req_rx_impl_nak_seq_err>::value,
        "ainic_req_rx_impl_nak_seq_err", mtrcs.req_rx_impl_nak_seq_err);
}

void
rocpd_processor_t::handle(
    [[maybe_unused]] const gpu_perf_counter_sample& gpu_perf_counter)
{
    if(gpu_perf_counter.entries.empty()) return;

    const auto*  name         = "rocm_counter_collection";
    const auto&  process_info = m_metadata->get_process_info();
    const agent* agent_ptr    = nullptr;
    try
    {
        agent_ptr = &m_agent_manager->get_agent_by_type_index(gpu_perf_counter.device_id,
                                                              agent_type::GPU);
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("GPU perf-counter sample skipped: agent lookup failed for "
                    "device_id={}: {}",
                    gpu_perf_counter.device_id, e.what());
        return;
    }

    const auto& agent_ref = *agent_ptr;
    const auto  agent_uid = make_agent_uid(agent_ref);
    auto        event     = make_event(0, 0, 0, name);

    for(const auto& entry : gpu_perf_counter.entries)
    {
        auto name_info = m_metadata->find_gpu_perf_counter_by_id(
            gpu_perf_counter.device_id, entry.counter_id);
        if(!name_info) continue;

        const auto& info = name_info->get();

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = event;
        pmc_data.value = entry.value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = info.track_name.c_str();
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = gpu_perf_counter.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = info.pmc_info_name;
        pmc_uid.agent_id = agent_uid;

        try_insert_pmc_event(pmc_data, pmc_uid, "GPU perf-counter sample");
    }
}

void
rocpd_processor_t::handle([[maybe_unused]] const cpu_pmc_sample& cpu_pmc_smpl)
{
    struct core_freq_sample
    {
        size_t id;
        float  value;
    };

    struct core_load_sample
    {
        size_t id;
        double value;
    };

    auto deserialize_freqs = [](const std::vector<std::uint8_t>& buffer) {
        std::vector<core_freq_sample> result;
        size_t                        offset = 0;

        while(offset + sizeof(float) + sizeof(size_t) <= buffer.size())
        {
            core_freq_sample core_sample{};
            std::memcpy(&core_sample.id, buffer.data() + offset, sizeof(size_t));
            offset += sizeof(size_t);
            std::memcpy(&core_sample.value, buffer.data() + offset, sizeof(float));
            offset += sizeof(float);
            result.push_back(core_sample);
        }
        return result;
    };

    auto deserialize_loads = [](const std::vector<std::uint8_t>& buffer) {
        std::vector<core_load_sample> result;
        size_t                        offset = 0;

        while(offset + sizeof(double) + sizeof(size_t) <= buffer.size())
        {
            core_load_sample core_sample{};
            std::memcpy(&core_sample.id, buffer.data() + offset, sizeof(size_t));
            offset += sizeof(size_t);
            std::memcpy(&core_sample.value, buffer.data() + offset, sizeof(double));
            offset += sizeof(double);
            result.push_back(core_sample);
        }
        return result;
    };

    const auto* name         = trait::name<category::cpu_freq>::value;
    const auto& process_info = m_metadata->get_process_info();

    const auto device_id = static_cast<size_t>(cpu_pmc_smpl.device_id);

    const agent* agent_ptr = nullptr;
    try
    {
        agent_ptr = &m_agent_manager->get_agent_by_type_index(device_id, agent_type::CPU);
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("CPU PMC sample skipped: agent lookup failed for device_id={}: {}",
                    device_id, e.what());
        return;
    }

    const auto& agent_ref = *agent_ptr;
    const auto  agent_uid = make_agent_uid(agent_ref);

    auto event = make_event(0, 0, 0, name);

    auto insert_event_and_sample = [&](const char* pmc_name, const char* track_name,
                                       double value) {
        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = event;
        pmc_data.value = value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = track_name;
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = cpu_pmc_smpl.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = pmc_name;
        pmc_uid.agent_id = agent_uid;

        try_insert_pmc_event(pmc_data, pmc_uid, "CPU PMC sample");
    };

    const auto& enabled_m = cpu_pmc_smpl.enabled_metric;

    // Process-level metrics are global — emit once from the lowest selected socket
    static auto s_process_device_id = device_id;
    const bool  is_process_owner    = (device_id == s_process_device_id);

    if(is_process_owner)
    {
        if(enabled_m.bits.page_rss)
        {
            insert_event_and_sample(
                trait::name<category::process_page>::value,
                trait::name<category::process_page>::value,
                static_cast<double>(cpu_pmc_smpl.process_data.page_rss) /
                    units::megabyte);
        }

        if(enabled_m.bits.virt_mem)
        {
            insert_event_and_sample(
                trait::name<category::process_virt>::value,
                trait::name<category::process_virt>::value,
                static_cast<double>(cpu_pmc_smpl.process_data.virt_mem) /
                    units::megabyte);
        }

        if(enabled_m.bits.peak_rss)
        {
            insert_event_and_sample(
                trait::name<category::process_peak>::value,
                trait::name<category::process_peak>::value,
                static_cast<double>(cpu_pmc_smpl.process_data.peak_rss) /
                    units::megabyte);
        }

        if(enabled_m.bits.ctx_switches)
        {
            insert_event_and_sample(
                trait::name<category::process_context_switch>::value,
                trait::name<category::process_context_switch>::value,
                static_cast<double>(cpu_pmc_smpl.process_data.context_switches));
        }

        if(enabled_m.bits.page_faults)
        {
            insert_event_and_sample(
                trait::name<category::process_page_fault>::value,
                trait::name<category::process_page_fault>::value,
                static_cast<double>(cpu_pmc_smpl.process_data.page_faults));
        }

        if(enabled_m.bits.user_time)
        {
            insert_event_and_sample(
                trait::name<category::process_user_mode_time>::value,
                trait::name<category::process_user_mode_time>::value,
                static_cast<double>(cpu_pmc_smpl.process_data.user_mode_time) /
                    units::sec);
        }

        if(enabled_m.bits.kernel_time)
        {
            insert_event_and_sample(
                trait::name<category::process_kernel_mode_time>::value,
                trait::name<category::process_kernel_mode_time>::value,
                static_cast<double>(cpu_pmc_smpl.process_data.kernel_mode_time) /
                    units::sec);
        }
    }

    if(enabled_m.bits.frequency)
    {
        auto get_freq_track_name = [device_id](const auto& cpu_id) {
            return std::string(trait::name<category::cpu_freq>::value) + " [" +
                   std::to_string(device_id) + "] Core [" + std::to_string(cpu_id) + "]";
        };

        const auto core_freq_samples = deserialize_freqs(cpu_pmc_smpl.freqs);
        for(const auto& core : core_freq_samples)
        {
            auto track_name = get_freq_track_name(core.id);
            insert_event_and_sample(trait::name<category::cpu_freq>::value,
                                    track_name.c_str(), static_cast<double>(core.value));
        }
    }

    if(enabled_m.bits.load)
    {
        auto get_load_track_name = [device_id](const auto& cpu_id) {
            return std::string(trait::name<category::cpu_load>::value) + " [" +
                   std::to_string(device_id) + "] Core [" + std::to_string(cpu_id) + "]";
        };

        const auto core_load_samples = deserialize_loads(cpu_pmc_smpl.loads);
        for(const auto& core : core_load_samples)
        {
            auto track_name = get_load_track_name(core.id);
            insert_event_and_sample(trait::name<category::cpu_load>::value,
                                    track_name.c_str(), static_cast<double>(core.value));
        }
    }
}

void
rocpd_processor_t::handle(const kfd_sample& kfd)
{
    auto& n_info       = node_info::get_instance();
    auto  process_info = m_metadata->get_process_info();

    auto event    = make_event(0, 0, 0, kfd.category.c_str());
    event.extdata = kfd.event_metadata;

    profiler_hub::writer_types::region_data_t region;
    region.event           = event;
    region.start_timestamp = kfd.start_timestamp;
    region.end_timestamp   = kfd.end_timestamp;
    region.name            = kfd.name;

    auto parsed_args = process_arguments_string(kfd.args_str);
    for(const auto& arg : parsed_args)
    {
        profiler_hub::writer_types::arg_data_t arg_data;
        arg_data.position = arg.arg_number;
        arg_data.type     = arg.arg_type;
        arg_data.name     = arg.arg_name;
        arg_data.value    = arg.arg_value;
        region.args.push_back(arg_data);
    }

    auto env = make_trace_env(n_info.id, process_info.pid, kfd.thread_id);
    m_writer->insert_region_data(region, env);

    try
    {
        const auto& agent_ref = m_agent_manager->get_agent_by_type_index(
            kfd.device_id, static_cast<agent_type>(kfd.device_type));

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = event;
        pmc_data.value = kfd.value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = kfd.track_name.c_str();
        track.node_id    = n_info.id;
        track.process_id = process_info.pid;
        if(kfd.system_tid.has_value()) track.thread_id = kfd.system_tid.value();

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = kfd.start_timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = kfd.pmc_info_name;
        pmc_uid.agent_id = make_agent_uid(agent_ref);

        try_insert_pmc_event(pmc_data, pmc_uid, "KFD PMC event");
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("KFD PMC event skipped: agent lookup failed for device_id={}, "
                    "device_type={}: {}",
                    kfd.device_id, kfd.device_type, e.what());
    }
}

rocpd_processor_t::rocpd_processor_t(const std::shared_ptr<metadata_registry>& md,
                                     const std::shared_ptr<agent_manager>&     agent_mngr,
                                     int pid, int ppid,
                                     output_file_registry& output_registry)
: processor_t<rocpd_processor_t>()
, m_metadata(md)
, m_agent_manager(agent_mngr)
, m_output_registry(output_registry)
, m_db_output_path(generate_db_output_path(pid))
{
    auto n_info = node_info::get_instance();
    auto uuid   = common::md5sum{ n_info.id, pid, ppid }.hexdigest();

    // Remove any existing database file so the writer starts from a clean
    // in-memory state. This handles multiple rocprofsys_init/finalize cycles
    // in the same process (e.g. with ROCPROFSYS_USE_PID=OFF) where successive
    // sessions would otherwise find the file left by the previous session and
    // fail with "Database already initialized!".
    if(std::filesystem::exists(m_db_output_path))
    {
        LOG_WARNING("rocpd output file already exists and will be overwritten: {}. "
                    "Previous profiling data in that file will be lost. "
                    "Set ROCPROFSYS_USE_PID=ON to give each session a unique path.",
                    m_db_output_path);
        std::filesystem::remove(m_db_output_path);
    }

    auto storage = std::make_unique<profiler_hub::storage_t>(m_db_output_path, uuid);
    m_writer     = std::make_unique<profiler_hub::writer_t>(std::move(storage));
}

void
rocpd_processor_t::try_insert_pmc_event(
    const profiler_hub::writer_types::pmc_event_data_t&     event_data,
    const profiler_hub::writer_types::pmc_info_unique_id_t& unique_id,
    std::string_view                                        context)
{
    try
    {
        m_writer->insert_pmc_event_data(event_data, unique_id);
    } catch(const std::runtime_error& e)
    {
        ++m_dropped_pmc_events_count;

        auto key = std::string{ unique_id.name };

        // Build the key for the PMC info. Two agents missing the same PMC info will
        // warn separately.
        if(unique_id.agent_id.has_value())
        {
            const auto& agent_id = unique_id.agent_id.value();
            key += fmt::format(" [{}:{}]", agent_id.agent_type.value_or("unknown"),
                               agent_id.type_index);
        }

        if(m_unregistered_pmcs_already_warned.emplace(std::move(key)).second)
        {
            LOG_WARNING("{} skipped: PMC info not registered for name={} - {}. "
                        "Further samples for this PMC will be dropped without warning.",
                        context, unique_id.name, e.what());
        }
    }
}

void
rocpd_processor_t::prepare_for_processing()
{
    LOG_DEBUG("Preparing rocpd processor for processing");
    post_process_metadata();
    LOG_TRACE("Rocpd processor prepared for processing");
}

void
rocpd_processor_t::finalize_processing()
{
    LOG_DEBUG("Finalizing rocpd processor");
    try
    {
        m_writer->flush_in_memory_data_to_disk();
    } catch(const std::exception& e)
    {
        // This can happen in multi-process scenarios when two processes attempt
        // to flush their in-memory databases to the same output file at the same
        // time (SQLITE_BUSY). The underlying fix belongs in profiler-hub (retry
        // with back-off or per-PID filenames), but crashing is never appropriate.
        LOG_ERROR("Failed to flush rocpd database to disk ({}): {}. "
                  "Profile data for this process may be incomplete.",
                  m_db_output_path, e.what());
        return;
    }

    m_output_registry.register_file(m_db_output_path, output_format::rocpd);

    if(m_dropped_pmc_events_count > 0)
    {
        // Sorted so the message is reproducible across runs.
        auto counters =
            std::vector<std::string>{ m_unregistered_pmcs_already_warned.begin(),
                                      m_unregistered_pmcs_already_warned.end() };
        std::ranges::sort(counters);

        LOG_WARNING("Rocpd processor finalized with {} PMC event(s) dropped across {} "
                    "unregistered counter(s); {} is incomplete. Counters: {}",
                    m_dropped_pmc_events_count, counters.size(), m_db_output_path,
                    fmt::join(counters, ", "));
    }
    else
    {
        LOG_INFO("Rocpd processor finalized successfully");
    }
}

void
rocpd_processor_t::post_process_metadata()
{
    if(!get_use_rocpd())
    {
        LOG_TRACE("Rocpd not enabled, skipping metadata post-processing");
        return;
    }
    LOG_DEBUG("Post-processing metadata for rocpd");
    auto n_info = node_info::get_instance();

    // Register node info
    profiler_hub::writer_types::node_info_t node;
    node.node_id       = n_info.id;
    node.hash          = n_info.hash;
    node.machine_id    = n_info.machine_id;
    node.system_name   = n_info.system_name;
    node.hostname      = n_info.node_name;
    node.release       = n_info.release;
    node.version       = n_info.version;
    node.hardware_name = n_info.machine;
    node.domain_name   = n_info.domain_name;
    m_writer->register_node_info(node);

    // Register process info
    auto process_info = m_metadata->get_process_info();
    profiler_hub::writer_types::process_info_t proc;
    proc.ppid    = process_info.ppid;
    proc.pid     = process_info.pid;
    proc.init    = 0;
    proc.fini    = 0;
    proc.start   = process_info.start;
    proc.end     = process_info.end;
    proc.command = process_info.command.c_str();
    proc.environment =
        process_info.environment.empty() ? "{}" : process_info.environment.c_str();
    proc.extdata = process_info.extdata.empty() ? "{}" : process_info.extdata.c_str();
    proc.node_id = n_info.id;
    m_writer->register_process_info(proc);

    // Register agents
    const auto& agents  = m_agent_manager->get_agents();
    int         counter = 0;
    for(const auto& rocpd_agent : agents)
    {
        profiler_hub::writer_types::agent_info_t agent_info;
        agent_info.unique_id      = make_agent_uid(*rocpd_agent);
        agent_info.absolute_index = static_cast<size_t>(counter++);

        agent_info.logical_index = rocpd_agent->logical_node_id;
        agent_info.uuid          = rocpd_agent->device_id;
        agent_info.name          = rocpd_agent->name;
        agent_info.model_name    = rocpd_agent->model_name;
        agent_info.vendor_name   = rocpd_agent->vendor_name;
        agent_info.product_name  = rocpd_agent->product_name;
        agent_info.user_name     = rocpd_agent->product_name;
        agent_info.extdata       = rocpd_agent->agent_info;
        agent_info.node_id       = n_info.id;
        agent_info.process_id    = process_info.pid;
        m_writer->register_agent_info(agent_info);
    }

    // Register strings
    auto string_list = m_metadata->get_string_list();
    for(auto& str : string_list)
    {
        m_writer->register_string(str);
    }

    // Register thread info
    auto thread_info_list = m_metadata->get_thread_info_list();
    for(auto& t_info : thread_info_list)
    {
        const auto& extended_info =
            thread_info::get(static_cast<std::int64_t>(t_info.thread_id), SystemTID);
        if(extended_info.has_value())
        {
            t_info.start = extended_info->get_start();
            t_info.end   = extended_info->get_stop();
        }

        auto thread_name = fmt::format("Thread {}", t_info.thread_id);

        profiler_hub::writer_types::thread_info_t thread_info;
        thread_info.parent_process_id = process_info.ppid;
        thread_info.thread_id         = t_info.thread_id;
        thread_info.name              = thread_name;
        thread_info.start             = t_info.start;
        thread_info.end               = t_info.end;
        thread_info.node_id           = n_info.id;
        thread_info.process_id        = process_info.pid;
        m_writer->register_thread_info(thread_info);
    }

    // Register tracks
    auto track_info_list = m_metadata->get_track_info_list();
    for(auto& track : track_info_list)
    {
        profiler_hub::writer_types::track_info_t track_info;
        track_info.name       = track.track_name;
        track_info.node_id    = n_info.id;
        track_info.process_id = process_info.pid;
        track_info.thread_id  = track.thread_id;
        m_writer->register_track_info(track_info);
    }

    // Register code objects
    auto code_object_list = m_metadata->get_code_object_list();
    for(const auto& code_object : code_object_list)
    {
        const auto& code_agent = m_agent_manager->get_agent_by_handle(
            get_handle_from_code_object(code_object));

        const char* strg_type = "UNKNOWN";
        switch(code_object.storage_type)
        {
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE: strg_type = "FILE"; break;
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY: strg_type = "MEMORY"; break;
            default: break;
        }

        profiler_hub::writer_types::code_object_info_t co;
        co.id           = code_object.code_object_id;
        co.uri          = code_object.uri;
        co.load_base    = code_object.load_base;
        co.load_size    = code_object.load_size;
        co.load_delta   = code_object.load_delta;
        co.storage_type = strg_type;
        co.node_id      = n_info.id;
        co.process_id   = process_info.pid;
        co.agent_id     = make_agent_uid(code_agent);
        m_writer->register_code_object_info(co);
    }

    // Register kernel symbols
    auto kernel_symbols_list = m_metadata->get_kernel_symbol_list();
    for(const auto& kernel_symbol : kernel_symbols_list)
    {
        auto kernel_name = rocprofsys::utility::demangle(kernel_symbol.kernel_name);

        profiler_hub::writer_types::kernel_symbol_info_t ksi;
        ksi.id                        = kernel_symbol.kernel_id;
        ksi.name                      = kernel_symbol.kernel_name;
        ksi.display_name              = kernel_name;
        ksi.kernel_object             = kernel_symbol.kernel_object;
        ksi.kernarg_segment_size      = kernel_symbol.kernarg_segment_size;
        ksi.kernarg_segment_alignment = kernel_symbol.kernarg_segment_alignment;
        ksi.group_segment_size        = kernel_symbol.group_segment_size;
        ksi.private_segment_size      = kernel_symbol.private_segment_size;
        ksi.sgpr_count                = kernel_symbol.sgpr_count;
        ksi.arch_vgpr_count           = kernel_symbol.arch_vgpr_count;
        ksi.accum_vgpr_count          = kernel_symbol.accum_vgpr_count;
        ksi.node_id                   = n_info.id;
        ksi.process_id                = process_info.pid;
        ksi.code_obj_id               = kernel_symbol.code_object_id;
        m_writer->register_kernel_symbol_info(ksi);

        m_writer->register_string(kernel_name);
    }

    // Register queue info
    auto queue_list = m_metadata->get_queue_list();
    for(const auto& queue_handle : queue_list)
    {
        auto queue_name = fmt::format("Queue {}", queue_handle);

        profiler_hub::writer_types::queue_info_t qi;
        qi.queue_id   = queue_handle;
        qi.name       = queue_name;
        qi.node_id    = n_info.id;
        qi.process_id = process_info.pid;
        m_writer->register_queue_info(qi);
    }

    // Register stream info
    auto stream_list = m_metadata->get_stream_list();
    for(const auto& stream_handle : stream_list)
    {
        auto stream_name = fmt::format("Stream {}", stream_handle);

        profiler_hub::writer_types::stream_info_t str_info;
        str_info.stream_id  = stream_handle;
        str_info.name       = stream_name;
        str_info.node_id    = n_info.id;
        str_info.process_id = process_info.pid;
        m_writer->register_stream_info(str_info);
    }

    // Register buffer info strings
    auto buffer_info_list = m_metadata->get_buffer_name_info();
    for(const auto& buffer_info : buffer_info_list)
    {
        for(const auto& item : buffer_info.items())
        {
            m_writer->register_string(*item.second);
        }
    }

    // Register callback tracing strings
    auto callback_info_list = m_metadata->get_callback_tracing_info();
    for(const auto& cb_info : callback_info_list)
    {
        for(const auto& item : cb_info.items())
        {
            m_writer->register_string(*item.second);
        }
    }

    // Register PMC info
    auto pmc_info_list = m_metadata->get_pmc_info_list();
    for(const auto& pmc_info : pmc_info_list)
    {
        constexpr std::array<agent_type, 3> k_known_agent_types = {
            agent_type::GPU,
            agent_type::CPU,
            agent_type::NIC,
        };

        const bool agent_type_found =
            std::ranges::find(k_known_agent_types, pmc_info.type) !=
            k_known_agent_types.end();

        const agent* pmc_agent_ptr = nullptr;
        try
        {
            pmc_agent_ptr = agent_type_found
                                ? &m_agent_manager->get_agent_by_type_index(
                                      pmc_info.agent_type_index, pmc_info.type)
                                : &m_agent_manager->get_agent_by_id(
                                      pmc_info.agent_type_index, pmc_info.type);
        } catch(const std::out_of_range& e)
        {
            LOG_WARNING("PMC info registration skipped: agent lookup failed for "
                        "agent_type_index={}, type={}: {}",
                        pmc_info.agent_type_index, to_string(pmc_info.type), e.what());
            continue;
        }

        const auto& pmc_agent     = *pmc_agent_ptr;
        auto        pmc_agent_uid = make_agent_uid(pmc_agent);

        LOG_TRACE("Inserting PMC description: agent_uid: {}, pmc_info: {}",
                  pmc_agent_uid.type_index, pmc_info.name);

        profiler_hub::writer_types::pmc_info_t           pmc_info_data;
        profiler_hub::writer_types::pmc_info_unique_id_t uid;
        uid.name                = pmc_info.name;
        uid.agent_id            = pmc_agent_uid;
        pmc_info_data.unique_id = uid;
        pmc_info_data.target_arch =
            agent_type_found ? std::optional<std::string_view>{ pmc_info.target_arch }
                             : std::nullopt;
        pmc_info_data.event_code       = pmc_info.event_code;
        pmc_info_data.instance_id      = pmc_info.instance_id;
        pmc_info_data.symbol           = pmc_info.symbol;
        pmc_info_data.description      = pmc_info.description;
        pmc_info_data.long_description = pmc_info.long_description;
        pmc_info_data.component        = pmc_info.component;
        pmc_info_data.units            = pmc_info.units;
        pmc_info_data.value_type       = pmc_info.value_type;
        pmc_info_data.block            = pmc_info.block;
        pmc_info_data.expression       = pmc_info.expression;
        pmc_info_data.is_constant      = pmc_info.is_constant;
        pmc_info_data.is_derived       = pmc_info.is_derived;
        pmc_info_data.node_id          = n_info.id;
        pmc_info_data.process_id       = process_info.pid;
        m_writer->register_pmc_info(pmc_info_data);
    }
}

}  // namespace rocprofsys::trace_cache
