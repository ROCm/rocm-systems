// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/trace_cache/rocpd_processor.hpp"
#include "common/md5sum.hpp"
#include "core/agent_manager.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/node_info.hpp"
#include "core/output_file_registry.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/rocpd_helpers.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer.hpp>
#include <profiler-hub/writer_types.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "library/rocprofiler-sdk/fwd.hpp"
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/version.h>

namespace rocprofsys
{
namespace trace_cache
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
    auto _tag    = std::to_string(pid);
    auto db_name = std::string{ "rocpd" };
    return rocprofsys::get_database_absolute_path(db_name, _tag);
}

}  // namespace

void
rocpd_processor_t::handle(const kernel_dispatch_sample& _kds)
{
    auto& n_info    = node_info::get_instance();
    auto  process   = m_metadata->get_process_info();
    auto& agent_ref = m_agent_manager->get_agent_by_handle(_kds.agent_id_handle);

    auto kernel_symbol = m_metadata->get_kernel_symbol(_kds.kernel_id);
    if(!kernel_symbol.has_value())
    {
        throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
    }

    auto kernel_name = rocprofsys::utility::demangle(kernel_symbol->kernel_name);

    auto ev = make_event(_kds.correlation_id_internal, _kds.correlation_id_ancestor, 0,
                         trait::name<category::rocm_kernel_dispatch>::value);

    profiler_hub::writer_types::kernel_dispatch_data_t kd;
    kd.event                = ev;
    kd.dispatch_id          = _kds.dispatch_id;
    kd.start_timestamp      = _kds.start_timestamp;
    kd.end_timestamp        = _kds.end_timestamp;
    kd.kernel_symbol_id     = _kds.kernel_id;
    kd.private_segment_size = _kds.private_segment_size;
    kd.group_segment_size   = _kds.group_segment_size;
    kd.workgroup_size_x     = _kds.workgroup_size_x;
    kd.workgroup_size_y     = _kds.workgroup_size_y;
    kd.workgroup_size_z     = _kds.workgroup_size_z;
    kd.grid_size_x          = _kds.grid_size_x;
    kd.grid_size_y          = _kds.grid_size_y;
    kd.grid_size_z          = _kds.grid_size_z;
    kd.name                 = kernel_name.c_str();

    auto env = make_trace_env_with_agent_queue_stream(
        n_info.id, process.pid, _kds.thread_id, agent_ref, _kds.queue_id_handle,
        _kds.stream_handle);

    m_writer->insert_kernel_dispatch_data(kd, env);
}

void
rocpd_processor_t::handle(const scratch_memory_sample& _sms)
{
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    const auto* _name = m_metadata->get_buffer_name_info().at(
        static_cast<rocprofiler_buffer_tracing_kind_t>(_sms.kind),
        static_cast<rocprofiler_tracing_operation_t>(_sms.operation));

    auto& agent_ref = m_agent_manager->get_agent_by_handle(_sms.agent_id_handle);

    auto [memory_operation, memory_type_val] = parse_memory_operation_name(_name);
    auto extdata_json_str = fmt::format("{{\"flags\": {}}}", _sms.flags);

    auto ev = make_event(_sms.correlation_id_internal, _sms.correlation_id_ancestor, 0,
                         trait::name<category::rocm_scratch_memory>::value);

    profiler_hub::writer_types::memory_alloc_data_t ma;
    ma.event           = ev;
    ma.type            = memory_operation.c_str();
    ma.level           = memory_type_val.c_str();
    ma.start_timestamp = _sms.start_timestamp;
    ma.end_timestamp   = _sms.end_timestamp;
    ma.address         = 0;
    ma.size            = _sms.allocation_size;
    ma.extdata         = extdata_json_str.c_str();

    auto env = make_trace_env_with_agent_queue_stream(
        n_info.id, process.pid, _sms.thread_id, agent_ref, _sms.queue_id_handle,
        _sms.stream_handle);

    m_writer->insert_memory_alloc_data(ma, env);
}

void
rocpd_processor_t::handle(const memory_copy_sample& _mcs)
{
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    auto _name = std::string{ m_metadata->get_buffer_name_info().at(
        static_cast<rocprofiler_buffer_tracing_kind_t>(_mcs.kind),
        static_cast<rocprofiler_tracing_operation_t>(_mcs.operation)) };

    auto& dst_agent = m_agent_manager->get_agent_by_handle(_mcs.dst_agent_id_handle);
    auto& src_agent = m_agent_manager->get_agent_by_handle(_mcs.src_agent_id_handle);

    auto ev = make_event(_mcs.correlation_id_internal, _mcs.correlation_id_ancestor, 0,
                         trait::name<category::rocm_memory_copy>::value);

    profiler_hub::writer_types::memory_copy_data_t mc;
    mc.event           = ev;
    mc.start_timestamp = _mcs.start_timestamp;
    mc.end_timestamp   = _mcs.end_timestamp;
    mc.dst_agent_id    = make_agent_uid(dst_agent);
    mc.dst_address     = _mcs.dst_address_value;
    mc.src_agent_id    = make_agent_uid(src_agent);
    mc.src_address     = _mcs.src_address_value;
    mc.size            = _mcs.bytes;
    mc.name            = _name.c_str();
    mc.region_name     = _name.c_str();

    auto env      = make_trace_env(n_info.id, process.pid, _mcs.thread_id);
    env.stream_id = _mcs.stream_handle;
    env.queue_id  = 0;

    m_writer->insert_memory_copy_data(mc, env);
}

void
rocpd_processor_t::handle([[maybe_unused]] const memory_allocate_sample& _mas)
{
#if(ROCPROFILER_VERSION >= 600)
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    const auto invalid_context = ROCPROFILER_CONTEXT_NONE;
    if(_mas.agent_id_handle != invalid_context.handle)
    {
        auto& agent_ref = m_agent_manager->get_agent_by_handle(_mas.agent_id_handle);

        const auto* _name = m_metadata->get_buffer_name_info().at(
            static_cast<rocprofiler_buffer_tracing_kind_t>(_mas.kind),
            static_cast<rocprofiler_tracing_operation_t>(_mas.operation));

        auto [memory_operation, memory_type_val] = parse_memory_operation_name(_name);

        auto ev = make_event(_mas.correlation_id_internal, _mas.correlation_id_ancestor,
                             0, trait::name<category::rocm_memory_allocate>::value);

        profiler_hub::writer_types::memory_alloc_data_t ma;
        ma.event           = ev;
        ma.type            = memory_operation.c_str();
        ma.level           = memory_type_val.c_str();
        ma.start_timestamp = _mas.start_timestamp;
        ma.end_timestamp   = _mas.end_timestamp;
        ma.address         = _mas.address_value;
        ma.size            = _mas.allocation_size;

        auto env =
            make_trace_env_with_agent(n_info.id, process.pid, _mas.thread_id, agent_ref);
        env.stream_id = _mas.stream_handle;
        env.queue_id  = 0;

        m_writer->insert_memory_alloc_data(ma, env);
    }
#endif
}

void
rocpd_processor_t::handle(const region_sample& _rs)
{
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    auto ev = make_event(_rs.correlation_id_internal, _rs.correlation_id_ancestor, 0,
                         _rs.category.c_str());
    ev.call_stack.push_back({});
    // call_stack and line_info are serialized JSON in the old code; in profiler-hub
    // they are structured types. For now pass the raw JSON via extdata.
    ev.extdata = _rs.call_stack.c_str();

    profiler_hub::writer_types::region_data_t region;
    region.event           = ev;
    region.start_timestamp = _rs.start_timestamp;
    region.end_timestamp   = _rs.end_timestamp;
    region.name            = _rs.name.c_str();

    auto parsed_args = process_arguments_string(_rs.args_str);
    for(const auto& arg : parsed_args)
    {
        profiler_hub::writer_types::arg_data_t ad;
        ad.position = arg.arg_number;
        ad.type     = arg.arg_type.c_str();
        ad.name     = arg.arg_name.c_str();
        ad.value    = arg.arg_value.c_str();
        region.args.push_back(ad);
    }

    auto env = make_trace_env(n_info.id, process.pid, _rs.thread_id);
    m_writer->insert_region_data(region, env);
}

void
rocpd_processor_t::handle(const backtrace_region_sample& _bts)
{
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    auto ev = make_event(0, 0, 0, _bts.category.c_str());
    ev.call_stack.push_back({});
    // call_stack and line_info are serialized JSON in the old code; in profiler-hub
    // they are structured types. For now pass the raw JSON via extdata.
    ev.extdata = _bts.call_stack.c_str();

    profiler_hub::writer_types::region_data_t region;
    region.event           = ev;
    region.start_timestamp = _bts.start_timestamp;
    region.end_timestamp   = _bts.end_timestamp;
    region.name            = _bts.name.c_str();

    auto env       = make_trace_env(n_info.id, process.pid, _bts.thread_id);
    env.track_name = _bts.track_name.c_str();

    m_writer->insert_region_data(region, env);
}

void
rocpd_processor_t::handle(const in_time_sample& _its)
{
    auto ev    = make_event(_its.stack_id, _its.parent_stack_id, _its.correlation_id,
                            _its.track_name.c_str());
    ev.extdata = _its.event_metadata.c_str();

    profiler_hub::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event = ev;
    pmc_data.value = 0.0;

    profiler_hub::writer_types::track_info_t track;
    track.name = _its.track_name.c_str();

    profiler_hub::writer_types::sample_data_t sample;
    sample.timestamp = _its.timestamp_ns;
    sample.track     = track;
    pmc_data.sample  = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
    pmc_uid.name = _its.track_name.c_str();

    m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
}

void
rocpd_processor_t::handle(const pmc_event_with_sample& _pmc)
{
    const auto& process_info = m_metadata->get_process_info();
    const auto& agent_ref    = m_agent_manager->get_agent_by_type_index(
        _pmc.device_id, static_cast<agent_type>(_pmc.device_type));

    auto ev    = make_event(_pmc.stack_id, _pmc.parent_stack_id, _pmc.correlation_id,
                            _pmc.track_name.c_str());
    ev.extdata = _pmc.event_metadata.c_str();

    profiler_hub::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event = ev;
    pmc_data.value = _pmc.value;

    profiler_hub::writer_types::track_info_t track;
    track.name       = _pmc.track_name.c_str();
    track.node_id    = node_info::get_instance().id;
    track.process_id = process_info.pid;
    track.thread_id  = _pmc.system_tid;

    profiler_hub::writer_types::sample_data_t sample;
    sample.timestamp = _pmc.timestamp_ns;
    sample.track     = track;
    pmc_data.sample  = sample;

    profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
    pmc_uid.name     = _pmc.pmc_info_name.c_str();
    pmc_uid.agent_id = make_agent_uid(agent_ref);

    m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
}

void
rocpd_processor_t::handle([[maybe_unused]] const gpu_pmc_sample& _gpu_pmc)
{
    const auto* _name        = trait::name<category::amd_smi>::value;
    const auto& process_info = m_metadata->get_process_info();
    const auto& agent_ref =
        m_agent_manager->get_agent_by_type_index(_gpu_pmc.device_id, agent_type::GPU);

    const auto agent_uid = make_agent_uid(agent_ref);

    auto ev = make_event(0, 0, 0, _name);

    auto insert_event_and_sample = [&](bool is_enabled, const char* pmc_name,
                                       const char* track_name, double value) {
        if(!is_enabled) return;

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = ev;
        pmc_data.value = value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = track_name;
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = _gpu_pmc.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = pmc_name;
        pmc_uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
    };

    const auto& m       = _gpu_pmc.metric_values;
    const auto& enabled = _gpu_pmc.enabled_metric;

    auto insert_scalar = [&](const char* name, const std::string& track, bool is_enabled,
                             double value) {
        insert_event_and_sample(is_enabled, name, track.c_str(), value);
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
                  enabled.bits.hotspot_temperature, m.hotspot_temperature);
    insert_scalar(trait::name<category::amd_smi_power>::value,
                  info::format_track_name<category::amd_smi_power>(),
                  enabled.bits.current_socket_power || enabled.bits.average_socket_power,
                  enabled.bits.current_socket_power ? m.current_socket_power
                                                    : m.average_socket_power);
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
                    continue;
                auto name = format_name(static_cast<int>(xcp), static_cast<int>(i));
                insert_event_and_sample(true, name.c_str(), name.c_str(),
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
        for(size_t i = 0; i < arr.size(); ++i)
        {
            if(arr[i] == pmc::collectors::gpu::METRIC_VALUE_NOT_SUPPORTED_16) continue;

            auto suffix     = "_" + std::to_string(i);
            auto pmc_name   = std::string(base_name) + suffix;
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

            std::string pmc_name = base_track_name + "_link" + std::to_string(i);
            std::string track_name =
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
rocpd_processor_t::handle([[maybe_unused]] const ainic_pmc_sample& _nic_sample)
{
    // Insert NIC RDMA metrics into rocpd database
    const auto* _name        = trait::name<category::amd_smi_nic>::value;
    const auto& process_info = m_metadata->get_process_info();
    const auto& nic_agent =
        m_agent_manager->get_agent_by_id(_nic_sample.device_id, agent_type::NIC);

    const auto agent_uid = make_agent_uid(nic_agent);

    auto ev = make_event(0, 0, 0, _name);

    auto insert_event_and_sample = [&](bool is_enabled, const char* pmc_name,
                                       const char* track_name, std::uint64_t value) {
        if(!is_enabled) return;

        LOG_TRACE("Inserting metric: pmc_name: {}, track_name: {}, value: {}", pmc_name,
                  track_name, value);

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = ev;
        pmc_data.value = static_cast<double>(value);

        profiler_hub::writer_types::track_info_t track;
        track.name       = track_name;
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = _nic_sample.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = pmc_name;
        pmc_uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
    };

    const auto& m       = _nic_sample.metric_values;
    const auto& enabled = _nic_sample.enabled_metric;

    insert_event_and_sample(enabled.bits.rx_rdma_ucast_bytes,
                            trait::name<category::amd_smi_nic_rx_ucast_bytes>::value,
                            "ainic_rx_rdma_ucast_bytes", m.rx_rdma_ucast_bytes);
    insert_event_and_sample(enabled.bits.tx_rdma_ucast_bytes,
                            trait::name<category::amd_smi_nic_tx_ucast_bytes>::value,
                            "ainic_tx_rdma_ucast_bytes", m.tx_rdma_ucast_bytes);
    insert_event_and_sample(enabled.bits.rx_rdma_ucast_pkts,
                            trait::name<category::amd_smi_nic_rx_ucast_pkts>::value,
                            "ainic_rx_rdma_ucast_pkts", m.rx_rdma_ucast_pkts);
    insert_event_and_sample(enabled.bits.tx_rdma_ucast_pkts,
                            trait::name<category::amd_smi_nic_tx_ucast_pkts>::value,
                            "ainic_tx_rdma_ucast_pkts", m.tx_rdma_ucast_pkts);
    insert_event_and_sample(enabled.bits.rx_rdma_cnp_pkts,
                            trait::name<category::amd_smi_nic_rx_cnp_pkts>::value,
                            "ainic_rx_rdma_cnp_pkts", m.rx_rdma_cnp_pkts);
    insert_event_and_sample(enabled.bits.tx_rdma_cnp_pkts,
                            trait::name<category::amd_smi_nic_tx_cnp_pkts>::value,
                            "ainic_tx_rdma_cnp_pkts", m.tx_rdma_cnp_pkts);
    insert_event_and_sample(enabled.bits.tx_rdma_ack_timeout,
                            trait::name<category::amd_smi_nic_tx_rdma_ack_timeout>::value,
                            "ainic_tx_rdma_ack_timeout", m.tx_rdma_ack_timeout);
    insert_event_and_sample(enabled.bits.resp_tx_pkt_seq_err,
                            trait::name<category::amd_smi_nic_resp_tx_pkt_seq_err>::value,
                            "ainic_resp_tx_pkt_seq_err", m.resp_tx_pkt_seq_err);
    insert_event_and_sample(enabled.bits.req_rx_pkt_seq_err,
                            trait::name<category::amd_smi_nic_req_rx_pkt_seq_err>::value,
                            "ainic_req_rx_pkt_seq_err", m.req_rx_pkt_seq_err);
    insert_event_and_sample(
        enabled.bits.req_rx_impl_nak_seq_err,
        trait::name<category::amd_smi_nic_req_rx_impl_nak_seq_err>::value,
        "ainic_req_rx_impl_nak_seq_err", m.req_rx_impl_nak_seq_err);
}

void
rocpd_processor_t::handle(
    [[maybe_unused]] const gpu_perf_counter_sample& _gpu_perf_counter)
{
    if(_gpu_perf_counter.entries.empty()) return;

    const auto* _name        = "rocm_counter_collection";
    const auto& process_info = m_metadata->get_process_info();
    const auto& agent_ref    = m_agent_manager->get_agent_by_type_index(
        _gpu_perf_counter.device_id, agent_type::GPU);

    const auto agent_uid = make_agent_uid(agent_ref);
    auto       ev        = make_event(0, 0, 0, _name);

    for(const auto& entry : _gpu_perf_counter.entries)
    {
        auto name_info = m_metadata->find_gpu_perf_counter_by_id(
            _gpu_perf_counter.device_id, entry.counter_id);
        if(!name_info) continue;

        const auto& info = name_info->get();

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = ev;
        pmc_data.value = entry.value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = info.track_name.c_str();
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = _gpu_perf_counter.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = info.pmc_info_name.c_str();
        pmc_uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
    }
}

void
rocpd_processor_t::handle([[maybe_unused]] const cpu_pmc_sample& _cpu_pmc_sample)
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
            core_freq_sample core_sample;
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
            core_load_sample core_sample;
            std::memcpy(&core_sample.id, buffer.data() + offset, sizeof(size_t));
            offset += sizeof(size_t);
            std::memcpy(&core_sample.value, buffer.data() + offset, sizeof(double));
            offset += sizeof(double);
            result.push_back(core_sample);
        }
        return result;
    };

    const auto* _name        = trait::name<category::cpu_freq>::value;
    const auto& process_info = m_metadata->get_process_info();

    const auto device_id = static_cast<size_t>(_cpu_pmc_sample.device_id);

    const auto& agent_ref =
        m_agent_manager->get_agent_by_type_index(device_id, agent_type::CPU);

    const auto agent_uid = make_agent_uid(agent_ref);

    auto ev = make_event(0, 0, 0, _name);

    auto insert_event_and_sample = [&](const char* pmc_name, const char* track_name,
                                       double value) {
        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = ev;
        pmc_data.value = value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = track_name;
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = _cpu_pmc_sample.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = pmc_name;
        pmc_uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
    };

    const auto& _em = _cpu_pmc_sample.enabled_metric;

    // Process-level metrics are global — emit once from the lowest selected socket
    static auto s_process_device_id = device_id;
    const bool  is_process_owner    = (device_id == s_process_device_id);

    if(is_process_owner)
    {
        if(_em.bits.page_rss)
            insert_event_and_sample(
                trait::name<category::process_page>::value,
                trait::name<category::process_page>::value,
                static_cast<double>(_cpu_pmc_sample.process_data.page_rss) /
                    units::megabyte);

        if(_em.bits.virt_mem)
            insert_event_and_sample(
                trait::name<category::process_virt>::value,
                trait::name<category::process_virt>::value,
                static_cast<double>(_cpu_pmc_sample.process_data.virt_mem) /
                    units::megabyte);

        if(_em.bits.peak_rss)
            insert_event_and_sample(
                trait::name<category::process_peak>::value,
                trait::name<category::process_peak>::value,
                static_cast<double>(_cpu_pmc_sample.process_data.peak_rss) /
                    units::megabyte);

        if(_em.bits.ctx_switches)
            insert_event_and_sample(trait::name<category::process_context_switch>::value,
                                    trait::name<category::process_context_switch>::value,
                                    _cpu_pmc_sample.process_data.context_switches);

        if(_em.bits.page_faults)
            insert_event_and_sample(trait::name<category::process_page_fault>::value,
                                    trait::name<category::process_page_fault>::value,
                                    _cpu_pmc_sample.process_data.page_faults);

        if(_em.bits.user_time)
            insert_event_and_sample(
                trait::name<category::process_user_mode_time>::value,
                trait::name<category::process_user_mode_time>::value,
                static_cast<double>(_cpu_pmc_sample.process_data.user_mode_time) /
                    units::sec);

        if(_em.bits.kernel_time)
            insert_event_and_sample(
                trait::name<category::process_kernel_mode_time>::value,
                trait::name<category::process_kernel_mode_time>::value,
                static_cast<double>(_cpu_pmc_sample.process_data.kernel_mode_time) /
                    units::sec);
    }

    if(_em.bits.frequency)
    {
        auto get_freq_track_name = [device_id](const auto& cpu_id) {
            return std::string(trait::name<category::cpu_freq>::value) + " [" +
                   std::to_string(device_id) + "] Core [" + std::to_string(cpu_id) + "]";
        };

        const auto core_freq_samples = deserialize_freqs(_cpu_pmc_sample.freqs);
        for(const auto& core : core_freq_samples)
        {
            auto track_name = get_freq_track_name(core.id);
            insert_event_and_sample(trait::name<category::cpu_freq>::value,
                                    track_name.c_str(), static_cast<double>(core.value));
        }
    }

    if(_em.bits.load)
    {
        auto get_load_track_name = [device_id](const auto& cpu_id) {
            return std::string(trait::name<category::cpu_load>::value) + " [" +
                   std::to_string(device_id) + "] Core [" + std::to_string(cpu_id) + "]";
        };

        const auto core_load_samples = deserialize_loads(_cpu_pmc_sample.loads);
        for(const auto& core : core_load_samples)
        {
            auto track_name = get_load_track_name(core.id);
            insert_event_and_sample(trait::name<category::cpu_load>::value,
                                    track_name.c_str(), static_cast<double>(core.value));
        }
    }
}

void
rocpd_processor_t::handle(const kfd_sample& _kfd)
{
    auto& n_info       = node_info::get_instance();
    auto  process_info = m_metadata->get_process_info();

    auto ev    = make_event(0, 0, 0, _kfd.category.c_str());
    ev.extdata = _kfd.event_metadata.c_str();

    profiler_hub::writer_types::region_data_t region;
    region.event           = ev;
    region.start_timestamp = _kfd.start_timestamp;
    region.end_timestamp   = _kfd.end_timestamp;
    region.name            = _kfd.name.c_str();

    auto parsed_args = process_arguments_string(_kfd.args_str);
    for(const auto& arg : parsed_args)
    {
        profiler_hub::writer_types::arg_data_t ad;
        ad.position = arg.arg_number;
        ad.type     = arg.arg_type.c_str();
        ad.name     = arg.arg_name.c_str();
        ad.value    = arg.arg_value.c_str();
        region.args.push_back(ad);
    }

    auto env = make_trace_env(n_info.id, process_info.pid, _kfd.thread_id);
    m_writer->insert_region_data(region, env);

    try
    {
        const auto& agent_ref = m_agent_manager->get_agent_by_type_index(
            _kfd.device_id, static_cast<agent_type>(_kfd.device_type));

        profiler_hub::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = ev;
        pmc_data.value = _kfd.value;

        profiler_hub::writer_types::track_info_t track;
        track.name       = _kfd.track_name.c_str();
        track.node_id    = n_info.id;
        track.process_id = process_info.pid;
        if(_kfd.system_tid.has_value()) track.thread_id = _kfd.system_tid.value();

        profiler_hub::writer_types::sample_data_t sample;
        sample.timestamp = _kfd.start_timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        profiler_hub::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = _kfd.pmc_info_name.c_str();
        pmc_uid.agent_id = make_agent_uid(agent_ref);

        m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
    } catch(const std::out_of_range& e)
    {
        LOG_WARNING("KFD PMC event skipped: agent lookup failed for device_id={}, "
                    "device_type={}: {}",
                    _kfd.device_id, _kfd.device_type, e.what());
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

    auto storage = std::make_unique<profiler_hub::storage_t>(m_db_output_path, uuid);
    m_writer     = std::make_unique<profiler_hub::writer_t>(std::move(storage));
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
    m_writer->flush_in_memory_data_to_disk();

    m_output_registry.register_file(m_db_output_path, output_format::rocpd);

    LOG_INFO("Rocpd processor finalized successfully");
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
    node.machine_id    = n_info.machine_id.c_str();
    node.system_name   = n_info.system_name.c_str();
    node.hostname      = n_info.node_name.c_str();
    node.release       = n_info.release.c_str();
    node.version       = n_info.version.c_str();
    node.hardware_name = n_info.machine.c_str();
    node.domain_name   = n_info.domain_name.c_str();
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
        agent_info.name          = rocpd_agent->name.c_str();
        agent_info.model_name    = rocpd_agent->model_name.c_str();
        agent_info.vendor_name   = rocpd_agent->vendor_name.c_str();
        agent_info.product_name  = rocpd_agent->product_name.c_str();
        agent_info.user_name     = rocpd_agent->product_name.c_str();
        agent_info.extdata       = rocpd_agent->agent_info.c_str();
        agent_info.node_id       = n_info.id;
        agent_info.process_id    = process_info.pid;
        m_writer->register_agent_info(agent_info);
    }

    // Register strings
    auto _string_list = m_metadata->get_string_list();
    for(auto& _string : _string_list)
    {
        m_writer->register_string(_string);
    }

    // Register thread info
    auto _thread_info_list = m_metadata->get_thread_info_list();
    for(auto& t_info : _thread_info_list)
    {
        const auto& extended_info = thread_info::get(t_info.thread_id, SystemTID);
        if(extended_info.has_value())
        {
            t_info.start = extended_info->get_start();
            t_info.end   = extended_info->get_stop();
        }

        auto thread_name = fmt::format("Thread {}", t_info.thread_id);

        profiler_hub::writer_types::thread_info_t ti;
        ti.parent_process_id = process_info.ppid;
        ti.thread_id         = t_info.thread_id;
        ti.name              = thread_name;
        ti.start             = t_info.start;
        ti.end               = t_info.end;
        ti.node_id           = n_info.id;
        ti.process_id        = process_info.pid;
        m_writer->register_thread_info(ti);
    }

    // Register tracks
    auto _track_info_list = m_metadata->get_track_info_list();
    for(auto& track : _track_info_list)
    {
        profiler_hub::writer_types::track_info_t ti;
        ti.name       = std::make_optional<std::string_view>(track.track_name.c_str());
        ti.node_id    = n_info.id;
        ti.process_id = process_info.pid;
        ti.thread_id  = track.thread_id;
        m_writer->register_track_info(ti);
    }

    // Register code objects
    auto _code_object_list = m_metadata->get_code_object_list();
    for(const auto& code_object : _code_object_list)
    {
        auto& code_agent = m_agent_manager->get_agent_by_handle(
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
    auto _kernel_symbols_list = m_metadata->get_kernel_symbol_list();
    for(const auto& kernel_symbol : _kernel_symbols_list)
    {
        auto kernel_name = rocprofsys::utility::demangle(kernel_symbol.kernel_name);

        profiler_hub::writer_types::kernel_symbol_info_t ks;
        ks.id                        = kernel_symbol.kernel_id;
        ks.name                      = kernel_symbol.kernel_name;
        ks.display_name              = kernel_name.c_str();
        ks.kernel_object             = kernel_symbol.kernel_object;
        ks.kernarg_segment_size      = kernel_symbol.kernarg_segment_size;
        ks.kernarg_segment_alignment = kernel_symbol.kernarg_segment_alignment;
        ks.group_segment_size        = kernel_symbol.group_segment_size;
        ks.private_segment_size      = kernel_symbol.private_segment_size;
        ks.sgpr_count                = kernel_symbol.sgpr_count;
        ks.arch_vgpr_count           = kernel_symbol.arch_vgpr_count;
        ks.accum_vgpr_count          = kernel_symbol.accum_vgpr_count;
        ks.node_id                   = n_info.id;
        ks.process_id                = process_info.pid;
        ks.code_obj_id               = kernel_symbol.code_object_id;
        m_writer->register_kernel_symbol_info(ks);

        m_writer->register_string(kernel_name.c_str());
    }

    // Register queue info
    auto _queue_list = m_metadata->get_queue_list();
    for(const auto& queue_handle : _queue_list)
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
    auto _stream_list = m_metadata->get_stream_list();
    for(const auto& stream_handle : _stream_list)
    {
        auto stream_name = fmt::format("Stream {}", stream_handle);

        profiler_hub::writer_types::stream_info_t si;
        si.stream_id  = stream_handle;
        si.name       = stream_name;
        si.node_id    = n_info.id;
        si.process_id = process_info.pid;
        m_writer->register_stream_info(si);
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
        constexpr std::array<agent_type, 2> cpu_gpu_types = {
            agent_type::GPU,
            agent_type::CPU,
        };

        const bool is_cpu_gpu_agent =
            std::find(cpu_gpu_types.begin(), cpu_gpu_types.end(), pmc_info.type) !=
            cpu_gpu_types.end();

        auto& pmc_agent =
            is_cpu_gpu_agent ? m_agent_manager->get_agent_by_type_index(
                                   pmc_info.agent_type_index, pmc_info.type)
                             : m_agent_manager->get_agent_by_id(pmc_info.agent_type_index,
                                                                pmc_info.type);
        auto pmc_agent_uid = make_agent_uid(pmc_agent);

        LOG_TRACE("Inserting PMC description: agent_uid: {}, pmc_info: {}",
                  pmc_agent_uid.type_index, pmc_info.name);

        profiler_hub::writer_types::pmc_info_t           pi;
        profiler_hub::writer_types::pmc_info_unique_id_t uid;
        uid.name            = pmc_info.name.c_str();
        uid.agent_id        = pmc_agent_uid;
        pi.unique_id        = uid;
        pi.target_arch      = is_cpu_gpu_agent
                                  ? std::optional<std::string_view>{ pmc_info.target_arch }
                                  : std::nullopt;
        pi.event_code       = pmc_info.event_code;
        pi.instance_id      = pmc_info.instance_id;
        pi.symbol           = pmc_info.symbol.c_str();
        pi.description      = pmc_info.description.c_str();
        pi.long_description = pmc_info.long_description.c_str();
        pi.component        = pmc_info.component.c_str();
        pi.units            = pmc_info.units.c_str();
        pi.value_type       = pmc_info.value_type.c_str();
        pi.block            = pmc_info.block.c_str();
        pi.expression       = pmc_info.expression.c_str();
        pi.is_constant      = pmc_info.is_constant;
        pi.is_derived       = pmc_info.is_derived;
        pi.node_id          = n_info.id;
        pi.process_id       = process_info.pid;
        m_writer->register_pmc_info(pi);
    }
}

}  // namespace trace_cache
}  // namespace rocprofsys
