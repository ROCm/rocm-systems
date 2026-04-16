// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "core/trace_cache/rocpd_processor.hpp"
#include "common/md5sum.hpp"
#include "core/agent_manager.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/gpu_metrics.hpp"
#include "core/node_info.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <rocpdsna/storage.hpp>
#include <rocpdsna/writer.hpp>
#include <rocpdsna/writer_types.hpp>

#include <bitset>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
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

auto
get_handle_from_code_object(
    const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
#if (ROCPROFILER_VERSION >= 600)
    return code_object.agent_id.handle;
#else
    return code_object.rocp_agent.handle;
#endif
}

rocpdsna::writer_types::agent_unique_id_t
make_agent_uid(const agent& a)
{
    const auto type_to_string = [](agent_type type) -> std::optional<std::string_view> {
        switch(type)
        {
            case agent_type::GPU: return "GPU";
            case agent_type::CPU: return "CPU";
            default: return std::nullopt;
        }
    };

    rocpdsna::writer_types::agent_unique_id_t uid;
    uid.agent_type = type_to_string(a.type);
    uid.type_index = a.device_type_index;
    return uid;
}

rocpdsna::writer_types::trace_environment_t
make_trace_env(size_t node_id, size_t process_id, size_t thread_id)
{
    rocpdsna::writer_types::trace_environment_t env;
    env.node_id    = node_id;
    env.process_id = process_id;
    env.thread_id  = thread_id;
    return env;
}

rocpdsna::writer_types::trace_environment_t
make_trace_env_with_agent(size_t node_id, size_t process_id, size_t thread_id,
                          const agent& a)
{
    auto env     = make_trace_env(node_id, process_id, thread_id);
    env.agent_id = make_agent_uid(a);
    return env;
}

rocpdsna::writer_types::trace_environment_t
make_trace_env_with_agent_queue_stream(size_t node_id, size_t process_id,
                                       size_t thread_id, const agent& a, size_t queue_id,
                                       size_t stream_id)
{
    auto env      = make_trace_env_with_agent(node_id, process_id, thread_id, a);
    env.queue_id  = queue_id;
    env.stream_id = stream_id;
    return env;
}

rocpdsna::writer_types::event_data_t
make_event(size_t stack_id, size_t parent_stack_id, size_t correlation_id,
           const char* category)
{
    rocpdsna::writer_types::event_data_t ev;
    ev.stack_id        = stack_id;
    ev.parent_stack_id = parent_stack_id;
    ev.correlation_id  = correlation_id;
    ev.event_category  = category;
    return ev;
}

using memory_operation = std::string;
using memory_type      = std::string;
std::pair<memory_operation, memory_type>
parse_memory_operation_name(std::string_view memory_operation_name)
{
    static const std::unordered_map<std::string_view,
                                    std::pair<memory_operation, memory_type>>
        parsing_map{
            { "MEMORY_ALLOCATION_NONE", { "NONE", "REAL" } },
            { "MEMORY_ALLOCATION_ALLOCATE", { "ALLOC", "REAL" } },
            { "MEMORY_ALLOCATION_VMEM_ALLOCATE", { "ALLOC", "VIRTUAL" } },
            { "MEMORY_ALLOCATION_FREE", { "FREE", "REAL" } },
            { "MEMORY_ALLOCATION_VMEM_FREE", { "FREE", "VIRTUAL" } },
            { "SCRATCH_MEMORY_NONE", { "NONE", "SCRATCH" } },
            { "SCRATCH_MEMORY_ALLOC", { "ALLOC", "SCRATCH" } },
            { "SCRATCH_MEMORY_FREE", { "FREE", "SCRATCH" } },
            { "SCRATCH_MEMORY_ASYNC_RECLAIM", { "ASYNC_RECLAIM", "SCRATCH" } },
        };

    auto item = parsing_map.find(memory_operation_name);
    if(item == parsing_map.end())
    {
        LOG_WARNING("Unknown memory operation name: {}", memory_operation_name);
        return { "UNKNOWN", "UNKNOWN" };
    }

    return item->second;
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

    rocpdsna::writer_types::kernel_dispatch_data_t kd;
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

    rocpdsna::writer_types::memory_alloc_data_t ma;
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

    rocpdsna::writer_types::memory_copy_data_t mc;
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

    m_writer->insert_memory_copy_data(mc, env);
}

void
rocpd_processor_t::handle([[maybe_unused]] const memory_allocate_sample& _mas)
{
#if (ROCPROFILER_VERSION >= 600)
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

        rocpdsna::writer_types::memory_alloc_data_t ma;
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
    // call_stack and line_info are serialized JSON in the old code; in rocpdsna
    // they are structured types. For now pass the raw JSON via extdata.
    ev.extdata = _rs.call_stack.c_str();

    rocpdsna::writer_types::region_data_t region;
    region.event           = ev;
    region.start_timestamp = _rs.start_timestamp;
    region.end_timestamp   = _rs.end_timestamp;
    region.name            = _rs.name.c_str();

    auto parsed_args = process_arguments_string(_rs.args_str);
    for(const auto& arg : parsed_args)
    {
        rocpdsna::writer_types::arg_data_t ad;
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

    auto ev    = make_event(0, 0, 0, _bts.category.c_str());
    ev.extdata = _bts.extdata.c_str();

    rocpdsna::writer_types::region_data_t region;
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

    rocpdsna::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event = ev;
    pmc_data.value = 0.0;

    rocpdsna::writer_types::track_info_t track;
    track.name = _its.track_name.c_str();

    rocpdsna::writer_types::sample_data_t sample;
    sample.timestamp = _its.timestamp_ns;
    sample.track     = track;
    pmc_data.sample  = sample;

    rocpdsna::writer_types::pmc_info_unique_id_t pmc_uid;
    pmc_uid.name = _its.track_name.c_str();

    m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
}

void
rocpd_processor_t::handle(const pmc_event_with_sample& _pmc)
{
    const auto& process_info = m_metadata->get_process_info();
    const auto& agent_ref    = m_agent_manager->get_agent_by_id(
        _pmc.device_id, static_cast<agent_type>(_pmc.device_type));

    auto ev    = make_event(_pmc.stack_id, _pmc.parent_stack_id, _pmc.correlation_id,
                            _pmc.track_name.c_str());
    ev.extdata = _pmc.event_metadata.c_str();

    rocpdsna::writer_types::pmc_event_data_t pmc_data;
    pmc_data.event = ev;
    pmc_data.value = _pmc.value;

    rocpdsna::writer_types::track_info_t track;
    track.name       = _pmc.track_name.c_str();
    track.node_id    = node_info::get_instance().id;
    track.process_id = process_info.pid;
    track.thread_id  = _pmc.system_tid;

    rocpdsna::writer_types::sample_data_t sample;
    sample.timestamp = _pmc.timestamp_ns;
    sample.track     = track;
    pmc_data.sample  = sample;

    rocpdsna::writer_types::pmc_info_unique_id_t pmc_uid;
    pmc_uid.name     = _pmc.pmc_info_name.c_str();
    pmc_uid.agent_id = make_agent_uid(agent_ref);

    m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
}

void
rocpd_processor_t::handle(const amd_smi_sample& _amd_smi)
{
    const auto* _name        = trait::name<category::amd_smi>::value;
    const auto& process_info = m_metadata->get_process_info();
    const auto& agent_ref =
        m_agent_manager->get_agent_by_type_index(_amd_smi.device_id, agent_type::GPU);

    const auto agent_uid = make_agent_uid(agent_ref);

    auto ev = make_event(0, 0, 0, _name);

    auto insert_event_and_sample = [&](bool enabled, const char* pmc_name,
                                       const char* track_name, double value) {
        if(!enabled) return;

        rocpdsna::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = ev;
        pmc_data.value = value;

        rocpdsna::writer_types::track_info_t track;
        track.name       = track_name;
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        rocpdsna::writer_types::sample_data_t sample;
        sample.timestamp = _amd_smi.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        rocpdsna::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = pmc_name;
        pmc_uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
    };

    using pos = trace_cache::amd_smi_sample::settings_positions;
    std::bitset<16> settings_bits(_amd_smi.settings);
    bool            is_busy_enabled  = settings_bits.test(static_cast<int>(pos::busy));
    bool            is_temp_enabled  = settings_bits.test(static_cast<int>(pos::temp));
    bool            is_power_enabled = settings_bits.test(static_cast<int>(pos::power));
    bool is_mem_usage_enabled = settings_bits.test(static_cast<int>(pos::mem_usage));

    bool is_vcn_enabled  = settings_bits.test(static_cast<int>(pos::vcn_activity));
    bool is_jpeg_enabled = settings_bits.test(static_cast<int>(pos::jpeg_activity));
    bool is_xgmi_enabled = settings_bits.test(static_cast<int>(pos::xgmi));
    bool is_pcie_enabled = settings_bits.test(static_cast<int>(pos::pcie));
    bool is_sdma_enabled = settings_bits.test(static_cast<int>(pos::sdma_usage));

    insert_event_and_sample(
        is_busy_enabled, trait::name<category::amd_smi_gfx_busy>::value,
        info::annotate_with_device_id<category::amd_smi_gfx_busy>(_amd_smi.device_id)
            .c_str(),
        _amd_smi.gfx_activity);
    insert_event_and_sample(
        is_busy_enabled, trait::name<category::amd_smi_umc_busy>::value,
        info::annotate_with_device_id<category::amd_smi_umc_busy>(_amd_smi.device_id)
            .c_str(),
        _amd_smi.umc_activity);
    insert_event_and_sample(
        is_busy_enabled, trait::name<category::amd_smi_mm_busy>::value,
        info::annotate_with_device_id<category::amd_smi_mm_busy>(_amd_smi.device_id)
            .c_str(),
        _amd_smi.mm_activity);
    insert_event_and_sample(
        is_temp_enabled, trait::name<category::amd_smi_temp>::value,
        info::annotate_with_device_id<category::amd_smi_temp>(_amd_smi.device_id).c_str(),
        _amd_smi.temperature);

    insert_event_and_sample(
        is_power_enabled, trait::name<category::amd_smi_power>::value,
        info::annotate_with_device_id<category::amd_smi_power>(_amd_smi.device_id)
            .c_str(),
        _amd_smi.power);

    auto mem_usage_mb = _amd_smi.mem_usage / static_cast<double>(units::megabyte);
    insert_event_and_sample(
        is_mem_usage_enabled, trait::name<category::amd_smi_memory_usage>::value,
        info::annotate_with_device_id<category::amd_smi_memory_usage>(_amd_smi.device_id)
            .c_str(),
        mem_usage_mb);

    // Insert SDMA usage metric (doesn't require gpu_metrics deserialization)
    insert_event_and_sample(
        is_sdma_enabled, trait::name<category::amd_smi_sdma_usage>::value,
        info::annotate_with_device_id<category::amd_smi_sdma_usage>(_amd_smi.device_id)
            .c_str(),
        static_cast<double>(_amd_smi.sdma_usage));

    if(!is_vcn_enabled && !is_jpeg_enabled && !is_xgmi_enabled && !is_pcie_enabled)
        return;

    gpu::gpu_metrics_t              gpu_metrics;
    gpu::gpu_metrics_capabilities_t capabilities;
    gpu::deserialize_gpu_metrics(_amd_smi.gpu_activity, gpu_metrics, is_vcn_enabled,
                                 is_jpeg_enabled, is_xgmi_enabled, is_pcie_enabled,
                                 capabilities);

    auto insert_decode_vector_metrics = [&](auto category, bool _is_enabled,
                                            const std::vector<uint16_t>& data,
                                            std::optional<size_t> _idx = std::nullopt) {
        if(!_is_enabled) return;

        using Category = std::decay_t<decltype(category)>;

        for(size_t i = 0; i < data.size(); ++i)
        {
            const auto value = data[i];
            if(value == std::numeric_limits<uint16_t>::max()) continue;

            auto pmc_name = info::annotate_category<Category>(_idx, i);
            auto track_name =
                info::annotate_with_device_id<Category>(_amd_smi.device_id, _idx, i);

            insert_event_and_sample(_is_enabled, pmc_name.c_str(), track_name.c_str(),
                                    static_cast<double>(value));
        }
    };

    auto insert_xgmi_vector_metrics = [&](auto category, bool _is_enabled,
                                          const std::vector<uint64_t>& data,
                                          std::optional<size_t> _idx = std::nullopt) {
        if(!_is_enabled) return;

        using Category = std::decay_t<decltype(category)>;

        for(size_t i = 0; i < data.size(); ++i)
        {
            const auto value = data[i];
            if(value == std::numeric_limits<uint64_t>::max()) continue;

            auto pmc_name = info::annotate_category<Category>(_idx, i);
            auto track_name =
                info::annotate_with_device_id<Category>(_amd_smi.device_id, _idx, i);

            insert_event_and_sample(_is_enabled, pmc_name.c_str(), track_name.c_str(),
                                    static_cast<double>(value));
        }
    };

    if(capabilities.flags.vcn_is_device_level_only)
    {
        insert_decode_vector_metrics(category::amd_smi_vcn_activity{}, is_vcn_enabled,
                                     gpu_metrics.vcn_activity, std::nullopt);
    }
    else
    {
        for(size_t xcp = 0; xcp < gpu_metrics.vcn_busy.size(); ++xcp)
        {
            insert_decode_vector_metrics(category::amd_smi_vcn_activity{}, is_vcn_enabled,
                                         gpu_metrics.vcn_busy[xcp], xcp);
        }
    }

    if(capabilities.flags.jpeg_is_device_level_only)
    {
        insert_decode_vector_metrics(category::amd_smi_jpeg_activity{}, is_jpeg_enabled,
                                     gpu_metrics.jpeg_activity, std::nullopt);
    }
    else
    {
        for(size_t xcp = 0; xcp < gpu_metrics.jpeg_busy.size(); ++xcp)
        {
            insert_decode_vector_metrics(category::amd_smi_jpeg_activity{},
                                         is_jpeg_enabled, gpu_metrics.jpeg_busy[xcp],
                                         xcp);
        }
    }

    insert_event_and_sample(
        is_xgmi_enabled, trait::name<category::amd_smi_xgmi_link_width>::value,
        info::annotate_with_device_id<category::amd_smi_xgmi_link_width>(
            _amd_smi.device_id)
            .c_str(),
        gpu_metrics.xgmi_link_width);

    insert_event_and_sample(
        is_xgmi_enabled, trait::name<category::amd_smi_xgmi_link_speed>::value,
        info::annotate_with_device_id<category::amd_smi_xgmi_link_speed>(
            _amd_smi.device_id)
            .c_str(),
        gpu_metrics.xgmi_link_speed);

    insert_xgmi_vector_metrics(category::amd_smi_xgmi_read_data{}, is_xgmi_enabled,
                               gpu_metrics.xgmi_read_data_acc, std::nullopt);

    insert_xgmi_vector_metrics(category::amd_smi_xgmi_write_data{}, is_xgmi_enabled,
                               gpu_metrics.xgmi_write_data_acc, std::nullopt);

    insert_event_and_sample(
        is_pcie_enabled, trait::name<category::amd_smi_pcie_link_width>::value,
        info::annotate_with_device_id<category::amd_smi_pcie_link_width>(
            _amd_smi.device_id)
            .c_str(),
        gpu_metrics.pcie_link_width);

    insert_event_and_sample(
        is_pcie_enabled, trait::name<category::amd_smi_pcie_link_speed>::value,
        info::annotate_with_device_id<category::amd_smi_pcie_link_speed>(
            _amd_smi.device_id)
            .c_str(),
        gpu_metrics.pcie_link_speed);

    insert_event_and_sample(
        is_pcie_enabled, trait::name<category::amd_smi_pcie_bandwidth_acc>::value,
        info::annotate_with_device_id<category::amd_smi_pcie_bandwidth_acc>(
            _amd_smi.device_id)
            .c_str(),
        static_cast<double>(gpu_metrics.pcie_bandwidth_acc));

    insert_event_and_sample(
        is_pcie_enabled, trait::name<category::amd_smi_pcie_bandwidth_inst>::value,
        info::annotate_with_device_id<category::amd_smi_pcie_bandwidth_inst>(
            _amd_smi.device_id)
            .c_str(),
        static_cast<double>(gpu_metrics.pcie_bandwidth_inst));
}

void
rocpd_processor_t::handle(const cpu_freq_sample& _cpu_freq_sample)
{
    struct core_freq_sample
    {
        size_t id;
        float  value;
    };

    auto deserialize_freqs = [](const std::vector<uint8_t>& buffer) {
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

    const auto* _name     = trait::name<category::cpu_freq>::value;
    const auto& agent_ref = m_agent_manager->get_agent_by_type_index(0, agent_type::CPU);
    const auto  agent_uid = make_agent_uid(agent_ref);

    auto ev = make_event(0, 0, 0, _name);

    auto insert_event_and_sample = [&](const char* name, double value) {
        rocpdsna::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = ev;
        pmc_data.value = value;

        rocpdsna::writer_types::track_info_t track;
        track.name = name;

        rocpdsna::writer_types::sample_data_t sample;
        sample.timestamp = _cpu_freq_sample.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        rocpdsna::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = name;
        pmc_uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
    };

    insert_event_and_sample(trait::name<category::process_page>::value,
                            static_cast<double>(_cpu_freq_sample.page_rss) /
                                units::megabyte);
    insert_event_and_sample(trait::name<category::process_virt>::value,
                            static_cast<double>(_cpu_freq_sample.virt_mem_usage) /
                                units::megabyte);
    insert_event_and_sample(trait::name<category::process_peak>::value,
                            static_cast<double>(_cpu_freq_sample.peak_rss) /
                                units::megabyte);
    insert_event_and_sample(trait::name<category::process_context_switch>::value,
                            _cpu_freq_sample.context_switch_count);
    insert_event_and_sample(trait::name<category::process_page_fault>::value,
                            _cpu_freq_sample.page_faults);
    insert_event_and_sample(trait::name<category::process_user_mode_time>::value,
                            static_cast<double>(_cpu_freq_sample.user_mode_time) /
                                units::sec);
    insert_event_and_sample(trait::name<category::process_kernel_mode_time>::value,
                            static_cast<double>(_cpu_freq_sample.kernel_mode_time) /
                                units::sec);

    auto get_track_name = [](const auto& cpu_id) {
        return std::string(trait::name<category::cpu_freq>::value) + " [" +
               std::to_string(cpu_id) + "]";
    };

    auto core_freq_samples = deserialize_freqs(_cpu_freq_sample.freqs);
    for(const auto& core : core_freq_samples)
    {
        insert_event_and_sample(get_track_name(core.id).c_str(), core.value);
    }
}

void
rocpd_processor_t::handle(const ainic_sample& _ainic)
{
    const auto* _name        = trait::name<category::amd_smi_nic>::value;
    const auto& process_info = m_metadata->get_process_info();
    const auto& nic_agent =
        m_agent_manager->get_agent_by_id(_ainic.nic_index, agent_type::NIC);

    const auto agent_uid = make_agent_uid(nic_agent);

    auto ev = make_event(0, 0, 0, _name);

    auto insert_event_and_sample = [&](const char* pmc_name, const char* track_name,
                                       double value) {
        rocpdsna::writer_types::pmc_event_data_t pmc_data;
        pmc_data.event = ev;
        pmc_data.value = value;

        rocpdsna::writer_types::track_info_t track;
        track.name       = track_name;
        track.node_id    = node_info::get_instance().id;
        track.process_id = process_info.pid;

        rocpdsna::writer_types::sample_data_t sample;
        sample.timestamp = _ainic.timestamp;
        sample.track     = track;
        pmc_data.sample  = sample;

        rocpdsna::writer_types::pmc_info_unique_id_t pmc_uid;
        pmc_uid.name     = pmc_name;
        pmc_uid.agent_id = agent_uid;

        m_writer->insert_pmc_event_data(pmc_data, pmc_uid);
    };

    const char* nic_name = nic_agent.name.c_str();

    insert_event_and_sample(trait::name<category::amd_smi_nic_rx_cnp_pkts>::value,
                            info::annotate_with_nic<category::amd_smi_nic_rx_cnp_pkts>(
                                nic_name, _ainic.nic_index)
                                .c_str(),
                            _ainic.rx_rdma_cnp_pkts);

    insert_event_and_sample(trait::name<category::amd_smi_nic_tx_cnp_pkts>::value,
                            info::annotate_with_nic<category::amd_smi_nic_tx_cnp_pkts>(
                                nic_name, _ainic.nic_index)
                                .c_str(),
                            _ainic.tx_rdma_cnp_pkts);

    insert_event_and_sample(trait::name<category::amd_smi_nic_rx_ucast_bytes>::value,
                            info::annotate_with_nic<category::amd_smi_nic_rx_ucast_bytes>(
                                nic_name, _ainic.nic_index)
                                .c_str(),
                            _ainic.rx_ucast_bytes);

    insert_event_and_sample(trait::name<category::amd_smi_nic_tx_ucast_bytes>::value,
                            info::annotate_with_nic<category::amd_smi_nic_tx_ucast_bytes>(
                                nic_name, _ainic.nic_index)
                                .c_str(),
                            _ainic.tx_ucast_bytes);

    insert_event_and_sample(trait::name<category::amd_smi_nic_rx_ucast_pkts>::value,
                            info::annotate_with_nic<category::amd_smi_nic_rx_ucast_pkts>(
                                nic_name, _ainic.nic_index)
                                .c_str(),
                            _ainic.rx_ucast_pkts);

    insert_event_and_sample(trait::name<category::amd_smi_nic_tx_ucast_pkts>::value,
                            info::annotate_with_nic<category::amd_smi_nic_tx_ucast_pkts>(
                                nic_name, _ainic.nic_index)
                                .c_str(),
                            _ainic.tx_ucast_pkts);
}

rocpd_processor_t::rocpd_processor_t(const std::shared_ptr<metadata_registry>& md,
                                     const std::shared_ptr<agent_manager>&     agent_mngr,
                                     int pid, int ppid)
: processor_t<rocpd_processor_t>()
, m_metadata(md)
, m_agent_manager(agent_mngr)
{
    auto _tag    = std::to_string(pid);
    auto db_path = rocprofsys::get_database_absolute_path("rocpd", _tag);
    auto n_info  = node_info::get_instance();
    auto uuid    = common::md5sum{ n_info.id, pid, ppid }.hexdigest();

    auto storage = std::make_unique<rocpdsna::storage_t>(db_path, uuid);
    m_writer     = std::make_unique<rocpdsna::writer_t>(std::move(storage));
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
    rocpdsna::writer_types::node_info_t node;
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
    auto                                   process_info = m_metadata->get_process_info();
    rocpdsna::writer_types::process_info_t proc;
    proc.ppid        = process_info.ppid;
    proc.pid         = process_info.pid;
    proc.init        = 0;
    proc.fini        = 0;
    proc.start       = process_info.start;
    proc.end         = process_info.end;
    proc.command     = process_info.command.c_str();
    proc.environment = "{}";
    proc.node_id     = n_info.id;
    m_writer->register_process_info(proc);

    // Register agents
    const auto& agents  = m_agent_manager->get_agents();
    int         counter = 0;
    for(const auto& rocpd_agent : agents)
    {
        rocpdsna::writer_types::agent_info_t agent_info;
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
        m_writer->register_string(std::string(_string).c_str());
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

        std::stringstream ss;
        ss << "Thread " << t_info.thread_id;

        rocpdsna::writer_types::thread_info_t ti;
        ti.parent_process_id = process_info.ppid;
        ti.thread_id         = t_info.thread_id;
        ti.name              = ss.str().c_str();
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
        rocpdsna::writer_types::track_info_t ti;
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

        rocpdsna::writer_types::code_object_info_t co;
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

        rocpdsna::writer_types::kernel_symbol_info_t ks;
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
        std::stringstream ss;
        ss << "Queue " << queue_handle;

        rocpdsna::writer_types::queue_info_t qi;
        qi.queue_id   = queue_handle;
        qi.name       = ss.str().c_str();
        qi.node_id    = n_info.id;
        qi.process_id = process_info.pid;
        m_writer->register_queue_info(qi);
    }

    // Register stream info
    auto _stream_list = m_metadata->get_stream_list();
    for(const auto& stream_handle : _stream_list)
    {
        std::stringstream ss;
        ss << "Stream " << stream_handle;

        rocpdsna::writer_types::stream_info_t si;
        si.stream_id  = stream_handle;
        si.name       = ss.str().c_str();
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
        auto& pmc_agent = m_agent_manager->get_agent_by_type_index(
            pmc_info.agent_type_index, pmc_info.type);
        auto pmc_agent_uid = make_agent_uid(pmc_agent);

        rocpdsna::writer_types::pmc_info_t           pi;
        rocpdsna::writer_types::pmc_info_unique_id_t uid;
        uid.name            = pmc_info.name.c_str();
        uid.agent_id        = pmc_agent_uid;
        pi.unique_id        = uid;
        pi.target_arch      = pmc_info.target_arch.c_str();
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
