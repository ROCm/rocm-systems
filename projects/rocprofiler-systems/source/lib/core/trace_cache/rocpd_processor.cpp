// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "core/trace_cache/rocpd_processor.hpp"
#include "core/agent_manager.hpp"
#include "core/common_types.hpp"
#include "core/config.hpp"
#include "core/demangler.hpp"
#include "core/gpu_metrics.hpp"
#include "core/node_info.hpp"
#include "core/rocpd/data_processor.hpp"
#include "core/rocpd/data_storage/database.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/thread_info.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#if ROCPROFSYS_USE_ROCM > 0
#    include "library/rocprofiler-sdk/fwd.hpp"
#    include <rocprofiler-sdk/context.h>
#    include <rocprofiler-sdk/version.h>
#endif

namespace rocprofsys
{
namespace trace_cache
{
namespace
{

#if ROCPROFSYS_USE_ROCM > 0
auto
get_handle_from_code_object(
    const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
#    if(ROCPROFILER_VERSION >= 600)
    return code_object.agent_id.handle;
#    else
    return code_object.rocp_agent.handle;
#    endif
}
#endif

#if ROCPROFSYS_USE_ROCM > 0
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
#endif
}  // namespace

void
rocpd_processor_t::handle([[maybe_unused]] const kernel_dispatch_sample& _kds)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();
    auto  agent_primary_key =
        m_agent_manager->get_agent_by_handle(_kds.agent_id_handle).base_id;

    auto thread_primary_key =
        m_data_processor->map_thread_id_to_primary_key(_kds.thread_id);

    auto category_id = m_data_processor->insert_string(
        trait::name<category::rocm_kernel_dispatch>::value);

    auto kernel_symbol = m_metadata->get_kernel_symbol(_kds.kernel_id);

    if(!kernel_symbol.has_value())
    {
        throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
    }

    auto region_name_primary_key = m_data_processor->insert_string(
        rocprofsys::utility::demangle(kernel_symbol->kernel_name).c_str());

    auto stack_id        = _kds.correlation_id_internal;
    auto parent_stack_id = _kds.correlation_id_ancestor;
    auto correlation_id  = 0;

    auto event_id = m_data_processor->insert_event(category_id, stack_id, parent_stack_id,
                                                   correlation_id);

    m_data_processor->insert_kernel_dispatch(
        n_info.id, process.pid, thread_primary_key, agent_primary_key, _kds.kernel_id,
        _kds.dispatch_id, _kds.queue_id_handle, _kds.stream_handle, _kds.start_timestamp,
        _kds.end_timestamp, _kds.private_segment_size, _kds.group_segment_size,
        _kds.workgroup_size_x, _kds.workgroup_size_y, _kds.workgroup_size_z,
        _kds.grid_size_x, _kds.grid_size_y, _kds.grid_size_z, region_name_primary_key,
        event_id);
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const scratch_memory_sample& _sms)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    const auto* _name = m_metadata->get_buffer_name_info().at(
        static_cast<rocprofiler_buffer_tracing_kind_t>(_sms.kind),
        static_cast<rocprofiler_tracing_operation_t>(_sms.operation));

    auto agent_primary_key =
        m_agent_manager->get_agent_by_handle(_sms.agent_id_handle).base_id;

    auto thread_primary_key =
        m_data_processor->map_thread_id_to_primary_key(_sms.thread_id);

    auto category_primary_key = m_data_processor->insert_string(
        trait::name<category::rocm_scratch_memory>::value);

    auto stack_id        = _sms.correlation_id_internal;
    auto parent_stack_id = _sms.correlation_id_ancestor;
    auto correlation_id  = 0;
    auto address_value   = 0;

    auto event_primary_key = m_data_processor->insert_event(
        category_primary_key, stack_id, parent_stack_id, correlation_id);

    auto [memory_operation, memory_type] = parse_memory_operation_name(_name);

    auto extdata_json_str = fmt::format("{{\"flags\": {}}}", _sms.flags);

    m_data_processor->insert_memory_alloc(
        n_info.id, process.pid, thread_primary_key, agent_primary_key,
        memory_operation.c_str(), memory_type.c_str(), _sms.start_timestamp,
        _sms.end_timestamp, address_value, _sms.allocation_size, _sms.queue_id_handle,
        _sms.stream_handle, event_primary_key, extdata_json_str.c_str());
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const memory_copy_sample& _mcs)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();

    auto _name            = std::string{ m_metadata->get_buffer_name_info().at(
        static_cast<rocprofiler_buffer_tracing_kind_t>(_mcs.kind),
        static_cast<rocprofiler_tracing_operation_t>(_mcs.operation)) };
    auto name_primary_key = m_data_processor->insert_string(_name.c_str());

    auto category_primary_key =
        m_data_processor->insert_string(trait::name<category::rocm_memory_copy>::value);

    auto thread_primary_key =
        m_data_processor->map_thread_id_to_primary_key(_mcs.thread_id);

    auto dst_agent_primary_key =
        m_agent_manager->get_agent_by_handle(_mcs.dst_agent_id_handle).base_id;
    auto src_agent_primary_key =
        m_agent_manager->get_agent_by_handle(_mcs.src_agent_id_handle).base_id;

    auto stack_id        = _mcs.correlation_id_internal;
    auto parent_stack_id = _mcs.correlation_id_ancestor;
    auto correlation_id  = 0;
    auto queue_id        = 0;

    auto event_primary_key = m_data_processor->insert_event(
        category_primary_key, stack_id, parent_stack_id, correlation_id);

    m_data_processor->insert_memory_copy(
        n_info.id, process.pid, thread_primary_key, _mcs.start_timestamp,
        _mcs.end_timestamp, name_primary_key, dst_agent_primary_key,
        _mcs.dst_address_value, src_agent_primary_key, _mcs.src_address_value, _mcs.bytes,
        queue_id, _mcs.stream_handle, name_primary_key, event_primary_key);
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const memory_allocate_sample& _mas)
{
#if ROCPROFSYS_USE_ROCM > 0 && (ROCPROFILER_VERSION >= 600)
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();
    auto  thread_primary_key =
        m_data_processor->map_thread_id_to_primary_key(_mas.thread_id);
    auto agent_primary_key = std::optional<uint64_t>{};

    const auto invalid_context = ROCPROFILER_CONTEXT_NONE;
    if(_mas.agent_id_handle != invalid_context.handle)
    {
        {
            agent_primary_key =
                m_agent_manager->get_agent_by_handle(_mas.agent_id_handle).base_id;
        }
        const auto* _name = m_metadata->get_buffer_name_info().at(
            static_cast<rocprofiler_buffer_tracing_kind_t>(_mas.kind),
            static_cast<rocprofiler_tracing_operation_t>(_mas.operation));

        auto [memory_operation, memory_type] = parse_memory_operation_name(_name);

        auto stack_id        = _mas.correlation_id_internal;
        auto parent_stack_id = _mas.correlation_id_ancestor;
        auto correlation_id  = 0;
        auto queue_id        = 0;

        auto category_primary_key = m_data_processor->insert_string(
            trait::name<category::rocm_memory_allocate>::value);

        auto event_primary_key = m_data_processor->insert_event(
            category_primary_key, stack_id, parent_stack_id, correlation_id);

        m_data_processor->insert_memory_alloc(
            n_info.id, process.pid, thread_primary_key, agent_primary_key,
            memory_operation.c_str(), memory_type.c_str(), _mas.start_timestamp,
            _mas.end_timestamp, _mas.address_value, _mas.allocation_size, queue_id,
            _mas.stream_handle, event_primary_key);
    }
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const region_sample& _rs)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();
    auto  thread_primary_key =
        m_data_processor->map_thread_id_to_primary_key(_rs.thread_id);

    auto name_primary_key     = m_data_processor->insert_string(_rs.name.c_str());
    auto category_primary_key = m_data_processor->insert_string(_rs.category.c_str());

    size_t stack_id        = _rs.correlation_id_internal;
    size_t parent_stack_id = _rs.correlation_id_ancestor;
    size_t correlation_id  = 0;

    auto event_primary_key =
        m_data_processor->insert_event(category_primary_key, stack_id, parent_stack_id,
                                       correlation_id, _rs.call_stack.c_str());

    auto args = process_arguments_string(_rs.args_str);
    for(const auto& arg : args)
    {
        m_data_processor->insert_args(event_primary_key, arg.arg_number,
                                      arg.arg_type.c_str(), arg.arg_name.c_str(),
                                      arg.arg_value.c_str());
    }

    m_data_processor->insert_region(n_info.id, process.pid, thread_primary_key,
                                    _rs.start_timestamp, _rs.end_timestamp,
                                    name_primary_key, event_primary_key);
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const backtrace_region_sample& _bts)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto& n_info  = node_info::get_instance();
    auto  process = m_metadata->get_process_info();
    auto  thread_primary_key =
        m_data_processor->map_thread_id_to_primary_key(_bts.thread_id);
    auto name_primary_key     = m_data_processor->insert_string(_bts.name.c_str());
    auto category_primary_key = m_data_processor->insert_string(_bts.category.c_str());

    auto event_primary_key = m_data_processor->insert_event(
        category_primary_key, 0, 0, 0, _bts.call_stack.c_str(), _bts.line_info.c_str(),
        _bts.extdata.c_str());

    m_data_processor->insert_region(n_info.id, process.pid, thread_primary_key,
                                    _bts.start_timestamp, _bts.end_timestamp,
                                    name_primary_key, event_primary_key);
    m_data_processor->insert_sample(_bts.track_name.c_str(), _bts.start_timestamp,
                                    event_primary_key);
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const in_time_sample& _its)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto track_primary_key = m_data_processor->insert_string(_its.track_name.c_str());

    auto event_id = m_data_processor->insert_event(
        track_primary_key, _its.stack_id, _its.parent_stack_id, _its.correlation_id,
        _its.call_stack.c_str(), _its.line_info.c_str(), _its.event_metadata.c_str());
    m_data_processor->insert_sample(_its.track_name.c_str(), _its.timestamp_ns, event_id,
                                    "{}");
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const pmc_event_with_sample& _pmc)
{
#if ROCPROFSYS_USE_ROCM > 0
    auto track_primary_key = m_data_processor->insert_string(_pmc.track_name.c_str());

    auto agent_primary_key =
        m_agent_manager
            ->get_agent_by_type_index(_pmc.device_id,
                                      static_cast<agent_type>(_pmc.device_type))
            .base_id;

    auto event_id = m_data_processor->insert_event(
        track_primary_key, _pmc.stack_id, _pmc.parent_stack_id, _pmc.correlation_id,
        _pmc.call_stack.c_str(), _pmc.line_info.c_str(), _pmc.event_metadata.c_str());
    m_data_processor->insert_sample(_pmc.track_name.c_str(), _pmc.timestamp_ns, event_id,
                                    "{}");

    m_data_processor->insert_pmc_event(event_id, agent_primary_key,
                                       _pmc.pmc_info_name.c_str(), _pmc.value,
                                       _pmc.event_metadata.c_str());
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const amd_smi_sample& _amd_smi)
{
#if ROCPROFSYS_USE_ROCM > 0

    const auto* _name            = trait::name<category::amd_smi>::value;
    auto        name_primary_key = m_data_processor->insert_string(_name);
    auto        event_id = m_data_processor->insert_event(name_primary_key, 0, 0, 0);

    auto base_id =
        m_agent_manager->get_agent_by_type_index(_amd_smi.device_id, agent_type::GPU)
            .base_id;

    auto insert_event_and_sample = [&](bool enabled, const char* pmc_name,
                                       const char* track_name, double value) {
        if(!enabled) return;
        m_data_processor->insert_pmc_event(event_id, base_id, pmc_name, value);
        m_data_processor->insert_sample(track_name, _amd_smi.timestamp, event_id);
    };

    using pos = trace_cache::amd_smi_sample::settings_positions;
    std::bitset<8> settings_bits(_amd_smi.settings);
    bool           is_busy_enabled  = settings_bits.test(static_cast<int>(pos::busy));
    bool           is_temp_enabled  = settings_bits.test(static_cast<int>(pos::temp));
    bool           is_power_enabled = settings_bits.test(static_cast<int>(pos::power));
    bool is_mem_usage_enabled = settings_bits.test(static_cast<int>(pos::mem_usage));

    bool is_vcn_enabled  = settings_bits.test(static_cast<int>(pos::vcn_activity));
    bool is_jpeg_enabled = settings_bits.test(static_cast<int>(pos::jpeg_activity));
    bool is_xgmi_enabled = settings_bits.test(static_cast<int>(pos::xgmi));
    bool is_pcie_enabled = settings_bits.test(static_cast<int>(pos::pcie));

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

    if(!is_vcn_enabled && !is_jpeg_enabled && !is_xgmi_enabled && !is_pcie_enabled)
        return;

    gpu::gpu_metrics_t              gpu_metrics;
    gpu::gpu_metrics_capabilities_t capabilities;
    gpu::deserialize_gpu_metrics(_amd_smi.gpu_activity, gpu_metrics, is_vcn_enabled,
                                 is_jpeg_enabled, is_xgmi_enabled, is_pcie_enabled,
                                 capabilities);

    // Insert VCN and JPEG activity metrics
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

    // Insert XGMI read/write data metrics
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

    // Insert VCN activity metrics
    if(capabilities.flags.vcn_is_device_level_only)
    {
        // Device-level: use vcn_activity vector
        insert_decode_vector_metrics(category::amd_smi_vcn_activity{}, is_vcn_enabled,
                                     gpu_metrics.vcn_activity, std::nullopt);
    }
    else
    {
        // Per-XCP: iterate through actual XCPs in vcn_busy
        for(size_t xcp = 0; xcp < gpu_metrics.vcn_busy.size(); ++xcp)
        {
            insert_decode_vector_metrics(category::amd_smi_vcn_activity{}, is_vcn_enabled,
                                         gpu_metrics.vcn_busy[xcp], xcp);
        }
    }

    // Insert JPEG activity metrics
    if(capabilities.flags.jpeg_is_device_level_only)
    {
        // Device-level: use jpeg_activity vector
        insert_decode_vector_metrics(category::amd_smi_jpeg_activity{}, is_jpeg_enabled,
                                     gpu_metrics.jpeg_activity, std::nullopt);
    }
    else
    {
        // Per-XCP: iterate through actual XCPs in jpeg_busy
        for(size_t xcp = 0; xcp < gpu_metrics.jpeg_busy.size(); ++xcp)
        {
            insert_decode_vector_metrics(category::amd_smi_jpeg_activity{},
                                         is_jpeg_enabled, gpu_metrics.jpeg_busy[xcp],
                                         xcp);
        }
    }

    // Insert XGMI metrics (scalar values)
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
#endif
}

void
rocpd_processor_t::handle([[maybe_unused]] const cpu_freq_sample& _cpu_freq_sample)
{
#if ROCPROFSYS_USE_ROCM > 0
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

    const auto* _name            = trait::name<category::cpu_freq>::value;
    auto        name_primary_key = m_data_processor->insert_string(_name);
    auto        event_id = m_data_processor->insert_event(name_primary_key, 0, 0, 0);

    auto device_id = 0;

    auto base_id =
        m_agent_manager->get_agent_by_type_index(device_id, agent_type::CPU).base_id;

    auto insert_event_and_sample = [&](const char* name, double value) {
        m_data_processor->insert_pmc_event(event_id, base_id, name, value);
        m_data_processor->insert_sample(name, _cpu_freq_sample.timestamp, event_id);
    };

    insert_event_and_sample(trait::name<category::process_page>::value,
                            _cpu_freq_sample.page_rss);
    insert_event_and_sample(trait::name<category::process_virt>::value,
                            _cpu_freq_sample.virt_mem_usage);
    insert_event_and_sample(trait::name<category::process_peak>::value,
                            _cpu_freq_sample.peak_rss);
    insert_event_and_sample(trait::name<category::process_context_switch>::value,
                            _cpu_freq_sample.context_switch_count);
    insert_event_and_sample(trait::name<category::process_page_fault>::value,
                            _cpu_freq_sample.page_faults);
    insert_event_and_sample(trait::name<category::process_user_mode_time>::value,
                            _cpu_freq_sample.user_mode_time);
    insert_event_and_sample(trait::name<category::process_kernel_mode_time>::value,
                            _cpu_freq_sample.kernel_mode_time);

    auto get_track_name = [](const auto& cpu_id) {
        return std::string(trait::name<category::cpu_freq>::value) + " [" +
               std::to_string(cpu_id) + "]";
    };

    auto core_freq_samples = deserialize_freqs(_cpu_freq_sample.freqs);
    for(const auto& core : core_freq_samples)
    {
        insert_event_and_sample(get_track_name(core.id).c_str(), core.value);
    }
#endif
}

rocpd_processor_t::rocpd_processor_t(const std::shared_ptr<metadata_registry>& md,
                                     const std::shared_ptr<agent_manager>&     agent_mngr,
                                     int pid, int ppid)
: processor_t<rocpd_processor_t>()
, m_metadata(md)
, m_agent_manager(agent_mngr)
, m_data_processor(std::make_shared<rocpd::data_processor>(
      std::make_shared<rocpd::data_storage::database>(pid, ppid)))
{}

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
    m_data_processor->flush();
    LOG_INFO("Rocpd processor finalized successfully");
}

void
rocpd_processor_t::post_process_metadata()
{
#if ROCPROFSYS_USE_ROCM > 0
    if(!get_use_rocpd())
    {
        LOG_TRACE("Rocpd not enabled, skipping metadata post-processing");
        return;
    }
    LOG_DEBUG("Post-processing metadata for rocpd");
    auto n_info = node_info::get_instance();

    m_data_processor->insert_node_info(
        n_info.id, n_info.hash, n_info.machine_id.c_str(), n_info.system_name.c_str(),
        n_info.node_name.c_str(), n_info.release.c_str(), n_info.version.c_str(),
        n_info.machine.c_str(), n_info.domain_name.c_str());

    auto process_info = m_metadata->get_process_info();
    m_data_processor->insert_process_info(n_info.id, process_info.ppid, process_info.pid,
                                          0, 0, process_info.start, process_info.end,
                                          process_info.command.c_str(), "{}");

    const auto& agents  = m_agent_manager->get_agents();
    int         counter = 0;
    for(const auto& rocpd_agent : agents)
    {
        auto _base_id = m_data_processor->insert_agent(
            n_info.id, process_info.pid,
            ((rocpd_agent->type == agent_type::GPU) ? "GPU" : "CPU"), counter++,
            rocpd_agent->logical_node_id, rocpd_agent->logical_node_type_id,
            rocpd_agent->device_id, rocpd_agent->name.c_str(),
            rocpd_agent->model_name.c_str(), rocpd_agent->vendor_name.c_str(),
            rocpd_agent->product_name.c_str(), rocpd_agent->product_name.c_str(),
            rocpd_agent->agent_info.c_str());
        rocpd_agent->base_id = _base_id;
    }
    auto _string_list = m_metadata->get_string_list();
    for(auto& _string : _string_list)
    {
        m_data_processor->insert_string(std::string(_string).c_str());
    }

    auto _thread_info_list = m_metadata->get_thread_info_list();
    for(auto& t_info : _thread_info_list)
    {
        insert_thread_id(t_info, n_info, process_info);
    }

    auto _track_info_list = m_metadata->get_track_info_list();
    for(auto& track : _track_info_list)
    {
        auto thread_id = track.thread_id.has_value()
                             ? std::make_optional<size_t>(
                                   m_data_processor->map_thread_id_to_primary_key(
                                       track.thread_id.value()))
                             : std::nullopt;
        m_data_processor->insert_track(track.track_name.c_str(), n_info.id,
                                       process_info.pid, thread_id);
    }

    auto _code_object_list = m_metadata->get_code_object_list();
    for(const auto& code_object : _code_object_list)
    {
        auto dev_id =
            m_agent_manager->get_agent_by_handle(get_handle_from_code_object(code_object))
                .base_id;

        const char* strg_type = "UNKNOWN";
        switch(code_object.storage_type)
        {
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE: strg_type = "FILE"; break;
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY: strg_type = "MEMORY"; break;
            default: break;
        }
        m_data_processor->insert_code_object(code_object.code_object_id, n_info.id,
                                             process_info.pid, dev_id, code_object.uri,
                                             code_object.load_base, code_object.load_size,
                                             code_object.load_delta, strg_type);
    }

    auto _kernel_symbols_list = m_metadata->get_kernel_symbol_list();
    for(const auto& kernel_symbol : _kernel_symbols_list)
    {
        auto kernel_name = rocprofsys::utility::demangle(kernel_symbol.kernel_name);
        m_data_processor->insert_kernel_symbol(
            kernel_symbol.kernel_id, n_info.id, process_info.pid,
            kernel_symbol.code_object_id, kernel_symbol.kernel_name, kernel_name.c_str(),
            kernel_symbol.kernel_object, kernel_symbol.kernarg_segment_size,
            kernel_symbol.kernarg_segment_alignment, kernel_symbol.group_segment_size,
            kernel_symbol.private_segment_size, kernel_symbol.sgpr_count,
            kernel_symbol.arch_vgpr_count, kernel_symbol.accum_vgpr_count);

        m_data_processor->insert_string(kernel_name.c_str());
    }

    auto _queue_list = m_metadata->get_queue_list();
    for(const auto& queue_handle : _queue_list)
    {
        std::stringstream ss;
        ss << "Queue " << queue_handle;
        m_data_processor->insert_queue_info(queue_handle, n_info.id, process_info.pid,
                                            ss.str().c_str());
    }

    auto _stream_list = m_metadata->get_stream_list();
    for(const auto& stream_handle : _stream_list)
    {
        std::stringstream ss;
        ss << "Stream " << stream_handle;
        m_data_processor->insert_stream_info(stream_handle, n_info.id, process_info.pid,
                                             ss.str().c_str());
    }

    auto buffer_info_list = m_metadata->get_buffer_name_info();
    for(const auto& buffer_info : buffer_info_list)
    {
        for(const auto& item : buffer_info.items())
        {
            m_data_processor->insert_string(*item.second);
        }
    }

    auto callback_info_list = m_metadata->get_callback_tracing_info();
    for(const auto& cb_info : callback_info_list)
    {
        for(const auto& item : cb_info.items())
        {
            m_data_processor->insert_string(*item.second);
        }
    }

    auto pmc_info_list = m_metadata->get_pmc_info_list();
    for(const auto& pmc_info : pmc_info_list)
    {
        const auto agent_primary_key =
            m_agent_manager
                ->get_agent_by_type_index(pmc_info.agent_type_index, pmc_info.type)
                .base_id;

        m_data_processor->insert_pmc_description(
            n_info.id, process_info.pid, agent_primary_key, pmc_info.target_arch.c_str(),
            pmc_info.event_code, pmc_info.instance_id, pmc_info.name.c_str(),
            pmc_info.symbol.c_str(), pmc_info.description.c_str(),
            pmc_info.long_description.c_str(), pmc_info.component.c_str(),
            pmc_info.units.c_str(), pmc_info.value_type.c_str(), pmc_info.block.c_str(),
            pmc_info.expression.c_str(), pmc_info.is_constant, pmc_info.is_derived);
    }
#endif
}

inline void
rocpd_processor_t::insert_thread_id(info::thread& t_info, const node_info& n_info,
                                    const info::process& process_info)
{
    const auto& extended_info = thread_info::get(t_info.thread_id, SystemTID);
    if(extended_info.has_value())
    {
        t_info.start = extended_info->get_start();
        t_info.end   = extended_info->get_stop();
    }

    std::stringstream ss;
    ss << "Thread " << t_info.thread_id;
    m_data_processor->insert_thread_info(n_info.id, process_info.ppid, process_info.pid,
                                         t_info.thread_id, ss.str().c_str(), t_info.start,
                                         t_info.end);
}

}  // namespace trace_cache
}  // namespace rocprofsys
