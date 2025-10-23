// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "trace_cache/rocpd_post_processing.hpp"
#include "agent_manager.hpp"
#include "config.hpp"
#include "debug.hpp"
#include "library/thread_info.hpp"
#include "node_info.hpp"
#include "rocpd/data_processor.hpp"
#include "rocpd/data_storage/database.hpp"
#include "trace_cache/metadata_registry.hpp"
#include "trace_cache/sample_type.hpp"
#include "trace_cache/storage_parser.hpp"
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <timemory/utility/demangle.hpp>
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
}  // namespace

std::shared_ptr<rocpd::data_processor>
rocpd_post_processing::get_data_processor() const
{
    return m_data_processor;
}

postprocessing_callback
rocpd_post_processing::get_kernel_dispatch_callback() const
{
    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto _kds = static_cast<const struct kernel_dispatch_sample&>(parsed);

        auto  data_processor = get_data_processor();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();
        auto  agent_primary_key =
            m_agent_manager.get_agent_by_handle(_kds.agent_id_handle).base_id;

        auto thread_primary_key =
            data_processor->map_thread_id_to_primary_key(_kds.thread_id);

        auto category_id = data_processor->insert_string(
            trait::name<category::rocm_kernel_dispatch>::value);

        auto kernel_symbol = m_metadata.get_kernel_symbol(_kds.kernel_id);

        if(!kernel_symbol.has_value())
        {
            throw std::runtime_error("Kernel symbol is missing for kernel dispatch");
            return;
        }

        auto region_name_primary_key = data_processor->insert_string(
            tim::demangle(kernel_symbol->kernel_name).c_str());

        auto stack_id        = _kds.correlation_id_internal;
        auto parent_stack_id = _kds.correlation_id_ancestor;
        auto correlation_id  = 0;

        auto event_id = data_processor->insert_event(category_id, stack_id,
                                                     parent_stack_id, correlation_id);

        data_processor->insert_kernel_dispatch(
            n_info.id, process.pid, thread_primary_key, agent_primary_key, _kds.kernel_id,
            _kds.dispatch_id, _kds.queue_id_handle, _kds.stream_handle,
            _kds.start_timestamp, _kds.end_timestamp, _kds.private_segment_size,
            _kds.group_segment_size, _kds.workgroup_size_x, _kds.workgroup_size_y,
            _kds.workgroup_size_z, _kds.grid_size_x, _kds.grid_size_y, _kds.grid_size_z,
            region_name_primary_key, event_id);
#endif
    };
}

postprocessing_callback
rocpd_post_processing::get_memory_copy_callback() const
{
    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto _mcs = static_cast<const struct memory_copy_sample&>(parsed);

        auto  data_processor = get_data_processor();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();

        auto _name            = std::string{ m_metadata.get_buffer_name_info().at(
            static_cast<rocprofiler_buffer_tracing_kind_t>(_mcs.kind),
            static_cast<rocprofiler_tracing_operation_t>(_mcs.operation)) };
        auto name_primary_key = data_processor->insert_string(_name.c_str());

        auto category_primary_key =
            data_processor->insert_string(trait::name<category::rocm_memory_copy>::value);

        auto thread_primary_key =
            data_processor->map_thread_id_to_primary_key(_mcs.thread_id);

        auto dst_agent_primary_key =
            m_agent_manager.get_agent_by_handle(_mcs.dst_agent_id_handle).base_id;
        auto src_agent_primary_key =
            m_agent_manager.get_agent_by_handle(_mcs.src_agent_id_handle).base_id;

        auto stack_id        = _mcs.correlation_id_internal;
        auto parent_stack_id = _mcs.correlation_id_ancestor;
        auto correlation_id  = 0;
        auto queue_id        = 0;

        auto event_primary_key = data_processor->insert_event(
            category_primary_key, stack_id, parent_stack_id, correlation_id);

        data_processor->insert_memory_copy(
            n_info.id, process.pid, thread_primary_key, _mcs.start_timestamp,
            _mcs.end_timestamp, name_primary_key, dst_agent_primary_key,
            _mcs.dst_address_value, src_agent_primary_key, _mcs.src_address_value,
            _mcs.bytes, queue_id, _mcs.stream_handle, name_primary_key,
            event_primary_key);
#endif
    };
}

#if(ROCPROFSYS_USE_ROCM > 0 && ROCPROFILER_VERSION >= 600)
postprocessing_callback
rocpd_post_processing::get_memory_allocate_callback() const
{
#    if ROCPROFSYS_USE_ROCM > 0
    auto memtype_to_db =
        [](std::string_view memory_type) -> std::pair<std::string, std::string> {
        constexpr auto MEMORY_PREFIX  = std::string_view{ "MEMORY_ALLOCATION_" };
        constexpr auto SCRATCH_PREFIX = std::string_view{ "SCRATCH_MEMORY_" };
        constexpr auto VMEM_PREFIX    = std::string_view{ "VMEM_" };
        constexpr auto ASYNC_PREFIX   = std::string_view{ "ASYNC_" };

        std::string _type;
        std::string _level;
        if(memory_type.find(MEMORY_PREFIX) == 0)
        {
            _type = memory_type.substr(MEMORY_PREFIX.length());
            if(_type.find(VMEM_PREFIX) == 0)
            {
                _type  = _type.substr(VMEM_PREFIX.length());
                _level = "VIRTUAL";
            }
            else
            {
                _level = "REAL";
            }
        }
        else if(memory_type.find(SCRATCH_PREFIX) == 0)
        {
            _type  = memory_type.substr(SCRATCH_PREFIX.length());
            _level = "SCRATCH";
            if(memory_type.find(ASYNC_PREFIX) == 0)
            {
                _type = memory_type.substr(ASYNC_PREFIX.length());  // RECLAIM
            }
        }

        if(_type == "ALLOCATE")
        {
            _type = "ALLOC";
        }

        return std::make_pair(_type, _level);
    };
#    endif

    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#    if ROCPROFSYS_USE_ROCM > 0
        auto  _mas           = static_cast<const struct memory_allocate_sample&>(parsed);
        auto  data_processor = get_data_processor();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();
        auto  thread_primary_key =
            data_processor->map_thread_id_to_primary_key(_mas.thread_id);
        auto agent_primary_key = std::optional<uint64_t>{};

        const auto invalid_context = ROCPROFILER_CONTEXT_NONE;
        if(_mas.agent_id_handle != invalid_context.handle)
        {
            {
                agent_primary_key =
                    m_agent_manager.get_agent_by_handle(_mas.agent_id_handle).base_id;
            }
            const auto* _name = m_metadata.get_buffer_name_info().at(
                static_cast<rocprofiler_buffer_tracing_kind_t>(_mas.kind),
                static_cast<rocprofiler_tracing_operation_t>(_mas.operation));

            auto [type, level] = memtype_to_db(_name);

            auto stack_id        = _mas.correlation_id_internal;
            auto parent_stack_id = _mas.correlation_id_ancestor;
            auto correlation_id  = 0;
            auto queue_id        = 0;

            auto category_primary_key = data_processor->insert_string(
                trait::name<category::rocm_memory_allocate>::value);

            auto event_primary_key = data_processor->insert_event(
                category_primary_key, stack_id, parent_stack_id, correlation_id);

            data_processor->insert_memory_alloc(
                n_info.id, process.pid, thread_primary_key, agent_primary_key,
                type.c_str(), level.c_str(), _mas.start_timestamp, _mas.end_timestamp,
                _mas.address_value, _mas.allocation_size, queue_id, _mas.stream_handle,
                event_primary_key);
#    endif
        };
    };
}
#endif

postprocessing_callback
rocpd_post_processing::get_region_callback() const
{
    [[maybe_unused]] auto parse_args = []([[maybe_unused]] const std::string& arg_str) {
#if ROCPROFSYS_USE_ROCM > 0
        rocprofiler_sdk::function_args_t args;
        const std::string                delimiter = ";;";

        auto split = [](const std::string& str, const std::string& _delimiter) {
            std::vector<std::string> tokens;
            size_t                   start = 0;
            size_t                   end   = str.find(_delimiter);

            while(end != std::string::npos)
            {
                tokens.push_back(str.substr(start, end - start));
                start = end + _delimiter.length();
                end   = str.find(_delimiter, start);
            }

            return tokens;
        };

        if(arg_str.empty())
        {
            return args;
        }

        auto tokens = split(arg_str, delimiter);

        // Ensure the number of tokens is a multiple of 4
        if(tokens.size() % 4 != 0)
        {
            throw std::invalid_argument("Malformed argument string.");
        }

        for(auto it = tokens.begin(); it != tokens.end(); it += 4)
        {
            rocprofiler_sdk::argument_info arg = { static_cast<uint32_t>(std::stoi(*it)),
                                                   *(it + 1), *(it + 2), *(it + 3) };
            args.push_back(arg);
        }

        return args;
#endif
    };

    return [&]([[maybe_unused]] const storage_parsed_type_base& parsed) {
#if ROCPROFSYS_USE_ROCM > 0
        auto  _rs            = static_cast<const struct region_sample&>(parsed);
        auto  data_processor = get_data_processor();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();
        auto  thread_primary_key =
            data_processor->map_thread_id_to_primary_key(_rs.thread_id);

        auto name_primary_key     = data_processor->insert_string(_rs.name.c_str());
        auto category_primary_key = data_processor->insert_string(_rs.category.c_str());

        size_t stack_id        = _rs.correlation_id_internal;
        size_t parent_stack_id = _rs.correlation_id_ancestor;
        size_t correlation_id  = 0;

        auto event_primary_key =
            data_processor->insert_event(category_primary_key, stack_id, parent_stack_id,
                                         correlation_id, _rs.call_stack.c_str());

        auto args = parse_args(_rs.args_str);
        for(const auto& arg : args)
        {
            data_processor->insert_args(event_primary_key, arg.arg_number,
                                        arg.arg_type.c_str(), arg.arg_name.c_str(),
                                        arg.arg_value.c_str());
        }

        data_processor->insert_region(n_info.id, process.pid, thread_primary_key,
                                      _rs.start_timestamp, _rs.end_timestamp,
                                      name_primary_key, event_primary_key);
#endif
    };
}

postprocessing_callback
rocpd_post_processing::get_backtrace_sample_callback() const
{
    return [&](const storage_parsed_type_base& parsed) {
        auto  _bts           = static_cast<const struct backtrace_region_sample&>(parsed);
        auto  data_processor = get_data_processor();
        auto& n_info         = node_info::get_instance();
        auto  process        = m_metadata.get_process_info();
        auto  thread_primary_key =
            data_processor->map_thread_id_to_primary_key(_bts.thread_id);
        auto name_primary_key     = data_processor->insert_string(_bts.name.c_str());
        auto category_primary_key = data_processor->insert_string(_bts.category.c_str());

        auto event_primary_key = data_processor->insert_event(
            category_primary_key, 0, 0, 0, _bts.call_stack.c_str(),
            _bts.line_info.c_str(), _bts.extdata.c_str());

        data_processor->insert_region(n_info.id, process.pid, thread_primary_key,
                                      _bts.start_timestamp, _bts.end_timestamp,
                                      name_primary_key, event_primary_key);
        data_processor->insert_sample(_bts.track_name.c_str(), _bts.start_timestamp,
                                      event_primary_key);
    };
}

postprocessing_callback
rocpd_post_processing::get_in_time_sample_callback() const
{
    return [&](const storage_parsed_type_base& parsed) {
        auto _its              = static_cast<const struct in_time_sample&>(parsed);
        auto data_processor    = get_data_processor();
        auto track_primary_key = data_processor->insert_string(_its.track_name.c_str());

        auto event_id = data_processor->insert_event(
            track_primary_key, _its.stack_id, _its.parent_stack_id, _its.correlation_id,
            _its.call_stack.c_str(), _its.line_info.c_str(), _its.event_metadata.c_str());
        data_processor->insert_sample(_its.track_name.c_str(), _its.timestamp_ns,
                                      event_id, "{}");
    };
}
postprocessing_callback
rocpd_post_processing::get_pmc_event_with_sample_callback() const
{
    return [&](const storage_parsed_type_base& parsed) {
        auto _pmc              = static_cast<const struct pmc_event_with_sample&>(parsed);
        auto data_processor    = get_data_processor();
        auto track_primary_key = data_processor->insert_string(_pmc.track_name.c_str());

        auto agent_primary_key =
            m_agent_manager
                .get_agent_by_id(_pmc.device_id,
                                 static_cast<agent_type>(_pmc.device_type))
                .base_id;

        auto event_id = data_processor->insert_event(
            track_primary_key, _pmc.stack_id, _pmc.parent_stack_id, _pmc.correlation_id,
            _pmc.call_stack.c_str(), _pmc.line_info.c_str(), _pmc.event_metadata.c_str());
        data_processor->insert_sample(_pmc.track_name.c_str(), _pmc.timestamp_ns,
                                      event_id, "{}");

        data_processor->insert_pmc_event(event_id, agent_primary_key,
                                         _pmc.pmc_info_name.c_str(), _pmc.value);
    };
}

postprocessing_callback
rocpd_post_processing::get_amd_smi_sample_callback() const
{
    struct xcp_metrics_t
    {
        std::vector<uint16_t> vcn_busy;
        std::vector<uint16_t> jpeg_busy;
    };

    auto deserialize_xcp_metrics = [](const std::vector<uint8_t>& serialized_data,
                                      bool& _is_vcn_supported, bool& _is_jpeg_supported,
                                      std::vector<xcp_metrics_t>& result) {
        if(serialized_data.size() < 5)
        {
            throw std::runtime_error("Invalid serialized data: insufficient header size");
        }

        size_t offset = 0;

        // Read header
        _is_vcn_supported   = static_cast<bool>(serialized_data[offset++]);
        _is_jpeg_supported  = static_cast<bool>(serialized_data[offset++]);
        uint8_t chunk_count = serialized_data[offset++];
        uint8_t vcn_count   = serialized_data[offset++];
        uint8_t jpeg_count  = serialized_data[offset++];

        constexpr size_t elem_size  = sizeof(uint16_t) / sizeof(uint8_t);
        const size_t     chunk_size = (vcn_count + jpeg_count) * elem_size;

        // Validate total size
        const size_t expected_size = 5 + (chunk_count * chunk_size);
        if(serialized_data.size() != expected_size)
        {
            throw std::runtime_error("Invalid serialized data: size mismatch");
        }

        auto deserialize_uint16_array = [](const std::vector<uint8_t>& data,
                                           size_t& _offset, int array_size) {
            std::vector<uint16_t> _result;
            _result.reserve(array_size);

            for(int i = 0; i < array_size; ++i)
            {
                if(_offset + 1 >= data.size())
                {
                    throw std::runtime_error(
                        "Invalid serialized data: unexpected end of data");
                }

                uint16_t value = static_cast<uint16_t>(data[_offset]) |
                                 (static_cast<uint16_t>(data[_offset + 1]) << 8);
                _result.push_back(value);
                _offset += 2;
            }

            return _result;
        };

        result.reserve(chunk_count);

        for(size_t count = 0; count < chunk_count; ++count)
        {
            xcp_metrics_t entry;
            entry.vcn_busy = deserialize_uint16_array(serialized_data, offset, vcn_count);
            entry.jpeg_busy =
                deserialize_uint16_array(serialized_data, offset, jpeg_count);

            result.emplace_back(std::move(entry));
        }
    };

    return [&](const storage_parsed_type_base& parsed) {
        auto _amd_smi = static_cast<const struct amd_smi_sample&>(parsed);

        auto data_processor = get_data_processor();

        const auto* _name            = trait::name<category::amd_smi>::value;
        auto        name_primary_key = data_processor->insert_string(_name);
        auto        event_id = data_processor->insert_event(name_primary_key, 0, 0, 0);

        auto base_id =
            m_agent_manager.get_agent_by_type_index(_amd_smi.device_id, agent_type::GPU)
                .base_id;

        auto insert_event_and_sample = [&](bool enabled, const char* pmc_name,
                                           const char* track_name, double value) {
            if(!enabled) return;
            data_processor->insert_pmc_event(event_id, base_id, pmc_name, value);
            data_processor->insert_sample(track_name, _amd_smi.timestamp, event_id);
        };

        using pos = trace_cache::amd_smi_sample::settings_positions;
        std::bitset<8> settings_bits(_amd_smi.settings);
        bool           is_busy_enabled = settings_bits.test(static_cast<int>(pos::busy));
        bool           is_temp_enabled = settings_bits.test(static_cast<int>(pos::temp));
        bool is_power_enabled          = settings_bits.test(static_cast<int>(pos::power));
        bool is_mem_usage_enabled = settings_bits.test(static_cast<int>(pos::mem_usage));

        bool is_vcn_enabled  = settings_bits.test(static_cast<int>(pos::vcn_activity));
        bool is_jpeg_enabled = settings_bits.test(static_cast<int>(pos::jpeg_activity));

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
            info::annotate_with_device_id<category::amd_smi_temp>(_amd_smi.device_id)
                .c_str(),
            _amd_smi.temperature);

        insert_event_and_sample(
            is_power_enabled, trait::name<category::amd_smi_power>::value,
            info::annotate_with_device_id<category::amd_smi_power>(_amd_smi.device_id)
                .c_str(),
            _amd_smi.power);
        insert_event_and_sample(
            is_mem_usage_enabled, trait::name<category::amd_smi_memory_usage>::value,
            info::annotate_with_device_id<category::amd_smi_memory_usage>(
                _amd_smi.device_id)
                .c_str(),
            _amd_smi.mem_usage);

        if(!is_vcn_enabled && !is_jpeg_enabled)
        {
            return;
        }

        std::vector<xcp_metrics_t> xcp_metrics;
        bool                       is_vcn_activity_supported;
        bool                       is_jpeg_activity_supported;
        deserialize_xcp_metrics(_amd_smi.xcp_activity, is_vcn_activity_supported,
                                is_jpeg_activity_supported, xcp_metrics);

        auto insert_xcp_metrics = [&](auto category, bool _is_enabled,
                                      const std::vector<uint16_t>& data,
                                      std::optional<size_t>        _idx = std::nullopt) {
            if(!_is_enabled)
            {
                return;
            }

            using Category = std::decay_t<decltype(category)>;

            for(size_t clk = 0; clk < data.size(); ++clk)
            {
                const auto value = data[clk];
                if(value == std::numeric_limits<uint16_t>::max())
                {
                    continue;
                }

                auto pmc_name   = info::annotate_category<Category>(_idx, clk);
                auto track_name = info::annotate_with_device_id<Category>(
                    _amd_smi.device_id, _idx, clk);

                insert_event_and_sample(_is_enabled, pmc_name.c_str(), track_name.c_str(),
                                        value);
            }
        };

        for(size_t idx = 0; idx < xcp_metrics.size(); ++idx)
        {
            auto dimension =
                xcp_metrics.size() == 1 ? std::nullopt : std::make_optional<size_t>(idx);

            insert_xcp_metrics(category::amd_smi_vcn_activity{}, is_vcn_enabled,
                               xcp_metrics[idx].vcn_busy, dimension);

            insert_xcp_metrics(category::amd_smi_jpeg_activity{}, is_jpeg_enabled,
                               xcp_metrics[idx].jpeg_busy, dimension);
        }
    };
}

postprocessing_callback
rocpd_post_processing::get_cpu_freq_sample_callback() const
{
    struct core_freq_sample
    {
        size_t id;
        float  value;
    };

    auto deserialize_freqs = [](std::vector<uint8_t>& buffer) {
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

    return [&](const storage_parsed_type_base& parsed) {
        auto _cpu_freq_sample = static_cast<const struct cpu_freq_sample&>(parsed);

        auto        data_processor   = get_data_processor();
        const auto* _name            = trait::name<category::cpu_freq>::value;
        auto        name_primary_key = data_processor->insert_string(_name);
        auto        event_id = data_processor->insert_event(name_primary_key, 0, 0, 0);

        auto device_id = 0;

        auto base_id =
            m_agent_manager.get_agent_by_type_index(device_id, agent_type::CPU).base_id;

        auto insert_event_and_sample = [&](const char* name, double value) {
            data_processor->insert_pmc_event(event_id, base_id, name, value);
            data_processor->insert_sample(name, _cpu_freq_sample.timestamp, event_id);
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
    };
}

rocpd_post_processing::rocpd_post_processing(metadata_registry& md,
                                             agent_manager& agent_mngr, int pid, int ppid)
: m_metadata(md)
, m_agent_manager(agent_mngr)
, m_data_processor(std::make_shared<rocpd::data_processor>(
      std::make_shared<rocpd::data_storage::database>(pid, ppid)))
{}

void
rocpd_post_processing::register_parser_callback([[maybe_unused]] storage_parser& parser)
{
#if ROCPROFSYS_USE_ROCM > 0
    if(!get_use_rocpd())
    {
        return;
    }
    parser.register_type_callback(entry_type::region, get_region_callback());
    parser.register_type_callback(entry_type::kernel_dispatch,
                                  get_kernel_dispatch_callback());
    parser.register_type_callback(entry_type::memory_copy, get_memory_copy_callback());
#    if(ROCPROFILER_VERSION >= 600)
    parser.register_type_callback(entry_type::memory_alloc,
                                  get_memory_allocate_callback());
#    endif
    parser.register_type_callback(entry_type::in_time_sample,
                                  get_in_time_sample_callback());
    parser.register_type_callback(entry_type::pmc_event_with_sample,
                                  get_pmc_event_with_sample_callback());
    parser.register_type_callback(entry_type::amd_smi_sample,
                                  get_amd_smi_sample_callback());
    parser.register_type_callback(entry_type::cpu_freq_sample,
                                  get_cpu_freq_sample_callback());
    parser.register_type_callback(entry_type::backtrace_region_sample,
                                  get_backtrace_sample_callback());
    ROCPROFSYS_DEBUG("Buffer parser callbacks are registered..\n");

    parser.register_on_finished_callback(
        std::make_unique<std::function<void()>>([this]() {
            if(m_data_processor != nullptr)
            {
                m_data_processor->flush();
            }
        }));
#endif
}

void
rocpd_post_processing::post_process_metadata()
{
#if ROCPROFSYS_USE_ROCM > 0
    if(!get_use_rocpd())
    {
        return;
    }
    ROCPROFSYS_DEBUG("Post processing metadata..\n");
    auto data_processor = get_data_processor();
    auto n_info         = node_info::get_instance();

    data_processor->insert_node_info(n_info.id, n_info.hash, n_info.machine_id.c_str(),
                                     n_info.system_name.c_str(), n_info.node_name.c_str(),
                                     n_info.release.c_str(), n_info.version.c_str(),
                                     n_info.machine.c_str(), n_info.domain_name.c_str());

    auto process_info = m_metadata.get_process_info();
    data_processor->insert_process_info(n_info.id, process_info.ppid, process_info.pid, 0,
                                        0, process_info.start, process_info.end,
                                        process_info.command.c_str(), "{}");

    const auto& agents  = m_agent_manager.get_agents();
    int         counter = 0;
    for(const auto& rocpd_agent : agents)
    {
        auto _base_id = data_processor->insert_agent(
            n_info.id, process_info.pid,
            ((rocpd_agent->type == agent_type::GPU) ? "GPU" : "CPU"), counter++,
            rocpd_agent->logical_node_id, rocpd_agent->logical_node_type_id,
            rocpd_agent->device_id, rocpd_agent->name.c_str(),
            rocpd_agent->model_name.c_str(), rocpd_agent->vendor_name.c_str(),
            rocpd_agent->product_name.c_str(), "");
        rocpd_agent->base_id = _base_id;
    }
    auto _string_list = m_metadata.get_string_list();
    for(auto& _string : _string_list)
    {
        data_processor->insert_string(std::string(_string).c_str());
    }

    auto _thread_info_list = m_metadata.get_thread_info_list();
    for(auto& t_info : _thread_info_list)
    {
        rocpd_insert_thread_id(t_info, n_info, process_info);
    }

    auto _track_info_list = m_metadata.get_track_info_list();
    for(auto& track : _track_info_list)
    {
        auto thread_id =
            track.thread_id.has_value()
                ? std::make_optional<size_t>(data_processor->map_thread_id_to_primary_key(
                      track.thread_id.value()))
                : std::nullopt;
        data_processor->insert_track(track.track_name.c_str(), n_info.id,
                                     process_info.pid, thread_id);
    }

    auto _code_object_list = m_metadata.get_code_object_list();
    for(const auto& code_object : _code_object_list)
    {
        auto dev_id =
            m_agent_manager.get_agent_by_handle(get_handle_from_code_object(code_object))
                .base_id;

        const char* strg_type = "UNKNOWN";
        switch(code_object.storage_type)
        {
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE: strg_type = "FILE"; break;
            case ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY: strg_type = "MEMORY"; break;
            default: break;
        }
        data_processor->insert_code_object(code_object.code_object_id, n_info.id,
                                           process_info.pid, dev_id, code_object.uri,
                                           code_object.load_base, code_object.load_size,
                                           code_object.load_delta, strg_type);
    }

    auto _kernel_symbols_list = m_metadata.get_kernel_symbol_list();
    for(const auto& kernel_symbol : _kernel_symbols_list)
    {
        auto kernel_name = tim::demangle(kernel_symbol.kernel_name);
        data_processor->insert_kernel_symbol(
            kernel_symbol.kernel_id, n_info.id, process_info.pid,
            kernel_symbol.code_object_id, kernel_symbol.kernel_name, kernel_name.c_str(),
            kernel_symbol.kernel_object, kernel_symbol.kernarg_segment_size,
            kernel_symbol.kernarg_segment_alignment, kernel_symbol.group_segment_size,
            kernel_symbol.private_segment_size, kernel_symbol.sgpr_count,
            kernel_symbol.arch_vgpr_count, kernel_symbol.accum_vgpr_count);

        data_processor->insert_string(kernel_name.c_str());
    }

    auto _queue_list = m_metadata.get_queue_list();
    for(const auto& queue_handle : _queue_list)
    {
        std::stringstream ss;
        ss << "Queue " << queue_handle;
        data_processor->insert_queue_info(queue_handle, n_info.id, process_info.pid,
                                          ss.str().c_str());
    }

    auto _stream_list = m_metadata.get_stream_list();
    for(const auto& stream_handle : _stream_list)
    {
        std::stringstream ss;
        ss << "Stream " << stream_handle;
        data_processor->insert_stream_info(stream_handle, n_info.id, process_info.pid,
                                           ss.str().c_str());
    }

    auto buffer_info_list = m_metadata.get_buffer_name_info();
    for(const auto& buffer_info : buffer_info_list)
    {
        for(const auto& item : buffer_info.items())
        {
            data_processor->insert_string(*item.second);
        }
    }

    auto callback_info_list = m_metadata.get_callback_tracing_info();
    for(const auto& cb_info : callback_info_list)
    {
        for(const auto& item : cb_info.items())
        {
            data_processor->insert_string(*item.second);
        }
    }

    auto pmc_info_list = m_metadata.get_pmc_info_list();
    for(const auto& pmc_info : pmc_info_list)
    {
        const auto agent_primary_key =
            m_agent_manager
                .get_agent_by_type_index(pmc_info.agent_type_index, pmc_info.type)
                .base_id;

        data_processor->insert_pmc_description(
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
rocpd_post_processing::rocpd_insert_thread_id(info::thread&        t_info,
                                              const node_info&     n_info,
                                              const info::process& process_info) const
{
    const auto& extended_info = thread_info::get(t_info.thread_id, SystemTID);
    if(extended_info.has_value())
    {
        t_info.start = extended_info->get_start();
        t_info.end   = extended_info->get_stop();
    }

    std::stringstream ss;
    ss << "Thread " << t_info.thread_id;
    get_data_processor()->insert_thread_info(n_info.id, process_info.ppid,
                                             process_info.pid, t_info.thread_id,
                                             ss.str().c_str(), t_info.start, t_info.end);
}

}  // namespace trace_cache
}  // namespace rocprofsys
