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

#include "storage_parser.hpp"
#include "debug.hpp"
#include "trace_cache/sample_type.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace rocprofsys
{
namespace trace_cache
{

void
storage_parser::register_type_callback(
    const entry_type&                                           type,
    const std::function<void(const storage_parsed_type_base&)>& callback)
{
    m_callbacks[type].push_back(callback);
}

void
storage_parser::consume_storage()
{
    std::ifstream ifs(filename, std::ios::binary);
    if(!ifs)
    {
        std::stringstream ss;
        ss << "Error opening file for reading: " << filename << "\n";
        throw std::runtime_error(ss.str());
    }

    bool _parsing_needed = !m_callbacks.empty();

    struct __attribute__((packed)) sample_header
    {
        entry_type type;
        size_t     sample_size;
    };

    sample_header header;

    while(!ifs.eof() && _parsing_needed)
    {
        ifs.read(reinterpret_cast<char*>(&header), sizeof(header));

        if(header.sample_size == 0 || ifs.eof())
        {
            continue;
        }

        std::vector<uint8_t> sample;
        sample.reserve(header.sample_size);
        ifs.read(reinterpret_cast<char*>(sample.data()), header.sample_size);

        if(ifs.bad())
        {
            continue;
        }

        switch(header.type)
        {
            case entry_type::kernel_dispatch:
            {
                kernel_dispatch_sample _kernel_dispatch_sample;
                parse_data(sample.data(), _kernel_dispatch_sample.start_timestamp,
                           _kernel_dispatch_sample.end_timestamp,
                           _kernel_dispatch_sample.thread_id,
                           _kernel_dispatch_sample.agent_id_handle,
                           _kernel_dispatch_sample.kernel_id,
                           _kernel_dispatch_sample.dispatch_id,
                           _kernel_dispatch_sample.queue_id_handle,
                           _kernel_dispatch_sample.correlation_id_internal,
                           _kernel_dispatch_sample.correlation_id_ancestor,
                           _kernel_dispatch_sample.private_segment_size,
                           _kernel_dispatch_sample.group_segment_size,
                           _kernel_dispatch_sample.workgroup_size_x,
                           _kernel_dispatch_sample.workgroup_size_y,
                           _kernel_dispatch_sample.workgroup_size_z,
                           _kernel_dispatch_sample.grid_size_x,
                           _kernel_dispatch_sample.grid_size_y,
                           _kernel_dispatch_sample.grid_size_z,
                           _kernel_dispatch_sample.stream_handle);

                invoke_callbacks(header.type, _kernel_dispatch_sample);
                break;
            }
            case entry_type::memory_copy:
            {
                memory_copy_sample _memory_copy_sample;
                parse_data(
                    sample.data(), _memory_copy_sample.start_timestamp,
                    _memory_copy_sample.end_timestamp, _memory_copy_sample.thread_id,
                    _memory_copy_sample.dst_agent_id_handle,
                    _memory_copy_sample.src_agent_id_handle, _memory_copy_sample.kind,
                    _memory_copy_sample.operation, _memory_copy_sample.bytes,
                    _memory_copy_sample.correlation_id_internal,
                    _memory_copy_sample.correlation_id_ancestor,
                    _memory_copy_sample.dst_address_value,
                    _memory_copy_sample.src_address_value,
                    _memory_copy_sample.stream_handle);
                invoke_callbacks(header.type, _memory_copy_sample);
                break;
            }
#if(ROCPROFILER_VERSION >= 600)
            case entry_type::memory_alloc:
            {
                memory_allocate_sample _memory_allocate_sample;
                parse_data(sample.data(), _memory_allocate_sample.start_timestamp,
                           _memory_allocate_sample.end_timestamp,
                           _memory_allocate_sample.thread_id,
                           _memory_allocate_sample.agent_id_handle,
                           _memory_allocate_sample.kind,
                           _memory_allocate_sample.operation,
                           _memory_allocate_sample.allocation_size,
                           _memory_allocate_sample.correlation_id_internal,
                           _memory_allocate_sample.correlation_id_ancestor,
                           _memory_allocate_sample.address_value,
                           _memory_allocate_sample.stream_handle);

                invoke_callbacks(header.type, _memory_allocate_sample);
                break;
            }
#endif
            case entry_type::region:
            {
                region_sample _region_sample;
                parse_data(sample.data(), _region_sample.thread_id, _region_sample.kind,
                           _region_sample.operation,
                           _region_sample.correlation_id_internal,
                           _region_sample.correlation_id_ancestor,
                           _region_sample.start_timestamp, _region_sample.end_timestamp,
                           _region_sample.call_stack, _region_sample.args_str,
                           _region_sample.category);

                invoke_callbacks(header.type, _region_sample);
                break;
            }
            case entry_type::in_time_sample:
            {
                in_time_sample _in_time_sample;
                parse_data(sample.data(), _in_time_sample.track_name,
                           _in_time_sample.timestamp_ns, _in_time_sample.event_metadata,
                           _in_time_sample.stack_id, _in_time_sample.parent_stack_id,
                           _in_time_sample.correlation_id, _in_time_sample.call_stack,
                           _in_time_sample.line_info);
                invoke_callbacks(header.type, _in_time_sample);
                break;
            }
            case entry_type::pmc_event_with_sample:
            {
                pmc_event_with_sample _pmc_event_with_sample;
                parse_data(
                    sample.data(), _pmc_event_with_sample.track_name,
                    _pmc_event_with_sample.timestamp_ns,
                    _pmc_event_with_sample.event_metadata,
                    _pmc_event_with_sample.stack_id,
                    _pmc_event_with_sample.parent_stack_id,
                    _pmc_event_with_sample.correlation_id,
                    _pmc_event_with_sample.call_stack, _pmc_event_with_sample.line_info,
                    _pmc_event_with_sample.agent_handle,
                    _pmc_event_with_sample.pmc_info_name, _pmc_event_with_sample.value);
                invoke_callbacks(header.type, _pmc_event_with_sample);
                break;
            }
            default: break;
        }
    }

    ifs.close();
    std::remove(filename.c_str());
}

void
storage_parser::invoke_callbacks(entry_type type, const storage_parsed_type_base& parsed)
{
    auto _callback_list = m_callbacks.find(type);
    if(_callback_list == m_callbacks.end())
    {
        ROCPROFSYS_VERBOSE(1, "Callback not found for cache postprocessing");
        return;
    }

    for(auto& cb : _callback_list->second)
    {
        cb(parsed);
    }
}
}  // namespace trace_cache
}  // namespace rocprofsys
