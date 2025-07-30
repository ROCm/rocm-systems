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

#pragma once

#include "common/synchronized.hpp"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <set>
#include <stdint.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <unordered_set>

namespace rocprofsys
{
namespace sample_cache
{
namespace info
{
struct process
{
    pid_t       pid;  // < Unique
    pid_t       ppid;
    std::string command;
};
struct pmc
{
    size_t      agent_handle;
    std::string target_arch;
    size_t      event_code;
    size_t      instance_id;
    std::string name;  // < Unique
    std::string symbol;
    std::string description;
    std::string long_description;
    std::string component;
    std::string units;
    std::string value_type;
    std::string block;
    std::string expression;
    uint32_t    is_constant;
    uint32_t    is_derived;
    std::string extdata;

    friend bool operator<(const pmc& lhs, const pmc& rhs)
    {
        return lhs.name.compare(rhs.name) < 0;
    }
};

struct thread
{
    int32_t     parent_process_id;
    int32_t     process_id;
    uint64_t    thread_id;  // < Unique
    uint32_t    start;
    uint32_t    end;
    std::string extdata;
    friend bool operator<(const thread& lhs, const thread& rhs)
    {
        return lhs.thread_id < rhs.thread_id;
    }
};

struct track
{
    std::string           track_name;  // < Unique
    std::optional<size_t> thread_id;
    std::string           extdata;

    friend bool operator<(const track& lhs, const track& rhs)
    {
        return lhs.track_name.compare(rhs.track_name) < 0;
    }
};

struct code_object_less
{
    bool operator()(const rocprofiler_callback_tracing_code_object_load_data_t& lhs,
                    const rocprofiler_callback_tracing_code_object_load_data_t& rhs) const
    {
        return lhs.code_object_id < rhs.code_object_id;
    }
};

struct kernel_symbol_less
{
    bool operator()(
        const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t& lhs,
        const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t& rhs)
        const
    {
        return lhs.kernel_object < rhs.kernel_object;
    }
};

}  // namespace info

class cache_manager;
struct metadata
{
    void set_process(const info::process& process);
    void add_pmc_info(const info::pmc& pmc_info);
    void add_thread_info(const info::thread& thread_info);
    void add_track(const info::track& track_info);
    void add_code_object(
        const rocprofiler_callback_tracing_code_object_load_data_t& code_object);
    void add_kernel_symbol(
        const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t&
            kernel_symbol);
    void add_queue(const uint64_t& queue_handle);
    void add_stream(const uint64_t& stream_handle);
    void add_string(const std::string_view& string_value);

    info::process               get_process_info() const;
    std::optional<info::pmc>    get_pmc_info(const std::string_view& unique_name) const;
    std::optional<info::thread> get_thread_info(const uint32_t& thread_id) const;
    std::optional<info::track>  get_track_info(const std::string_view& track_name) const;
    std::optional<rocprofiler_callback_tracing_code_object_load_data_t> get_code_object(
        uint64_t code_object_id) const;
    std::optional<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>
    get_kernel_symbol(uint64_t kernel_id) const;

    std::vector<info::pmc>    get_pmc_info_list() const;
    std::vector<info::thread> get_thread_info_list() const;
    std::vector<info::track>  get_track_info_list() const;
    std::vector<rocprofiler_callback_tracing_code_object_load_data_t>
    get_code_object_list() const;
    std::vector<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>
                                  get_kernel_symbol_list() const;
    std::vector<uint64_t>         get_queue_list() const;
    std::vector<uint64_t>         get_stream_list() const;
    std::vector<std::string_view> get_string_list() const;

    rocprofiler::sdk::buffer_name_info_t<const char*>   get_buffer_name_info() const;
    rocprofiler::sdk::callback_name_info_t<const char*> get_callback_tracing_info() const;

private:
    friend class cache_manager;
    metadata() = default;
    common::synchronized<info::process>          m_process;
    common::synchronized<std::set<info::pmc>>    m_pmc_infos;
    common::synchronized<std::set<info::thread>> m_threads;
    common::synchronized<std::set<info::track>>  m_tracks;
    common::synchronized<std::set<rocprofiler_callback_tracing_code_object_load_data_t,
                                  info::code_object_less>>
        m_code_objects;
    common::synchronized<
        std::set<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t,
                 info::kernel_symbol_less>>
        m_kernel_symbols;

    common::synchronized<std::set<uint64_t>>                   m_streams;
    common::synchronized<std::set<uint64_t>>                   m_queues;
    common::synchronized<std::unordered_set<std::string_view>> m_strings;
    rocprofiler::sdk::buffer_name_info_t<const char*>          m_buffered_tracing_info{
        rocprofiler::sdk::get_buffer_tracing_names<const char*>()
    };
    rocprofiler::sdk::callback_name_info_t<const char*> m_callback_tracing_info{
        rocprofiler::sdk::get_callback_tracing_names<const char*>()
    };
};

}  // namespace sample_cache
}  // namespace rocprofsys
