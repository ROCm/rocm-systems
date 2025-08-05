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

#include "metadata_registry.hpp"
#include <algorithm>
#include <cstdint>

namespace rocprofsys
{
namespace trace_cache
{

namespace
{

template <typename ReturnType, typename DataType, typename Filter>
std::optional<ReturnType>
get_type_info(const DataType& data, const Filter& filter)
{
    std::optional<ReturnType> result = std::nullopt;
    data.rlock([&filter, &result](const auto& _data) {
        auto it = std::find_if(_data.begin(), _data.end(), filter);
        result  = it == _data.end() ? std::nullopt : std::optional<ReturnType>(*it);
    });
    return result;
}

template <typename T>
auto
assign_set_to_vector(T& result)
{
    return [&result](const auto& _data) { result.assign(_data.cbegin(), _data.cend()); };
}
}  // namespace

void
metadata_registry::set_process(const info::process& process)
{
    m_process.wlock([&process](auto& _process) { _process = process; });
}

void
metadata_registry::add_pmc_info(const info::pmc& pmc_info)
{
    m_pmc_infos.wlock([&pmc_info](auto& _data) {
        if(_data.count(pmc_info) > 0)
        {
            return;
        }
        _data.emplace(pmc_info);
    });
}

void
metadata_registry::add_thread_info(const info::thread& thread_info)
{
    m_threads.wlock([&thread_info](auto& _data) {
        if(_data.count(thread_info) > 0)
        {
            return;
        }
        _data.emplace(thread_info);
    });
}

void
metadata_registry::add_track(const info::track& track_info)
{
    m_tracks.wlock([&track_info](auto& _data) {
        if(_data.count(track_info) > 0)
        {
            return;
        }
        _data.emplace(track_info);
    });
}

void
metadata_registry::add_queue(const uint64_t& queue_handle)
{
    m_queues.wlock([&queue_handle](auto& _data) {
        if(_data.count(queue_handle) > 0)
        {
            return;
        }
        _data.emplace(queue_handle);
    });
}

void
metadata_registry::add_stream(const uint64_t& stream_handle)
{
    m_streams.wlock([&stream_handle](auto& _data) {
        if(_data.count(stream_handle) > 0)
        {
            return;
        }
        _data.emplace(stream_handle);
    });
}

void
metadata_registry::add_string(const std::string_view& string_value)
{
    m_strings.wlock([&string_value](auto& _data) {
        if(_data.count(string_value) > 0)
        {
            return;
        }
        _data.emplace(string_value);
    });
}

info::process
metadata_registry::get_process_info() const
{
    info::process result;
    m_process.rlock([&result](const auto& _process) { result = _process; });
    return result;
}

std::optional<info::pmc>
metadata_registry::get_pmc_info(const std::string_view& unique_name) const
{
    return get_type_info<info::pmc>(m_pmc_infos, [&unique_name](const info::pmc& val) {
        return val.name == unique_name;
    });
}

std::optional<info::thread>
metadata_registry::get_thread_info(const uint32_t& thread_id) const
{
    return get_type_info<info::thread>(m_threads, [&thread_id](const info::thread& val) {
        return val.thread_id == thread_id;
    });
}

std::optional<info::track>
metadata_registry::get_track_info(const std::string_view& track_name) const
{
    return get_type_info<info::track>(m_tracks, [&track_name](const info::track& val) {
        return val.track_name == track_name;
    });
}

std::vector<info::pmc>
metadata_registry::get_pmc_info_list() const
{
    std::vector<info::pmc> result;
    m_pmc_infos.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<info::thread>
metadata_registry::get_thread_info_list() const
{
    std::vector<info::thread> result;
    m_threads.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<info::track>
metadata_registry::get_track_info_list() const
{
    std::vector<info::track> result;
    m_tracks.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<uint64_t>
metadata_registry::get_queue_list() const
{
    std::vector<uint64_t> result;
    m_queues.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<uint64_t>
metadata_registry::get_stream_list() const
{
    std::vector<uint64_t> result;
    m_streams.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<std::string_view>
metadata_registry::get_string_list() const
{
    std::vector<std::string_view> result;
    m_strings.rlock(assign_set_to_vector(result));
    return result;
}

#if ROCPROFSYS_USE_ROCM

void
metadata_registry::add_code_object(
    const rocprofiler_callback_tracing_code_object_load_data_t& code_object)
{
    m_code_objects.wlock([&code_object](auto& _data) {
        if(_data.count(code_object) > 0)
        {
            return;
        }
        _data.emplace(code_object);
    });
}

void
metadata_registry::add_kernel_symbol(
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t&
        kernel_symbol)
{
    m_kernel_symbols.wlock([&kernel_symbol](auto& _data) {
        if(_data.count(kernel_symbol) > 0)
        {
            return;
        }
        _data.emplace(kernel_symbol);
    });
}

std::optional<rocprofiler_callback_tracing_code_object_load_data_t>
metadata_registry::get_code_object(uint64_t code_object_id) const
{
    return get_type_info<rocprofiler_callback_tracing_code_object_load_data_t>(
        m_code_objects,
        [&code_object_id](
            const rocprofiler_callback_tracing_code_object_load_data_t& val) {
            return val.code_object_id == code_object_id;
        });
}

std::optional<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>
metadata_registry::get_kernel_symbol(uint64_t kernel_id) const
{
    return get_type_info<
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>(
        m_kernel_symbols,
        [&kernel_id](
            const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t&
                val) { return val.kernel_id == kernel_id; });
}

std::vector<rocprofiler_callback_tracing_code_object_load_data_t>
metadata_registry::get_code_object_list() const
{
    std::vector<rocprofiler_callback_tracing_code_object_load_data_t> result;
    m_code_objects.rlock(assign_set_to_vector(result));
    return result;
}

std::vector<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>
metadata_registry::get_kernel_symbol_list() const
{
    std::vector<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t>
        result;
    m_kernel_symbols.rlock(assign_set_to_vector(result));
    return result;
}

rocprofiler::sdk::buffer_name_info_t<const char*>
metadata_registry::get_buffer_name_info() const
{
    return m_buffered_tracing_info;
}

rocprofiler::sdk::callback_name_info_t<const char*>
metadata_registry::get_callback_tracing_info() const
{
    return m_callback_tracing_info;
}

#endif

}  // namespace trace_cache
}  // namespace rocprofsys
