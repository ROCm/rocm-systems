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
#include "core/debug.hpp"
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

#if ROCPROFSYS_USE_ROCM > 0

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

// As the underlying implementation of callback_name_info_t resizes the category storage
// during emplace, this special method is required
void
metadata_registry::overwrite_callback_names(
    std::initializer_list<
        std::pair<rocprofiler_callback_tracing_kind_t, callback_rename_map_t>>
        rename_table)
{
    if(rename_table.size() == 0) return;

    using callback_kind_t   = rocprofiler_callback_tracing_kind_t;
    using operation_names_t = std::vector<std::string_view>;

    auto category_names = std::vector<std::string_view>{};
    auto modified_ops   = std::map<callback_kind_t, operation_names_t>{};

    auto extract_operations = [&](callback_kind_t cat) -> operation_names_t {
        auto        items           = m_callback_tracing_info.items();
        const auto* target_category = items[static_cast<size_t>(cat)];

        auto              operations_data = target_category->items();
        operation_names_t operation_names;
        operation_names.reserve(operations_data.size());

        for(const auto& [op_idx, op_name] : operations_data)
            operation_names.push_back(*op_name);

        return operation_names;
    };

    // Store category names
    category_names.resize(ROCPROFILER_CALLBACK_TRACING_LAST);
    for(callback_kind_t i = ROCPROFILER_CALLBACK_TRACING_NONE;
        i < ROCPROFILER_CALLBACK_TRACING_LAST;
        i = static_cast<callback_kind_t>(static_cast<int>(i) + 1))
    {
        category_names[i] = m_callback_tracing_info.at(i);
    }

    // Process list
    for(const auto& category_info : rename_table)
    {
        auto callback_kind = category_info.first;
        // Store operations of all following categories
        //  as they will be deleted
        for(callback_kind_t i =
                static_cast<callback_kind_t>(static_cast<int>(callback_kind) + 1);
            i < ROCPROFILER_CALLBACK_TRACING_LAST;
            i = static_cast<callback_kind_t>(static_cast<int>(i) + 1))
        {
            if(modified_ops.find(i) != modified_ops.end()) break;
            modified_ops[i] = extract_operations(i);
        }

        ROCPROFSYS_CI_THROW(modified_ops.find(callback_kind) != modified_ops.end(),
                            "Overwriting a previously overwritten entry is forbidden");

        ROCPROFSYS_CI_THROW(!modified_ops.empty() &&
                                callback_kind >= modified_ops.begin()->first,
                            "Category must have a larger enum value than all previously "
                            "modified_ops categories");

        // Overwrite desired category
        auto operation_names = extract_operations(callback_kind);
        for(const auto& [index, new_value] : category_info.second)
        {
            ROCPROFSYS_CI_THROW(index < 0 ||
                                    static_cast<size_t>(index) >= operation_names.size(),
                                "Index is invalid");
            operation_names[index] = new_value;
        }
        modified_ops[callback_kind] = std::move(operation_names);
    }
    if(modified_ops.empty()) return;

    // Emplace the changed category operations
    for(callback_kind_t i = modified_ops.begin()->first;
        i < ROCPROFILER_CALLBACK_TRACING_LAST;
        i = static_cast<callback_kind_t>(static_cast<int>(i) + 1))
    {
        auto renaming_entry = modified_ops.find(i);

        ROCPROFSYS_CI_THROW(renaming_entry == modified_ops.end(),
                            "A category that needs to be emplaced is missing");

        const auto& operations_vec = renaming_entry->second;
        m_callback_tracing_info.emplace(i, category_names.at(i).data());
        for(size_t op_idx = 0; op_idx < operations_vec.size(); ++op_idx)
        {
            m_callback_tracing_info.emplace(
                i, static_cast<rocprofiler_tracing_operation_t>(op_idx),
                operations_vec[op_idx].data());
        }
    }
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

metadata_registry::metadata_registry()
{
#if ROCPROFSYS_USE_ROCM > 0
    overwrite_callback_names({
#    if(ROCPROFILER_VERSION >= 600)
        { ROCPROFILER_CALLBACK_TRACING_OMPT,
          { { ROCPROFILER_OMPT_ID_thread_begin, "omp_thread" },
            { ROCPROFILER_OMPT_ID_thread_end, "omp_thread" },
            { ROCPROFILER_OMPT_ID_parallel_begin, "omp_parallel" },
            { ROCPROFILER_OMPT_ID_parallel_end, "omp_parallel" } } }
#    endif
    });
#endif
}

}  // namespace trace_cache
}  // namespace rocprofsys
