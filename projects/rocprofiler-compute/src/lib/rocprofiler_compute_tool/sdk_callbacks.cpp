// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "sdk_callbacks.h"

#include "gsl_assert.h"

#include <cxxabi.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>

using namespace rocprofiler_compute_tool;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

SdkCallbacksImpl::SdkCallbacksImpl(const std::shared_ptr<SdkWrapper>& sdk_wrapper)
    : m_sdk_wrapper(sdk_wrapper)
{
    Expects(m_sdk_wrapper)
}

void SdkCallbacksImpl::dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                         rocprofiler_counter_config_id_t*             config,
                                         void* callback_data_args)
{
    auto kernel_id = dispatch_data.dispatch_info.kernel_id;
    auto agent_id  = dispatch_data.dispatch_info.agent_id.handle;

    uint64_t kernel_dispatch_count = 0;
    {
        // Acquire unique lock for update and ensure map is updated correctly
        std::unique_lock<std::shared_mutex> lock(m_kernel_id_iteration_mutex);
        auto&                               count = m_kernel_dispatch_count_by_kernel_id[kernel_id];
        count += 1;
        kernel_dispatch_count = count;
    }

    // static cast tool
    auto*        tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(callback_data_args);
    tool_data_t* tool;
    {
        std::lock_guard<std::mutex> lock(tool_data_ptr->get()->mut);
        tool = tool_data_ptr->get();
    }

    // kernel filtering
    if (!is_targeted_dispatch(tool, kernel_id, kernel_dispatch_count))
    {
        return;
    }

    // check cache for existing profile for this agent
    auto search_profile_cache = [&]()
    {
        if (auto pos = m_profile_cache_per_agent.find(agent_id); pos != m_profile_cache_per_agent.end())
            return true;
        return false;
    };

    auto set_config_from_cache = [&]()
    {
        if (tool->iteration_multiplexing_mode != iteration_multiplexing_mode_t::DISABLED &&
            m_iteration_multiplexing_per_agent.find(agent_id) == m_iteration_multiplexing_per_agent.end())
        {
            // First time setting up iteration multiplexing data for this agent
            m_iteration_multiplexing_per_agent[agent_id] = iteration_multiplexing_dispatch_record_t{};
            if (tool->iteration_multiplexing_mode == iteration_multiplexing_mode_t::SIMPLE)
            {
                m_iteration_multiplexing_per_agent[agent_id].config = -1;  // so first increment sets to 0
            }
        }

        kernel_dispatch_info_t dispatch_info{dispatch_data.dispatch_info.kernel_id,
                                             dispatch_data.dispatch_info.queue_id.handle,
                                             dispatch_data.dispatch_info.workgroup_size,
                                             dispatch_data.dispatch_info.grid_size,
                                             dispatch_data.dispatch_info.group_segment_size};
        switch (tool->iteration_multiplexing_mode)
        {
        case iteration_multiplexing_mode_t::DISABLED:
            if (search_profile_cache())
            {
                *config = m_profile_cache_per_agent[agent_id][0];
            }
            return;

        case iteration_multiplexing_mode_t::SIMPLE:
            m_iteration_multiplexing_per_agent[agent_id].config =
                (m_iteration_multiplexing_per_agent[agent_id].config + 1) %
                m_profile_cache_per_agent[agent_id].size();
            *config =
                m_profile_cache_per_agent[agent_id][m_iteration_multiplexing_per_agent[agent_id].config];
            return;

        case iteration_multiplexing_mode_t::KERNEL:
            if (m_iteration_multiplexing_per_agent[agent_id].kernel_id_to_profile_index.find(kernel_id) ==
                m_iteration_multiplexing_per_agent[agent_id].kernel_id_to_profile_index.end())
            {
                // First time seeing this kernel_id for this agent
                m_iteration_multiplexing_per_agent[agent_id].kernel_id_to_profile_index[kernel_id] =
                    -1;  // so first increment sets to 0
            }
            m_iteration_multiplexing_per_agent[agent_id].kernel_id_to_profile_index[kernel_id] =
                (m_iteration_multiplexing_per_agent[agent_id].kernel_id_to_profile_index[kernel_id] + 1) %
                m_profile_cache_per_agent[agent_id].size();
            *config = m_profile_cache_per_agent[agent_id][m_iteration_multiplexing_per_agent[agent_id]
                                                              .kernel_id_to_profile_index[kernel_id]];
            return;

        case iteration_multiplexing_mode_t::LAUNCH:
            if (m_iteration_multiplexing_per_agent[agent_id].kernel_params_to_profile_index.find(
                    dispatch_info) ==
                m_iteration_multiplexing_per_agent[agent_id].kernel_params_to_profile_index.end())
            {
                // First time seeing this dispatch_info for this agent
                m_iteration_multiplexing_per_agent[agent_id].kernel_params_to_profile_index[dispatch_info] =
                    -1;  // so first increment sets to 0
            }
            m_iteration_multiplexing_per_agent[agent_id].kernel_params_to_profile_index[dispatch_info] =
                (m_iteration_multiplexing_per_agent[agent_id].kernel_params_to_profile_index[dispatch_info] +
                 1) %
                m_profile_cache_per_agent[agent_id].size();
            *config = m_profile_cache_per_agent[agent_id][m_iteration_multiplexing_per_agent[agent_id]
                                                              .kernel_params_to_profile_index[dispatch_info]];
            return;

        default:
            throw std::runtime_error("[" + std::string(__FUNCTION__) +
                                     "] Unsupported iteration multiplexing mode");
        }
    };

    {
        auto rlock = std::shared_lock{m_mutex};
        if ((tool->iteration_multiplexing_mode == iteration_multiplexing_mode_t::DISABLED) &&
            search_profile_cache())
        {
            *config = m_profile_cache_per_agent[agent_id][0];
            return;
        }
    }

    // get write lock to update cache
    auto wlock = std::unique_lock{m_mutex};
    if (search_profile_cache())
    {
        set_config_from_cache();
        return;
    }

    create_counter_collection_profile(tool, dispatch_data.dispatch_info.agent_id, m_profile_cache_per_agent);

    // Return the profile to collect those counters for this dispatch
    set_config_from_cache();
}

bool SdkCallbacksImpl::is_targeted_dispatch(const tool_data_t* tool, uint64_t kernel_id, uint64_t kernel_iteration)
{
    if (!tool->target_kernel_ids.empty() && !tool->target_kernel_ids.count(kernel_id))
        return false;

    if (!tool->kernel_filter_ranges.empty())
        return std::any_of(tool->kernel_filter_ranges.begin(),
                           tool->kernel_filter_ranges.end(),
                           [kernel_iteration](const auto& range)
                           {
                               return kernel_iteration >= range.first && kernel_iteration <= range.second;
                           });

    return true;
}

void SdkCallbacksImpl::create_counter_collection_profile(
    tool_data_t*                                                                tool,
    rocprofiler_agent_id_t                                                      agent_id,
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_config_id_t>>& profile_cache) const
{
    // get counters to collect
    std::set<std::set<std::string>> counters_to_collect;
    for (const std::string& counters_str : split_by_regex(tool->requested_counters, "[,]"))
    {
        if (!counters_str.empty())
        {
            auto pos = counters_str.find(':');
            if (pos != std::string::npos)
            {
                std::istringstream    ss(counters_str.substr(pos + 1));
                std::set<std::string> counters;
                for (std::string token; ss >> token;)
                {
                    counters.insert(token);
                }
                counters_to_collect.insert(counters);
            }
        }
    }

    // Get available counters for this agent
    std::vector<rocprofiler_counter_id_t> gpu_counters;
    m_sdk_wrapper->iterate_agent_supported_counters(
        agent_id,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters, size_t num_counters, void* user_data)
        {
            std::vector<rocprofiler_counter_id_t>* vec =
                static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
            for (size_t i = 0; i < num_counters; i++)
            {
                vec->push_back(counters[i]);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        },
        static_cast<void*>(&gpu_counters));

    std::vector<std::string>                        gpu_counter_names;
    std::map<std::string, rocprofiler_counter_id_t> gpu_counter_map;
    for (auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t info;
        m_sdk_wrapper->query_counter_info(counter,
                                          ROCPROFILER_COUNTER_INFO_VERSION_0,
                                          static_cast<void*>(&info));
        gpu_counter_names.push_back(std::string(info.name));
        gpu_counter_map.insert({std::string(info.name), counter});
    }

    // Identify counters requested to collect which are available
    std::vector<std::vector<std::string>>              collect_counter_names;
    std::vector<std::vector<rocprofiler_counter_id_t>> collect_counters;
    std::vector<std::string>                           unsupported_counters;
    for (const auto& counters : counters_to_collect)
    {
        std::vector<std::string>              counter_names;
        std::vector<rocprofiler_counter_id_t> counter_ids;
        for (const auto& counter_name : counters)
        {
            if (std::find(gpu_counter_names.begin(), gpu_counter_names.end(), counter_name) !=
                gpu_counter_names.end())
            {
                counter_names.push_back(counter_name);
                counter_ids.push_back(gpu_counter_map[counter_name]);
                tool->counter_id_name_map[gpu_counter_map[counter_name].handle] = counter_name;
            }
            else
            {
                unsupported_counters.push_back(counter_name);
            }
        }
        collect_counter_names.push_back(counter_names);
        collect_counters.push_back(counter_ids);
    }

    if (!unsupported_counters.empty())
    {
        std::clog << "\033[33m[rocprofiler-compute] [" << __FUNCTION__
                  << "] WARNING: Requested counters not available: ";
        for (size_t i = 0; i < unsupported_counters.size(); ++i)
        {
            std::clog << unsupported_counters[i];
            if (i + 1 < unsupported_counters.size())
                std::clog << ", ";
        }
        std::clog << "\033[0m" << std::endl;
    }

    // Create a profile cache for the agent
    std::vector<rocprofiler_counter_config_id_t> profiles{};
    // Create a collection profile for the counters
    for (auto& collect_counters_one_iter : collect_counters)
    {
        rocprofiler_counter_config_id_t profile = {.handle = 0};
        m_sdk_wrapper->create_counter_config(agent_id,
                                             collect_counters_one_iter.data(),
                                             collect_counters_one_iter.size(),
                                             &profile);
        profiles.push_back(profile);
        profile_cache[agent_id.handle] = profiles;
    }
}

void SdkCallbacksImpl::record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                       rocprofiler_counter_record_t*                record_data,
                                       size_t                                       record_count,
                                       void* callback_data_args)
{
    auto*        tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(callback_data_args);
    tool_data_t* tool;
    {
        std::lock_guard<std::mutex> lock(tool_data_ptr->get()->mut);
        tool = tool_data_ptr->get();
    }

    // For each counter, write: dispatch_id, counter_id, counter_name,
    // counter_value
    for (size_t i = 0; i < record_count; ++i)
    {
        rocprofiler_counter_id_t counter_id{};
        m_sdk_wrapper->query_record_counter_id(record_data[i].id, &counter_id);

        // Store the counter info record in tool_data
        counter_info_record_t record{dispatch_data.dispatch_info.dispatch_id,
                                     dispatch_data.dispatch_info.agent_id.handle,
                                     dispatch_data.dispatch_info.kernel_id,
                                     dispatch_data.dispatch_info.group_segment_size,
                                     counter_id.handle,
                                     tool->counter_id_name_map[counter_id.handle],
                                     record_data[i].counter_value};
        {
            std::lock_guard<std::mutex> lock(tool->mut);
            tool->counter_records.push_back(std::move(record));
        }
    }
}

void SdkCallbacksImpl::tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                             void*                                 callback_data)
{
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD &&
        record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
        record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        // check if regex can be found in kernel name matches regex from tool data,
        // if matches store kernel id
        auto* tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(callback_data);
        auto* tool          = tool_data_ptr->get();
        // Lock before modifying target_kernel_ids
        std::lock_guard<std::mutex> lock(tool->mut);
        if (!tool->kernel_filter_include_regex.empty())
        {
            try
            {
                int  demangle_status = 0;
                auto kernel_name     = cxa_demangle(data->kernel_name, &demangle_status);
                kernel_name          = truncate_name(kernel_name);

                std::regex re(tool->kernel_filter_include_regex);
                if (!kernel_name.empty() && std::regex_search(kernel_name, re))
                {
                    tool->target_kernel_ids.insert(data->kernel_id);
                }
            }
            catch (const std::regex_error& e)
            {
                std::cerr << "[rocprofiler-compute] [" << __FUNCTION__
                          << "] ERROR: Invalid regex in ROCPROF_KERNEL_FILTER_INCLUDE_REGEX: "
                          << tool->kernel_filter_include_regex << " : " << e.what() << std::endl;
            }
        }
        // If no regex specified, collect for all kernels
        else
        {
            tool->target_kernel_ids.insert(data->kernel_id);
        }
    }
}

std::string SdkCallbacksImpl::truncate_name(std::string_view name)
{
    // The function extracts the kernel name from
    // input string. By using the iterators it finds the
    // window in the string which contains only the kernel name.
    // For example 'Foo<int, float>::foo(a[], int (int))' -> 'foo'
    auto     rit         = name.rbegin();
    auto     rend        = name.rend();
    uint32_t counter     = 0;
    char     open_token  = 0;
    char     close_token = 0;
    while (rit != rend)
    {
        if (counter == 0)
        {
            switch (*rit)
            {
            case ')':
                counter     = 1;
                open_token  = ')';
                close_token = '(';
                break;
            case '>':
                counter     = 1;
                open_token  = '>';
                close_token = '<';
                break;
            case ']':
                counter     = 1;
                open_token  = ']';
                close_token = '[';
                break;
            case ' ':
                ++rit;
                continue;
            }
            if (counter == 0)
                break;
        }
        else
        {
            if (*rit == open_token)
                counter++;
            if (*rit == close_token)
                counter--;
        }
        ++rit;
    }
    auto rbeg = rit;
    while ((rit != rend) && (*rit != ' ') && (*rit != ':'))
        rit++;
    return std::string{name.substr(rend - rit, rit - rbeg)};
}

std::string SdkCallbacksImpl::cxa_demangle(const std::string& mangled_name, int* status)
{
    // return the mangled since there is no buffer
    if (mangled_name.empty())
    {
        *status = -2;
        return std::string{};
    }

    auto _demangled_name = std::string{mangled_name};

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be
    //      demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into
    //      which the demangled name is stored. If output_buffer is not long
    //      enough, it is expanded using realloc. output_buffer may instead be
    //      NULL; in that case, the demangled name is placed in a region of memory
    //      allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the
    //      demangled name is placed in *length.
    //  status:
    //      *status is set to one of the following values
    size_t _demang_len = 0;
    char*  _demang = abi::__cxa_demangle(_demangled_name.c_str(), nullptr, &_demang_len, status);
    switch (*status)
    {
    //  0 : The demangling operation succeeded.
    // -1 : A memory allocation failure occurred.
    // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
    // -3 : One of the arguments is invalid.
    case 0:
    {
        if (_demang)
            _demangled_name = std::string{_demang};
        break;
    }
    case -1:
    {
        std::clog << "[rocprofiler-compute] memory allocation failure occurred "
                     "demangling "
                  << _demangled_name << std::endl;
        break;
    }
    case -2:
    {
        break;
    }
    case -3:
    {
        std::clog << "[rocprofiler-compute] Invalid argument in: (\"" << _demangled_name
                  << "\", nullptr, nullptr, " << static_cast<void*>(status) << ")" << std::endl;
        break;
    }
    default:
        break;
    };

    // if it "demangled" but the length is zero, set the status to -2
    if (_demang_len == 0 && *status == 0)
        *status = -2;

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
}

std::vector<std::string> SdkCallbacksImpl::split_by_regex(const std::string& s,
                                                          const std::string& regex_pattern)
{
    std::vector<std::string> tokens;
    std::regex               re(regex_pattern);

    // -1 indicates to return the submatches that are not part of the delimiter
    // itself
    std::sregex_token_iterator iter(s.begin(), s.end(), re, -1);
    std::sregex_token_iterator end;

    while (iter != end)
    {
        // Ensure that empty strings resulting from consecutive delimiters are not
        // added
        if (!iter->str().empty())
        {
            tokens.push_back(*iter);
        }
        ++iter;
    }
    return tokens;
}
