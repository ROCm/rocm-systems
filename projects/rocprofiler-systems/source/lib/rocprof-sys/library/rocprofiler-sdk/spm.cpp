// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file is compiled only when rocprofiler-sdk/experimental/spm.h is
// available. Keep SDK SPM runtime wiring, callbacks, and agent configuration
// here; configuration-only code and no-SDK fallbacks belong in spm_config.cpp.

#include "library/rocprofiler-sdk/spm.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/utility.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"

#include "logger/debug.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace spm
{
namespace
{
constexpr auto invalid_context_handle = 0UL;

struct requested_counter
{
    std::string                  name      = {};
    std::optional<std::uint64_t> device_id = std::nullopt;
};

struct resolved_counter
{
    rocprofiler_counter_id_t                               id           = {};
    std::vector<trace_cache::info::spm_counter_name_entry> name_entries = {};
};

using spm_available_config_vec_t = std::vector<rocprofiler_spm_available_configuration_t>;
using spm_counter_id_vec_t       = std::vector<rocprofiler_counter_id_t>;
using resolved_counter_vec_t     = std::vector<resolved_counter>;
using requested_counter_vec_t    = std::vector<requested_counter>;
namespace gpu_perf_counter       = pmc::collectors::gpu_perf_counter;

enum class counter_query_status
{
    success,
    skipped,
    failed,
};

struct counter_query_result
{
    counter_query_status   status   = counter_query_status::failed;
    resolved_counter_vec_t counters = {};
};

enum class agent_config_status
{
    configured,
    skipped,
    failed,
};

struct agent_config_result
{
    agent_config_status             status = agent_config_status::failed;
    rocprofiler_counter_config_id_t config = {};
};

struct counter_config_data
{
    spm_counter_id_vec_t                                   counter_ids  = {};
    std::vector<trace_cache::info::spm_counter_name_entry> name_entries = {};
};

rocprofiler_status_t
spm_configurations_callback(const rocprofiler_spm_available_configuration_t** configs,
                            size_t num_configs, void* user_data)
{
    auto* output = static_cast<spm_available_config_vec_t*>(user_data);
    if(output == nullptr || (configs == nullptr && num_configs > 0))
        return ROCPROFILER_STATUS_ERROR;

    output->reserve(num_configs);
    for(size_t i = 0; i < num_configs; ++i)
    {
        if(configs[i] != nullptr) output->emplace_back(*configs[i]);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
spm_supported_counters_callback(rocprofiler_agent_id_t /*agent_id*/,
                                rocprofiler_counter_id_t* counters, size_t num_counters,
                                void* user_data)
{
    auto* output = static_cast<spm_counter_id_vec_t*>(user_data);
    if(output == nullptr || (counters == nullptr && num_counters > 0))
        return ROCPROFILER_STATUS_ERROR;

    if(num_counters == 0) return ROCPROFILER_STATUS_SUCCESS;
    output->assign(counters, counters + num_counters);
    return ROCPROFILER_STATUS_SUCCESS;
}

std::optional<std::uint64_t>
parse_device_id(std::string_view value)
{
    std::uint64_t result = 0;
    if(value.empty()) return std::nullopt;

    const auto* first  = value.data();
    const auto* last   = value.data() + value.size();
    const auto  parsed = std::from_chars(first, last, result);
    if(parsed.ec != std::errc{} || parsed.ptr != last) return std::nullopt;
    return result;
}

std::string
parse_counter_name(std::string_view event)
{
    constexpr auto device_qualifier = std::string_view{ ":device=" };

    auto name = std::string{ event.substr(0, event.find(device_qualifier)) };
    utility::trim_str(name);
    return name;
}

requested_counter_vec_t
parse_requested_counters(const request& req)
{
    constexpr auto device_qualifier = std::string_view{ ":device=" };

    auto counters = requested_counter_vec_t{};
    counters.reserve(req.events.size());
    for(const auto& event : req.events)
    {
        auto trimmed_event = event;
        utility::trim_str(trimmed_event);
        if(trimmed_event.empty()) continue;

        auto       name = parse_counter_name(trimmed_event);
        const auto pos  = trimmed_event.find(device_qualifier);
        if(pos == std::string::npos)
        {
            counters.push_back({ std::move(name), std::nullopt });
            continue;
        }

        auto device = parse_device_id(
            std::string_view{ trimmed_event }.substr(pos + device_qualifier.size()));

        if(name.empty() || !device)
        {
            LOG_WARNING("Invalid SPM device qualifier '{}'. Expected COUNTER:device=N",
                        event);
            continue;
        }

        counters.push_back({ std::move(name), *device });
    }
    return counters;
}

bool
requested_on_device(const requested_counter& counter, std::uint64_t device_id)
{
    return !counter.device_id || *counter.device_id == device_id;
}

bool
sample_interval_supported(rocprofiler_agent_id_t agent_id, const request& req)
{
    auto configs = spm_available_config_vec_t{};
    auto status  = rocprofiler_spm_query_agent_configurations(
        agent_id, spm_configurations_callback, &configs);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to query SPM configurations for agent {}: {} ({})",
                    agent_id.handle, static_cast<int>(status),
                    rocprofiler_get_status_string(status));
        return false;
    }

    return std::any_of(configs.begin(), configs.end(), [&req](const auto& config) {
        return config.type ==
                   ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES &&
               config.interval.min_interval <= req.sample_interval &&
               config.interval.max_interval >= req.sample_interval;
    });
}

requested_counter_vec_t
requested_counters_for_device(const requested_counter_vec_t& all_requested,
                              std::uint64_t                  device_id)
{
    auto requested = requested_counter_vec_t{};
    requested.reserve(all_requested.size());
    std::copy_if(
        all_requested.begin(), all_requested.end(), std::back_inserter(requested),
        [device_id](const auto& itr) { return requested_on_device(itr, device_id); });
    return requested;
}

std::unordered_set<std::string>
requested_counter_names(const requested_counter_vec_t& requested)
{
    auto requested_names = std::unordered_set<std::string>{};
    for(const auto& itr : requested)
        requested_names.emplace(itr.name);
    return requested_names;
}

counter_config_data
make_counter_config_data(resolved_counter_vec_t& counters)
{
    auto config_data = counter_config_data{};
    config_data.counter_ids.reserve(counters.size());

    const auto name_entry_count =
        std::accumulate(counters.begin(), counters.end(), std::size_t{ 0 },
                        [](std::size_t total, const auto& counter) {
                            return total + counter.name_entries.size();
                        });
    config_data.name_entries.reserve(name_entry_count);

    for(auto& counter : counters)
    {
        config_data.counter_ids.emplace_back(counter.id);
        config_data.name_entries.insert(
            config_data.name_entries.end(),
            std::make_move_iterator(counter.name_entries.begin()),
            std::make_move_iterator(counter.name_entries.end()));
    }

    return config_data;
}

std::string
safe_str(const char* value)
{
    return value != nullptr ? std::string{ value } : std::string{};
}

std::vector<gpu_perf_counter::dimension_position>
counter_dimensions(const rocprofiler_counter_record_dimension_instance_info_t& instance)
{
    auto dims = std::vector<gpu_perf_counter::dimension_position>{};
    if(instance.dimensions == nullptr) return dims;
    dims.reserve(instance.dimensions_count);

    for(std::uint64_t i = 0; i < instance.dimensions_count; ++i)
    {
        const auto* dim = instance.dimensions[i];
        if(dim == nullptr) continue;
        dims.push_back({ safe_str(dim->dimension_name), dim->index });
    }
    return dims;
}

std::vector<trace_cache::info::spm_counter_name_entry>
spm_counter_name_entries(const rocprofiler_counter_info_v1_t& info)
{
    auto entries = std::vector<trace_cache::info::spm_counter_name_entry>{};
    auto name    = safe_str(info.name);

    if(info.dimensions_instances == nullptr) return entries;
    entries.reserve(info.dimensions_instances_count);

    for(std::uint64_t i = 0; i < info.dimensions_instances_count; ++i)
    {
        const auto* instance = info.dimensions_instances[i];
        if(instance == nullptr) continue;

        auto metadata = gpu_perf_counter::counter_metadata{
            instance->instance_id,      name,
            safe_str(info.description), safe_str(info.block),
            safe_str(info.expression),  info.is_constant != 0,
            info.is_derived != 0,       counter_dimensions(*instance),
        };

        entries.push_back({ info.id.handle, instance->instance_id,
                            gpu_perf_counter::make_qualified_name(metadata) });
    }

    return entries;
}

counter_query_result
query_supported_spm_counters(rocprofiler_agent_id_t                 agent_id,
                             const std::unordered_set<std::string>& requested_names,
                             std::uint64_t                          device_id)
{
    auto supported = spm_counter_id_vec_t{};
    auto status    = rocprofiler_spm_iterate_agent_supported_counters(
        agent_id, spm_supported_counters_callback, &supported);
    // The SDK reports an unsupported architecture as an error status, but it is an
    // expected capability result rather than a broken query.
    if(status == ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED)
    {
        LOG_WARNING("SPM is not supported on the architecture of device {} (agent {})",
                    device_id, agent_id.handle);
        return { counter_query_status::skipped, {} };
    }

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to query SPM counters for agent {}: {} ({})", agent_id.handle,
                    static_cast<int>(status), rocprofiler_get_status_string(status));
        return { counter_query_status::failed, {} };
    }

    auto counters = resolved_counter_vec_t{};
    auto matched  = std::unordered_set<std::string>{};
    for(const auto& counter : supported)
    {
        auto info = rocprofiler_counter_info_v1_t{};
        status    = rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_1, &info);
        if(status != ROCPROFILER_STATUS_SUCCESS || info.name == nullptr ||
           info.spm_support == 0)
            continue;

        auto name = std::string{ info.name };
        if(requested_names.count(name) > 0)
        {
            auto name_entries = spm_counter_name_entries(info);
            if(name_entries.empty())
            {
                LOG_WARNING("Requested SPM counter '{}' has no SDK dimension-instance "
                            "metadata for device {} (agent {})",
                            name, device_id, agent_id.handle);
                return { counter_query_status::skipped, {} };
            }

            counters.push_back({ counter, std::move(name_entries) });
            matched.emplace(std::move(name));
        }
    }

    if(matched.size() == requested_names.size())
        return { counter_query_status::success, std::move(counters) };

    for(const auto& name : requested_names)
    {
        if(matched.count(name) == 0)
        {
            LOG_WARNING("Requested SPM counter '{}' is not supported for device {} "
                        "(agent {})",
                        name, device_id, agent_id.handle);
        }
    }

    return { counter_query_status::skipped, {} };
}

std::optional<rocprofiler_counter_config_id_t>
create_sdk_spm_counter_config(rocprofiler_agent_id_t agent_id, std::uint64_t device_id,
                              const request& req, spm_counter_id_vec_t& counters)
{
    auto param = rocprofiler_spm_parameters_t{
        sizeof(rocprofiler_spm_parameters_t),
        ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
        req.sample_interval,
    };
    auto params = std::array<rocprofiler_spm_parameters_t*, 1>{ &param };
    auto config = rocprofiler_counter_config_id_t{};

    LOG_DEBUG("Creating SPM counter config for device {} (agent {}) with {} counters",
              device_id, agent_id.handle, counters.size());

    auto status =
        rocprofiler_spm_create_counter_config(agent_id, counters.data(), counters.size(),
                                              params.data(), params.size(), &config);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to create SPM counter config for device {} (agent {}): {} "
                    "({})",
                    device_id, agent_id.handle, static_cast<int>(status),
                    rocprofiler_get_status_string(status));
        return std::nullopt;
    }

    return config;
}

agent_config_result
create_agent_spm_config(rocprofiler_agent_id_t agent_id, std::uint64_t device_id,
                        const request& req, const requested_counter_vec_t& requested)
{
    const auto requested_names = requested_counter_names(requested);
    auto counters = query_supported_spm_counters(agent_id, requested_names, device_id);
    if(counters.status == counter_query_status::skipped)
        return { agent_config_status::skipped, {} };
    if(counters.status == counter_query_status::failed)
        return { agent_config_status::failed, {} };

    if(!sample_interval_supported(agent_id, req))
    {
        LOG_WARNING("SPM sample interval {} SCLK cycles is not supported for device {} "
                    "(agent {})",
                    req.sample_interval, device_id, agent_id.handle);
        return { agent_config_status::failed, {} };
    }

    auto config_data = make_counter_config_data(counters.counters);
    auto config =
        create_sdk_spm_counter_config(agent_id, device_id, req, config_data.counter_ids);
    if(config)
    {
        trace_cache::get_metadata_registry().set_spm_counter_names(
            static_cast<std::uint32_t>(device_id), std::move(config_data.name_entries));
        return { agent_config_status::configured, *config };
    }

    return { agent_config_status::failed, {} };
}

bool
configure_agent_spm_configs(client_data& data, const request& req)
{
    if(data.gpu_agents.empty())
    {
        LOG_WARNING("SPM runtime collection requested but no GPU agents are available");
        return false;
    }

    const auto all_requested = parse_requested_counters(req);

    return data.agent_spm_counter_configs.wlock([&](auto& configs) {
        configs.clear();
        auto matched_agent  = false;
        auto skipped_agents = std::size_t{ 0 };
        for(const auto& agent : data.gpu_agents)
        {
            if(agent.agent == nullptr) continue;
            const auto device_id = agent.device_id;
            const auto requested =
                requested_counters_for_device(all_requested, device_id);
            if(requested.empty())
            {
                LOG_DEBUG("No SPM counters requested for device {}", device_id);
                continue;
            }

            auto config = create_agent_spm_config(
                rocprofiler_agent_id_t{ agent.agent->handle }, device_id, req, requested);
            if(config.status == agent_config_status::skipped)
            {
                ++skipped_agents;
                continue;
            }
            if(config.status == agent_config_status::failed) return false;

            configs.emplace(rocprofiler_agent_id_t{ agent.agent->handle }, config.config);
            matched_agent = true;
        }

        if(!matched_agent)
        {
            LOG_WARNING("SPM runtime collection requested but no GPU agent matched the "
                        "requested counters and device filters");
        }
        else if(skipped_agents > 0)
        {
            LOG_WARNING("SPM configured on {} GPU agent(s); skipped {} requested GPU "
                        "agent(s) without SPM support for this request",
                        configs.size(), skipped_agents);
        }
        else
        {
            LOG_INFO("SPM configured on {} GPU agent(s)", configs.size());
        }
        return matched_agent;
    });
}

std::optional<trace_cache::spm_counter_info>
make_counter_info(rocprofiler_counter_instance_id_t instance_id)
{
    auto counter_id = rocprofiler_counter_id_t{};
    auto status     = rocprofiler_query_record_counter_id(instance_id, &counter_id);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Dropping SPM sample: failed to decode counter id from instance "
                    "{}: {} ({})",
                    instance_id, static_cast<int>(status),
                    rocprofiler_get_status_string(status));
        return std::nullopt;
    }

    return trace_cache::spm_counter_info{
        counter_id.handle,
        instance_id,
    };
}

void
spm_dispatch_callback(
    const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
    rocprofiler_counter_config_id_t* config, rocprofiler_user_data_t* user_data,
    void* callback_data_args)
{
    if(config) *config = {};
    if(user_data) *user_data = {};
    if(dispatch_data == nullptr || config == nullptr || callback_data_args == nullptr)
        return;

    const auto* data = static_cast<const client_data*>(callback_data_args);

    // The SDK asks this callback for the SPM config for each dispatch. Leaving
    // *config zeroed is the SDK opt-out path; a nonzero handle must remain valid,
    // so the per-agent config map is populated during setup and only read here.
    data->agent_spm_counter_configs.rlock([&](const auto& configs) {
        if(const auto itr = configs.find(dispatch_data->dispatch_info.agent_id);
           itr != configs.end())
            *config = itr->second;
    });
}

void
spm_record_callback(const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
                    const rocprofiler_spm_counter_record_t** records, size_t record_count,
                    rocprofiler_spm_record_flag_t flags,
                    rocprofiler_user_data_t /*userdata*/, void* record_callback_args)
{
    if(dispatch_data == nullptr) return;

    // The SDK sends a dispatch-complete notification with no SPM records.
    // rocprofv3 handles this before data/data-loss flags, so keep the same
    // ordering here.
    if((flags & ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END) != 0) return;

    if(records == nullptr) return;
    if(record_count == 0) return;

    const auto data_loss = ((flags & ROCPROFILER_SPM_RECORD_FLAG_DATA_LOSS) != 0);
    if(data_loss)
    {
        auto* data = static_cast<client_data*>(record_callback_args);
        if(data != nullptr)
            data->spm_data_loss_reports.fetch_add(1, std::memory_order_relaxed);
    }

    if((flags & ROCPROFILER_SPM_RECORD_FLAG_DATA) == 0) return;

    auto counters             = std::vector<trace_cache::spm_counter_info>{};
    auto samples              = std::vector<trace_cache::spm_timestamp_sample>{};
    auto counter_info_indices = std::unordered_map<std::uint64_t, std::uint32_t>{};
    auto sample_indices       = std::unordered_map<std::uint64_t, size_t>{};
    samples.reserve(record_count);
    for(size_t i = 0; i < record_count; ++i)
    {
        if(records[i] == nullptr) continue;

        const auto* record                   = records[i];
        auto [counter_itr, inserted_counter] = counter_info_indices.emplace(
            record->id, static_cast<std::uint32_t>(counters.size()));
        if(inserted_counter)
        {
            auto counter_info = make_counter_info(record->id);
            if(!counter_info)
            {
                counter_info_indices.erase(counter_itr);
                continue;
            }
            counters.emplace_back(*counter_info);
        }

        auto [sample_itr, inserted_sample] =
            sample_indices.emplace(record->timestamp, samples.size());
        if(inserted_sample)
            samples.emplace_back(
                trace_cache::spm_timestamp_sample{ record->timestamp, {} });

        samples[sample_itr->second].values.emplace_back(
            trace_cache::spm_counter_value{ counter_itr->second, record->value });
    }
    if(samples.empty()) return;

    const auto& info = dispatch_data->dispatch_info;
    trace_cache::get_buffer_storage().store(trace_cache::spm_sample{
        info.agent_id.handle,
        info.dispatch_id,
        info.kernel_id,
        info.queue_id.handle,
        dispatch_data->correlation_id.internal,
        dispatch_data->correlation_id.external.value,
        0,  // TODO: wire HIP stream correlation for SPM dispatch callbacks.
        data_loss,
        std::move(counters),
        std::move(samples),
    });
}
}  // namespace

bool
configure_runtime(client_data* data, const request& req)
{
    if(!req.requested()) return true;

    if(data == nullptr)
    {
        LOG_WARNING("SPM runtime collection requested but client data is unavailable");
        return false;
    }

    if(!configure_agent_spm_configs(*data, req))
    {
        return false;
    }

    if(data->spm_ctx.handle == invalid_context_handle)
    {
        auto status = rocprofiler_create_context(&data->spm_ctx);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Failed to create SPM context: {} ({})", static_cast<int>(status),
                        rocprofiler_get_status_string(status));
            return false;
        }
    }

    auto status = rocprofiler_spm_configure_callback_dispatch_service(
        data->spm_ctx, spm_dispatch_callback, data, spm_record_callback, data);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to configure SPM callback dispatch service: {} ({})",
                    static_cast<int>(status), rocprofiler_get_status_string(status));
        return false;
    }

    LOG_DEBUG("Configured SPM callback dispatch service on spm_ctx={}",
              data->spm_ctx.handle);
    return true;
}

void
report_runtime_summary(client_data* data)
{
    if(data == nullptr) return;

    const auto data_loss_reports =
        data->spm_data_loss_reports.exchange(0, std::memory_order_relaxed);
    if(data_loss_reports > 0)
    {
        LOG_WARNING("SPM data loss was reported in {} callback batch(es)",
                    data_loss_reports);
    }
}
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
