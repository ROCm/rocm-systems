// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/rocprofiler-sdk/spm.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"

#include "logger/debug.hpp"

#if ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
#    include <rocprofiler-sdk/context.h>
#    include <rocprofiler-sdk/experimental/spm.h>
#    include <rocprofiler-sdk/rocprofiler.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
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
#if ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
constexpr auto invalid_context_handle = 0UL;
#endif

void
warn_once(std::atomic<bool>& warned, std::string_view message) noexcept
{
    if(!warned.exchange(true)) LOG_WARNING("{}", message);
}

#if ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
std::string_view
status_name(rocprofiler_status_t status)
{
    return rocprofiler_get_status_string(status);
}

struct spm_dimensions
{
    std::uint32_t xcc           = 0;
    std::uint32_t shader_engine = 0;
    std::uint32_t instance      = 0;
};

struct requested_counter
{
    std::string                  name      = {};
    std::optional<std::uint64_t> device_id = std::nullopt;
};

struct agent_spm_config_result
{
    bool                                           requested = false;
    std::optional<rocprofiler_counter_config_id_t> config    = std::nullopt;
};

using spm_available_config_vec_t = std::vector<rocprofiler_spm_available_configuration_t>;
using spm_counter_id_vec_t       = std::vector<rocprofiler_counter_id_t>;
using requested_counter_vec_t    = std::vector<requested_counter>;
using spm_dimension_cache_t =
    std::unordered_map<rocprofiler_counter_instance_id_t, spm_dimensions>;

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

std::string
trim(std::string_view value)
{
    const auto begin = value.find_first_not_of(" \t\n\r");
    if(begin == std::string_view::npos) return {};

    const auto end = value.find_last_not_of(" \t\n\r");
    return std::string{ value.substr(begin, end - begin + 1) };
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

requested_counter_vec_t
requested_counters(const beta_request& request)
{
    constexpr auto device_qualifier = std::string_view{ ":device=" };

    auto counters = requested_counter_vec_t{};
    counters.reserve(request.events.size());
    for(const auto& event : request.events)
    {
        const auto trimmed_event = trim(event);
        if(trimmed_event.empty()) continue;

        const auto pos = trimmed_event.find(device_qualifier);
        if(pos == std::string::npos)
        {
            counters.push_back({ trimmed_event, std::nullopt });
            continue;
        }

        auto name   = trim(std::string_view{ trimmed_event }.substr(0, pos));
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
sample_interval_supported(rocprofiler_agent_id_t agent_id, const beta_request& request)
{
    auto configs = spm_available_config_vec_t{};
    auto status  = rocprofiler_spm_query_agent_configurations(
        agent_id, spm_configurations_callback, &configs);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to query SPM configurations for agent {}: {} ({})",
                    agent_id.handle, static_cast<int>(status), status_name(status));
        return false;
    }

    return std::any_of(configs.begin(), configs.end(), [&request](const auto& config) {
        return config.type ==
                   ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES &&
               config.interval.min_interval <= request.sample_interval &&
               config.interval.max_interval >= request.sample_interval;
    });
}

agent_spm_config_result
create_agent_spm_config(rocprofiler_agent_id_t agent_id, std::uint64_t device_id,
                        const beta_request& request)
{
    auto requested = requested_counters(request);
    requested.erase(std::remove_if(requested.begin(), requested.end(),
                                   [device_id](const auto& itr) {
                                       return !requested_on_device(itr, device_id);
                                   }),
                    requested.end());
    if(requested.empty())
    {
        LOG_DEBUG("No SPM counters requested for device {}", device_id);
        return {};
    }

    if(!sample_interval_supported(agent_id, request))
    {
        LOG_WARNING("SPM sample interval {} {} is not supported for device {} "
                    "(agent {})",
                    request.sample_interval, request.sample_interval_unit, device_id,
                    agent_id.handle);
        return { true, std::nullopt };
    }

    auto requested_names = std::unordered_set<std::string>{};
    for(const auto& itr : requested)
        requested_names.emplace(itr.name);

    auto supported = spm_counter_id_vec_t{};
    auto status    = rocprofiler_spm_iterate_agent_supported_counters(
        agent_id, spm_supported_counters_callback, &supported);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to query SPM counters for agent {}: {} ({})", agent_id.handle,
                    static_cast<int>(status), status_name(status));
        return { true, std::nullopt };
    }

    auto counters = spm_counter_id_vec_t{};
    auto matched  = std::unordered_set<std::string>{};
    for(const auto& counter : supported)
    {
        auto info = rocprofiler_counter_info_v0_t{};
        status    = rocprofiler_query_counter_info(
            counter, ROCPROFILER_COUNTER_INFO_VERSION_0, &info);
        if(status != ROCPROFILER_STATUS_SUCCESS || info.name == nullptr) continue;

        auto name = std::string{ info.name };
        if(requested_names.count(name) > 0)
        {
            counters.emplace_back(counter);
            matched.emplace(std::move(name));
        }
    }

    if(matched.size() != requested_names.size())
    {
        for(const auto& name : requested_names)
        {
            if(matched.count(name) == 0)
            {
                LOG_WARNING("Requested SPM counter '{}' is not supported for device {} "
                            "(agent {})",
                            name, device_id, agent_id.handle);
            }
        }
        return { true, std::nullopt };
    }

    auto param = rocprofiler_spm_parameters_t{
        sizeof(rocprofiler_spm_parameters_t),
        ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
        request.sample_interval,
    };
    auto params = std::array<rocprofiler_spm_parameters_t*, 1>{ &param };
    auto config = rocprofiler_counter_config_id_t{};

    LOG_DEBUG("Creating SPM counter config for device {} (agent {}) with {} counters",
              device_id, agent_id.handle, requested_names.size());

    status =
        rocprofiler_spm_create_counter_config(agent_id, counters.data(), counters.size(),
                                              params.data(), params.size(), &config);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to create SPM counter config for device {} (agent {}): {} "
                    "({})",
                    device_id, agent_id.handle, static_cast<int>(status),
                    status_name(status));
        return { true, std::nullopt };
    }

    return { true, config };
}

bool
configure_agent_spm_configs(client_data& data, const beta_request& request)
{
    if(data.gpu_agents.empty())
    {
        LOG_WARNING("SPM runtime collection requested but no GPU agents are available");
        return false;
    }

    return data.agent_spm_counter_configs.wlock([&](auto& configs) {
        configs.clear();
        auto matched_agent = false;
        for(const auto& agent : data.gpu_agents)
        {
            if(agent.agent == nullptr) continue;
            const auto device_id     = agent.device_id;
            auto       config_result = create_agent_spm_config(
                rocprofiler_agent_id_t{ agent.agent->handle }, device_id, request);
            if(!config_result.requested) continue;
            if(!config_result.config) return false;

            configs.emplace(rocprofiler_agent_id_t{ agent.agent->handle },
                            *config_result.config);
            matched_agent = true;
        }

        if(!matched_agent)
        {
            LOG_WARNING("SPM runtime collection requested but no GPU agent matched the "
                        "requested counters and device filters");
        }
        return matched_agent;
    });
}

bool
contains_token(std::string_view value, std::string_view token)
{
    return value.find(token) != std::string_view::npos;
}

void
apply_dimension(spm_dimensions& dims, std::string_view name, std::uint32_t value)
{
    if(contains_token(name, "XCC") || contains_token(name, "XCD"))
        dims.xcc = value;
    else if(contains_token(name, "SHADER_ENGINE") || name == "SE")
        dims.shader_engine = value;
    else if(contains_token(name, "INSTANCE"))
        dims.instance = value;
}

spm_dimensions
decode_dimensions(rocprofiler_counter_instance_id_t instance_id,
                  rocprofiler_counter_id_t          counter_id)
{
    static auto cache = common::synchronized<spm_dimension_cache_t>{};
    if(auto cached = cache.rlock([&](const auto& data) -> std::optional<spm_dimensions> {
           auto itr = data.find(instance_id);
           if(itr == data.end()) return std::nullopt;
           return itr->second;
       }))
        return *cached;

    auto info   = rocprofiler_counter_info_v1_t{};
    auto status = rocprofiler_query_counter_info(
        counter_id, ROCPROFILER_COUNTER_INFO_VERSION_1, &info);
    if(status != ROCPROFILER_STATUS_SUCCESS) return {};

    for(std::uint64_t i = 0; i < info.dimensions_instances_count; ++i)
    {
        const auto* dim_instance = info.dimensions_instances[i];
        if(dim_instance == nullptr || dim_instance->instance_id != instance_id) continue;

        auto dims = spm_dimensions{};
        for(std::uint64_t j = 0; j < dim_instance->dimensions_count; ++j)
        {
            const auto* dim = dim_instance->dimensions[j];
            if(dim == nullptr || dim->dimension_name == nullptr) continue;
            apply_dimension(dims, dim->dimension_name,
                            static_cast<std::uint32_t>(dim->index));
        }
        cache.wlock([&](auto& data) { data.emplace(instance_id, dims); });
        return dims;
    }
    return {};
}

trace_cache::spm_counter_sample
make_counter_sample(const rocprofiler_spm_counter_record_t& record)
{
    auto counter_id = rocprofiler_counter_id_t{};
    auto status     = rocprofiler_query_record_counter_id(record.id, &counter_id);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to decode SPM counter id from instance {}: {} ({})",
                    record.id, static_cast<int>(status), status_name(status));
    }

    const auto dims = decode_dimensions(record.id, counter_id);
    return trace_cache::spm_counter_sample{
        record.timestamp, record.value,       counter_id.handle, record.id,
        dims.xcc,         dims.shader_engine, dims.instance,
    };
}

void
spm_dispatch_callback(
    const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
    rocprofiler_counter_config_id_t* config, rocprofiler_user_data_t* user_data,
    void* callback_data_args) noexcept
{
    try
    {
        if(config) *config = {};
        if(user_data) *user_data = {};
        if(dispatch_data == nullptr || config == nullptr || callback_data_args == nullptr)
            return;

        const auto* data = static_cast<const client_data*>(callback_data_args);
        if(data == nullptr) return;

        data->agent_spm_counter_configs.rlock([&](const auto& configs) {
            if(const auto itr = configs.find(dispatch_data->dispatch_info.agent_id);
               itr != configs.end())
                *config = itr->second;
        });
    } catch(const std::exception& e)
    {
        LOG_WARNING("SPM dispatch callback failed: {}", e.what());
    } catch(...)
    {
        LOG_WARNING("SPM dispatch callback failed with an unknown exception");
    }
}

void
spm_record_callback(const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
                    const rocprofiler_spm_counter_record_t** records, size_t record_count,
                    rocprofiler_spm_record_flag_t flags,
                    rocprofiler_user_data_t /*userdata*/,
                    void* /*record_callback_args*/) noexcept
{
    try
    {
        if(dispatch_data == nullptr || records == nullptr) return;

        const auto data_loss = ((flags & ROCPROFILER_SPM_RECORD_FLAG_DATA_LOSS) != 0);
        if(data_loss)
        {
            LOG_WARNING("SPM data loss reported for dispatch {}",
                        dispatch_data->dispatch_info.dispatch_id);
        }

        if((flags & ROCPROFILER_SPM_RECORD_FLAG_DATA) == 0 || record_count == 0) return;

        auto samples = std::vector<trace_cache::spm_counter_sample>{};
        samples.reserve(record_count);
        for(size_t i = 0; i < record_count; ++i)
        {
            if(records[i] == nullptr) continue;
            samples.emplace_back(make_counter_sample(*records[i]));
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
            std::move(samples),
        });
    } catch(const std::exception& e)
    {
        LOG_WARNING("SPM record callback failed: {}", e.what());
    } catch(...)
    {
        LOG_WARNING("SPM record callback failed with an unknown exception");
    }
}
#endif
}  // namespace

bool
configure_runtime(client_data* data, const beta_request& request)
{
    if(!request.requested()) return true;

#if !ROCPROFSYS_HAS_ROCPROFILER_SDK_SPM
    (void) data;
    static auto warned = std::atomic<bool>{ false };
    warn_once(warned, "SPM runtime collection was requested, but this "
                      "rocprofiler-sdk build does not provide the experimental "
                      "SPM API. Build with a rocprofiler-sdk version that "
                      "provides rocprofiler-sdk/experimental/spm.h.");
    return false;
#else
    if(data == nullptr)
    {
        LOG_WARNING("SPM runtime collection requested but client data is unavailable");
        return false;
    }

    if(!configure_agent_spm_configs(*data, request))
    {
        return false;
    }

    if(data->spm_ctx.handle == invalid_context_handle)
    {
        auto status = rocprofiler_create_context(&data->spm_ctx);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Failed to create SPM context: {} ({})", static_cast<int>(status),
                        status_name(status));
            return false;
        }
    }

    auto status = rocprofiler_spm_configure_callback_dispatch_service(
        data->spm_ctx, spm_dispatch_callback, data, spm_record_callback, data);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to configure SPM callback dispatch service: {} ({})",
                    static_cast<int>(status), status_name(status));
        return false;
    }

    LOG_DEBUG("Configured SPM callback dispatch service on spm_ctx={}",
              data->spm_ctx.handle);
    return true;
#endif
}
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
