// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Keep SPM configuration handling, validation, SDK runtime wiring, callbacks, and
// no-SDK runtime fallbacks here.

#include "library/rocprofiler-sdk/spm.hpp"
#include "library/rocprofiler-sdk/spm_internal.hpp"

#include "common/delimit.hpp"
#include "common/env_vars.hpp"
#include "core/config.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

// SPM runtime compilation is controlled by two inputs:
//
//   ROCPROFSYS_USE_SPM
//     Set to 0/1 by cmake/Packages.cmake after probing for
//     <rocprofiler-sdk/experimental/spm.h>. This file is the only consumer that
//     branches on it; fwd.hpp deliberately keeps SPM handle storage
//     macro-independent so client_data has one layout in every translation unit.
//
//   ROCPROFSYS_DISABLE_SPM_RUNTIME
//     Defined only by the unit-test target so the SPM configuration/validation helpers
//     can be compiled and tested without pulling in SDK SPM runtime dependencies.

#if ROCPROFSYS_USE_SPM && !defined(ROCPROFSYS_DISABLE_SPM_RUNTIME)
#    include "backends/rocprofiler_sdk/backend.hpp"
#    include "backends/rocprofiler_sdk/wrapper.hpp"
#    include "core/trace_cache/cache_manager.hpp"
#    include "core/trace_cache/metadata_registry.hpp"
#    include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#    include "library/rocprofiler-sdk/fwd.hpp"
#    include "library/rocprofiler-sdk/spm_sample.hpp"

#    include <rocprofiler-sdk/context.h>
#    include <rocprofiler-sdk/counters.h>
#    include <rocprofiler-sdk/experimental/spm.h>
#    include <rocprofiler-sdk/fwd.h>
#    include <rocprofiler-sdk/rocprofiler.h>
#    include <spdlog/fmt/fmt.h>  // NOLINT(misc-include-cleaner): public fmt facade

#    include <array>
#    include <atomic>
#    include <cstddef>
#    include <exception>
#    include <numeric>
#    include <span>
#    include <unordered_map>
#endif

namespace rocprofsys::rocprofiler_sdk::spm
{
bool
configuration::requested() const noexcept
{
    return !counter_events.empty();
}

std::vector<std::string>
get_events()
{
    return rocprofsys::delimit(
        get_setting_value<std::string>(std::string{ env_vars::ROCM_SPM_EVENTS })
            .value_or(std::string{}),
        " ,;\t\n");
}

std::uint64_t
get_sample_interval()
{
    return get_setting_value<std::uint64_t>(
               std::string{ env_vars::ROCM_SPM_SAMPLE_INTERVAL })
        .value_or(0);
}

bool
is_config_valid(const configuration&            requested_config,
                const std::vector<std::string>& dispatch_counter_events,
                const std::string&              gpu_perf_counter_events)
{
    // Backstop for direct library load paths. Tool initialization must reject
    // SPM requests when the required interval or mutual-exclusion constraints
    // are not satisfied.
    if(!requested_config.requested())
    {
        if(requested_config.sample_interval > 0)
        {
            LOG_WARNING("ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL is set but "
                        "ROCPROFSYS_ROCM_SPM_EVENTS is empty; no SPM counters will be "
                        "collected. Set ROCPROFSYS_ROCM_SPM_EVENTS or pass --spm-events "
                        "to request SPM collection.");
        }
        return true;
    }

    if(!dispatch_counter_events.empty())
    {
        LOG_ERROR("Invalid SPM configuration: SPM counter collection is mutually "
                  "exclusive with ROCPROFSYS_ROCM_EVENTS");
        return false;
    }

    // ROCPROFSYS_GPU_PERF_COUNTERS is kept as a raw setting string here. Treat
    // any non-whitespace value as a requested device-counting collection.
    if(std::ranges::any_of(gpu_perf_counter_events, [](unsigned char character) {
           return std::isspace(character) == 0;
       }))
    {
        LOG_ERROR("Invalid SPM configuration: SPM counter collection is mutually "
                  "exclusive with ROCPROFSYS_GPU_PERF_COUNTERS");
        return false;
    }

    if(requested_config.sample_interval == 0)
    {
        LOG_ERROR("Invalid SPM configuration: SPM counter collection requires a "
                  "positive sample interval. Set ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL or "
                  "pass --spm-sample-interval (for example, 8192). Supported intervals "
                  "are hardware-limited and can be queried with 'rocprofv3-avail info "
                  "--spm-config'.");
        return false;
    }

    return true;
}

namespace detail
{
namespace
{
constexpr auto k_device_qualifier = std::string_view{ ":device=" };
}

std::optional<std::uint64_t>
parse_device_id(std::string_view value)
{
    std::uint64_t result = 0;
    if(value.empty())
    {
        return std::nullopt;
    }

    const auto* first  = value.data();
    const auto* last   = value.data() + value.size();
    const auto  parsed = std::from_chars(first, last, result);
    if(parsed.ec != std::errc{} || parsed.ptr != last)
    {
        return std::nullopt;
    }
    return result;
}

std::string
parse_counter_name(std::string_view event)
{
    auto name = std::string{ event.substr(0, event.find(k_device_qualifier)) };
    utility::trim_str(name);
    return name;
}

requested_counter_vec_t
parse_requested_counters(const configuration& requested_config)
{
    auto counters = requested_counter_vec_t{};
    counters.reserve(requested_config.counter_events.size());
    for(const auto& event : requested_config.counter_events)
    {
        auto trimmed_event = event;
        utility::trim_str(trimmed_event);
        if(trimmed_event.empty())
        {
            continue;
        }

        auto       name = parse_counter_name(trimmed_event);
        const auto pos  = trimmed_event.find(k_device_qualifier);
        if(pos == std::string::npos)
        {
            counters.push_back({ .name = std::move(name), .device_id = std::nullopt });
            continue;
        }

        auto device = parse_device_id(
            std::string_view{ trimmed_event }.substr(pos + k_device_qualifier.size()));

        if(name.empty() || !device.has_value())
        {
            LOG_WARNING("Invalid SPM device qualifier '{}'. Expected COUNTER:device=N",
                        event);
            continue;
        }

        counters.push_back({ .name = std::move(name), .device_id = device });
    }
    return counters;
}

requested_counter_vec_t
requested_counters_for_device(const requested_counter_vec_t& all_requested,
                              std::uint64_t                  device_id)
{
    auto requested = requested_counter_vec_t{};
    requested.reserve(all_requested.size());
    std::ranges::copy_if(
        all_requested, std::back_inserter(requested), [device_id](const auto& itr) {
            return !itr.device_id.has_value() || itr.device_id.value() == device_id;
        });
    return requested;
}

std::unordered_set<std::string>
requested_counter_names(const requested_counter_vec_t& requested)
{
    auto requested_names = std::unordered_set<std::string>{};
    for(const auto& itr : requested)
    {
        requested_names.emplace(itr.name);
    }
    return requested_names;
}
}  // namespace detail

#if ROCPROFSYS_USE_SPM && !defined(ROCPROFSYS_DISABLE_SPM_RUNTIME)
namespace
{
constexpr auto k_invalid_context_handle = 0UL;
static_assert(sizeof(rocprofiler_counter_config_id_t) == sizeof(spm_counter_config_id_t),
              "SPM config handle mirror diverged from rocprofiler-sdk");

template <typename FunctionT>
void
log_noexcept(FunctionT&& log_operation) noexcept
{
    try
    {
        std::forward<FunctionT>(log_operation)();
    } catch(...)
    {
        return;
    }
}

/// One requested SPM counter, resolved against an agent.
///
/// `base_counter_id` is the base counter id that
/// `rocprofiler_spm_create_counter_config()` requires. `name_entries` holds one entry
/// per dimension instance, each keyed by its own instance id, used to resolve samples
/// back to pmc_info/track names. The two id spaces are distinct: do not derive one
/// from the other.
struct resolved_counter
{
    rocprofiler_counter_id_t                                    base_counter_id = {};
    std::vector<trace_cache::info::gpu_perf_counter_name_entry> name_entries;
};

using spm_available_config_vec_t = std::vector<rocprofiler_spm_available_configuration_t>;
using spm_counter_id_vec_t       = std::vector<rocprofiler_counter_id_t>;
using resolved_counter_vec_t     = std::vector<resolved_counter>;
using requested_counter_vec_t    = detail::requested_counter_vec_t;
using counter_info_index_map_t   = std::unordered_map<std::uint64_t, std::uint32_t>;
using sample_index_map_t         = std::unordered_map<std::uint64_t, size_t>;
using counter_detail_backend     = ::rocprofsys::backends::rocprofiler_sdk::backend<
        ::rocprofsys::rocprofiler_sdk::wrapper>;

enum class SpmStatus
{
    Success,
    Skipped,
    Failed,
};

struct counter_query_result
{
    SpmStatus              status = SpmStatus::Failed;
    resolved_counter_vec_t resolved_counters;
};

/// Flattened per-agent aggregate of `resolved_counter_vec_t`, consumed by SDK
/// counter config creation and the metadata registry.
struct counter_config_data
{
    spm_counter_id_vec_t                                        base_counter_ids;
    std::vector<trace_cache::info::gpu_perf_counter_name_entry> name_entries;
};

void
log_sdk_status_failure(std::string_view message, rocprofiler_status_t status)
{
    if(status == ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED)
    {
        LOG_WARNING("{}: {} ({}). Set ROCPROFILER_SPM_BETA_ENABLED=ON to acknowledge "
                    "the beta risk and enable SDK SPM collection",
                    message, static_cast<int>(status),
                    rocprofiler_get_status_string(status));
        return;
    }

    LOG_WARNING("{}: {} ({})", message, static_cast<int>(status),
                rocprofiler_get_status_string(status));
}

void
destroy_spm_counter_config(rocprofiler_agent_id_t          agent_id,
                           rocprofiler_counter_config_id_t config_id) noexcept
{
    if(config_id.handle == k_invalid_context_handle)
    {
        return;
    }

    const auto status = rocprofiler_spm_destroy_counter_config(config_id);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        log_noexcept([&]() {
            LOG_WARNING("Failed to destroy SPM counter config for agent {}: {} ({})",
                        agent_id.handle, static_cast<int>(status),
                        rocprofiler_get_status_string(status));
        });
    }
}

class counter_config_owner
{
public:
    counter_config_owner() = default;

    counter_config_owner(rocprofiler_agent_id_t          agent_id,
                         rocprofiler_counter_config_id_t config_id) noexcept
    : m_agent_id{ agent_id }
    , m_config_id{ config_id }
    {}

    ~counter_config_owner() noexcept { reset(); }

    counter_config_owner(const counter_config_owner&)            = delete;
    counter_config_owner& operator=(const counter_config_owner&) = delete;

    counter_config_owner(counter_config_owner&& rhs) noexcept
    : m_agent_id{ rhs.m_agent_id }
    , m_config_id{ rhs.release() }
    {}

    counter_config_owner& operator=(counter_config_owner&& rhs) noexcept
    {
        if(this == &rhs)
        {
            return *this;
        }

        reset();
        m_agent_id  = rhs.m_agent_id;
        m_config_id = rhs.release();
        return *this;
    }

    [[nodiscard]] rocprofiler_counter_config_id_t get() const noexcept
    {
        return m_config_id;
    }

    [[nodiscard]] rocprofiler_counter_config_id_t release() noexcept
    {
        return std::exchange(m_config_id, rocprofiler_counter_config_id_t{});
    }

private:
    void reset() noexcept
    {
        destroy_spm_counter_config(m_agent_id, m_config_id);
        m_config_id = {};
    }

    rocprofiler_agent_id_t          m_agent_id  = {};
    rocprofiler_counter_config_id_t m_config_id = {};
};

struct agent_config_result
{
    SpmStatus            status = SpmStatus::Failed;
    counter_config_owner config;
};

void
destroy_spm_counter_configs(agent_spm_counter_config_map_t& configs)
{
    for(const auto& [agent_id, config_id] : configs)
    {
        destroy_spm_counter_config(agent_id,
                                   rocprofiler_counter_config_id_t{ config_id.handle });
    }
    configs.clear();
}

void
destroy_spm_counter_configs(client_data& data)
{
    data.agent_spm_counter_configs.wlock(
        [](auto& configs) { destroy_spm_counter_configs(configs); });
}

rocprofiler_status_t
copy_spm_configurations(const rocprofiler_spm_available_configuration_t** configs,
                        size_t num_configs, spm_available_config_vec_t& output)
{
    auto result = spm_available_config_vec_t{};
    result.reserve(num_configs);
    for(size_t index = 0; index < num_configs; ++index)
    {
        if(configs[index] != nullptr)
        {
            result.emplace_back(*configs[index]);
        }
    }
    output.swap(result);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
spm_configurations_callback(const rocprofiler_spm_available_configuration_t** configs,
                            size_t num_configs, void* user_data) noexcept
{
    auto* output = static_cast<spm_available_config_vec_t*>(user_data);
    if(output == nullptr || (configs == nullptr && num_configs > 0))
    {
        return ROCPROFILER_STATUS_ERROR;
    }

    try
    {
        return copy_spm_configurations(configs, num_configs, *output);
    } catch(...)
    {
        output->clear();
        return ROCPROFILER_STATUS_ERROR;
    }
}

rocprofiler_status_t
spm_supported_counters_callback(rocprofiler_agent_id_t /*agent_id*/,
                                rocprofiler_counter_id_t* counters, size_t num_counters,
                                void* user_data) noexcept
{
    auto* output = static_cast<spm_counter_id_vec_t*>(user_data);
    if(output == nullptr || (counters == nullptr && num_counters > 0))
    {
        return ROCPROFILER_STATUS_ERROR;
    }

    try
    {
        auto result = spm_counter_id_vec_t{};
        if(num_counters > 0)
        {
            result.assign(counters, counters + num_counters);
        }
        output->swap(result);
        return ROCPROFILER_STATUS_SUCCESS;
    } catch(...)
    {
        output->clear();
        return ROCPROFILER_STATUS_ERROR;
    }
}

SpmStatus
query_sample_interval_support(rocprofiler_agent_id_t agent_id,
                              const configuration&   requested_config)
{
    auto configs = spm_available_config_vec_t{};
    auto status  = rocprofiler_spm_query_agent_configurations(
        agent_id, spm_configurations_callback, &configs);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to query SPM configurations for agent {}: {} ({})",
                    agent_id.handle, static_cast<int>(status),
                    rocprofiler_get_status_string(status));
        return SpmStatus::Failed;
    }

    const auto supported =
        std::ranges::any_of(configs, [&requested_config](const auto& config) {
            return config.type ==
                       ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES &&
                   config.interval.min_interval <= requested_config.sample_interval &&
                   config.interval.max_interval >= requested_config.sample_interval;
        });
    return supported ? SpmStatus::Success : SpmStatus::Skipped;
}

counter_config_data
make_counter_config_data(resolved_counter_vec_t& counters)
{
    auto config_data = counter_config_data{};
    config_data.base_counter_ids.reserve(counters.size());

    const auto name_entry_count =
        std::accumulate(counters.begin(), counters.end(), std::size_t{ 0 },
                        [](std::size_t total, const auto& counter) {
                            return total + counter.name_entries.size();
                        });
    config_data.name_entries.reserve(name_entry_count);

    for(auto& counter : counters)
    {
        config_data.base_counter_ids.emplace_back(counter.base_counter_id);
        config_data.name_entries.insert(
            config_data.name_entries.end(),
            std::make_move_iterator(counter.name_entries.begin()),
            std::make_move_iterator(counter.name_entries.end()));
    }

    return config_data;
}

std::vector<trace_cache::info::gpu_perf_counter_name_entry>
spm_counter_name_entries(
    const std::vector<pmc::collectors::gpu_perf_counter::counter_metadata>&
                  counter_details,
    std::uint64_t device_id)
{
    auto entries = std::vector<trace_cache::info::gpu_perf_counter_name_entry>{};
    entries.reserve(counter_details.size());

    for(const auto& metadata : counter_details)
    {
        auto counter_name =
            pmc::collectors::gpu_perf_counter::make_qualified_name(metadata);
        // NOLINTNEXTLINE(misc-include-cleaner): provided by spdlog fmt facade
        auto track_name = fmt::format("GPU SPM {} [{}]", counter_name, device_id);
        entries.push_back({ .counter_id    = metadata.counter_id,
                            .pmc_info_name = std::move(counter_name),
                            .track_name    = std::move(track_name) });
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
        return { .status = SpmStatus::Skipped, .resolved_counters = {} };
    }

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to query SPM counters for agent {}: {} ({})", agent_id.handle,
                    static_cast<int>(status), rocprofiler_get_status_string(status));
        return { .status = SpmStatus::Failed, .resolved_counters = {} };
    }

    auto counters = resolved_counter_vec_t{};
    auto matched  = std::unordered_set<std::string>{};
    for(const auto& counter : supported)
    {
        auto details = counter_detail_backend::query_counter_details(counter);
        if(details.empty())
        {
            continue;
        }

        auto name = details.front().name;
        if(requested_names.contains(name))
        {
            auto name_entries = spm_counter_name_entries(details, device_id);
            counters.push_back(
                { .base_counter_id = counter, .name_entries = std::move(name_entries) });
            matched.emplace(std::move(name));
        }
    }

    if(matched.size() == requested_names.size())
    {
        return { .status = SpmStatus::Success, .resolved_counters = std::move(counters) };
    }

    for(const auto& name : requested_names)
    {
        if(!matched.contains(name))
        {
            LOG_WARNING("Requested SPM counter '{}' is not supported for device {} "
                        "(agent {})",
                        name, device_id, agent_id.handle);
        }
    }

    return { .status = SpmStatus::Skipped, .resolved_counters = {} };
}

std::optional<counter_config_owner>
create_sdk_spm_counter_config(rocprofiler_agent_id_t agent_id, std::uint64_t device_id,
                              const configuration&  requested_config,
                              spm_counter_id_vec_t& base_counter_ids)
{
    auto param = rocprofiler_spm_parameters_t{
        .size  = sizeof(rocprofiler_spm_parameters_t),
        .type  = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
        .value = requested_config.sample_interval,
    };
    auto params = std::array<rocprofiler_spm_parameters_t*, 1>{ &param };
    auto config = rocprofiler_counter_config_id_t{};

    LOG_DEBUG("Creating SPM counter config for device {} (agent {}) with {} counters",
              device_id, agent_id.handle, base_counter_ids.size());

    auto status = rocprofiler_spm_create_counter_config(
        agent_id, base_counter_ids.data(), base_counter_ids.size(), params.data(),
        params.size(), &config);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        log_sdk_status_failure(
            // NOLINTNEXTLINE(misc-include-cleaner): provided by spdlog fmt facade
            fmt::format("Failed to create SPM counter config for device {} (agent {})",
                        device_id, agent_id.handle),
            status);
        return std::nullopt;
    }

    return counter_config_owner{ agent_id, config };
}

agent_config_result
create_agent_spm_config(rocprofiler_agent_id_t agent_id, std::uint64_t device_id,
                        const configuration&           requested_config,
                        const requested_counter_vec_t& requested)
{
    const auto requested_names = detail::requested_counter_names(requested);
    auto       counter_query =
        query_supported_spm_counters(agent_id, requested_names, device_id);
    if(counter_query.status != SpmStatus::Success)
    {
        return { .status = counter_query.status, .config = {} };
    }

    const auto interval_status =
        query_sample_interval_support(agent_id, requested_config);
    if(interval_status == SpmStatus::Skipped)
    {
        LOG_WARNING("SPM sample interval {} SCLK cycles is not supported for device {} "
                    "(agent {})",
                    requested_config.sample_interval, device_id, agent_id.handle);
    }
    if(interval_status != SpmStatus::Success)
    {
        return { .status = interval_status, .config = {} };
    }

    auto config_data = make_counter_config_data(counter_query.resolved_counters);
    auto config = create_sdk_spm_counter_config(agent_id, device_id, requested_config,
                                                config_data.base_counter_ids);
    if(config)
    {
        trace_cache::get_metadata_registry().set_spm_counter_names(
            static_cast<std::uint32_t>(device_id), std::move(config_data.name_entries));
        return { .status = SpmStatus::Success, .config = std::move(*config) };
    }

    return { .status = SpmStatus::Failed, .config = {} };
}

void
log_agent_configuration_summary(bool matched_agent, std::size_t skipped_agents,
                                std::size_t configured_agents)
{
    if(!matched_agent)
    {
        LOG_WARNING("SPM runtime collection requested but no GPU agent matched "
                    "the requested counters and device filters");
    }
    else if(skipped_agents > 0)
    {
        LOG_WARNING("SPM configured on {} GPU agent(s); skipped {} requested "
                    "GPU agent(s) without SPM support for this configuration",
                    configured_agents, skipped_agents);
    }
    else
    {
        LOG_INFO("SPM configured on {} GPU agent(s)", configured_agents);
    }
}

enum class AgentSetupStatus
{
    NotRequested,
    Configured,
    Skipped,
    Failed,
};

AgentSetupStatus
configure_agent_spm_config(agent_spm_counter_config_map_t& configs,
                           const tool_agent& agent, const configuration& requested_config,
                           const requested_counter_vec_t& all_requested)
{
    if(agent.agent == nullptr)
    {
        return AgentSetupStatus::NotRequested;
    }

    const auto device_id = agent.device_id;
    const auto requested =
        detail::requested_counters_for_device(all_requested, device_id);
    if(requested.empty())
    {
        LOG_DEBUG("No SPM counters requested for device {}", device_id);
        return AgentSetupStatus::NotRequested;
    }

    auto config = create_agent_spm_config(rocprofiler_agent_id_t{ agent.agent->handle },
                                          device_id, requested_config, requested);
    if(config.status == SpmStatus::Skipped)
    {
        return AgentSetupStatus::Skipped;
    }
    if(config.status == SpmStatus::Failed)
    {
        return AgentSetupStatus::Failed;
    }

    const auto agent_id = rocprofiler_agent_id_t{ agent.agent->handle };
    const auto inserted =
        configs.emplace(agent_id, spm_counter_config_id_t{ config.config.get().handle })
            .second;
    if(!inserted)
    {
        return AgentSetupStatus::Failed;
    }

    static_cast<void>(config.config.release());
    return AgentSetupStatus::Configured;
}

bool
populate_agent_spm_configs_unchecked(agent_spm_counter_config_map_t& configs,
                                     const client_data&              data,
                                     const configuration&            requested_config,
                                     const requested_counter_vec_t&  all_requested)
{
    auto matched_agent  = false;
    auto skipped_agents = std::size_t{ 0 };
    for(const auto& agent : data.gpu_agents)
    {
        const auto setup_status =
            configure_agent_spm_config(configs, agent, requested_config, all_requested);
        if(setup_status == AgentSetupStatus::Failed)
        {
            destroy_spm_counter_configs(configs);
            return false;
        }
        if(setup_status == AgentSetupStatus::Configured)
        {
            matched_agent = true;
        }
        else if(setup_status == AgentSetupStatus::Skipped)
        {
            ++skipped_agents;
        }
    }

    log_agent_configuration_summary(matched_agent, skipped_agents, configs.size());
    return matched_agent;
}

bool
populate_agent_spm_configs(agent_spm_counter_config_map_t& configs,
                           const client_data& data, const configuration& requested_config,
                           const requested_counter_vec_t& all_requested)
{
    try
    {
        return populate_agent_spm_configs_unchecked(configs, data, requested_config,
                                                    all_requested);
    } catch(const std::exception& error)
    {
        destroy_spm_counter_configs(configs);
        log_noexcept([&]() {
            LOG_WARNING("Failed to configure SPM counter configs: {}", error.what());
        });
        return false;
    } catch(...)
    {
        destroy_spm_counter_configs(configs);
        log_noexcept([]() {
            LOG_WARNING("Failed to configure SPM counter configs: unknown error");
        });
        return false;
    }
}

bool
configure_agent_spm_configs(client_data& data, const configuration& requested_config)
{
    if(data.gpu_agents.empty())
    {
        LOG_WARNING("SPM runtime collection requested but no GPU agents are available");
        return false;
    }

    const auto all_requested = detail::parse_requested_counters(requested_config);

    return data.agent_spm_counter_configs.wlock([&](auto& configs) {
        destroy_spm_counter_configs(configs);
        return populate_agent_spm_configs(configs, data, requested_config, all_requested);
    });
}

std::optional<counter_info>
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

    return counter_info{
        .counter_id          = counter_id.handle,
        .counter_instance_id = instance_id,
    };
}

std::optional<std::uint32_t>
get_counter_info_index(const rocprofiler_spm_counter_record_t& record,
                       std::vector<counter_info>&              counters,
                       counter_info_index_map_t&               counter_info_indices)
{
    auto [counter_itr, inserted] = counter_info_indices.emplace(
        record.id, static_cast<std::uint32_t>(counters.size()));
    if(!inserted)
    {
        return counter_itr->second;
    }

    auto info = make_counter_info(record.id);
    if(!info)
    {
        counter_info_indices.erase(counter_itr);
        return std::nullopt;
    }

    counters.emplace_back(*info);
    return counter_itr->second;
}

size_t
get_sample_index(const rocprofiler_spm_counter_record_t& record,
                 std::vector<timestamp_sample>&          samples,
                 sample_index_map_t&                     sample_indices)
{
    auto [sample_itr, inserted] =
        sample_indices.emplace(record.timestamp, samples.size());
    if(inserted)
    {
        samples.emplace_back(timestamp_sample{ .timestamp = record.timestamp });
    }
    return sample_itr->second;
}

// This helper is the exception boundary for the lock-backed dispatch lookup.
// NOLINTBEGIN(readability-function-size)
void
set_spm_dispatch_config(
    const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
    rocprofiler_counter_config_id_t& config, const client_data& data) noexcept
{
    // The SDK asks this callback for the SPM config for each dispatch. Leaving
    // config zeroed is the SDK opt-out path; a nonzero handle must remain valid,
    // so the per-agent config map is populated during setup and only read here.
    try
    {
        data.agent_spm_counter_configs.rlock([&](const auto& configs) {
            if(const auto itr = configs.find(dispatch_data->dispatch_info.agent_id);
               itr != configs.end())
            {
                config = rocprofiler_counter_config_id_t{ itr->second.handle };
            }
        });
    } catch(const std::exception& error)
    {
        log_noexcept([&]() {
            LOG_WARNING("Skipping SPM config lookup for dispatch {}: {}",
                        dispatch_data->dispatch_info.dispatch_id, error.what());
        });
    } catch(...)
    {
        log_noexcept([&]() {
            LOG_WARNING("Skipping SPM config lookup for dispatch {}: unknown error",
                        dispatch_data->dispatch_info.dispatch_id);
        });
    }
}
// NOLINTEND(readability-function-size)

void
spm_dispatch_callback(
    const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
    rocprofiler_counter_config_id_t* config, rocprofiler_user_data_t* user_data,
    void* callback_data_args) noexcept
{
    if(config)
    {
        *config = {};
    }
    if(user_data)
    {
        *user_data = {};
    }
    if(dispatch_data == nullptr || config == nullptr || callback_data_args == nullptr)
    {
        return;
    }

    const auto& data = *static_cast<const client_data*>(callback_data_args);
    set_spm_dispatch_config(dispatch_data, *config, data);
}

void
store_spm_records(const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
                  const rocprofiler_spm_counter_record_t** records, size_t record_count,
                  bool data_loss)
{
    auto counters             = std::vector<counter_info>{};
    auto samples              = std::vector<timestamp_sample>{};
    auto counter_info_indices = counter_info_index_map_t{};
    auto sample_indices       = sample_index_map_t{};
    samples.reserve(record_count);
    sample_indices.reserve(record_count);
    for(const auto* record :
        std::span<const rocprofiler_spm_counter_record_t*>{ records, record_count })
    {
        if(record == nullptr)
        {
            continue;
        }
        const auto counter_info_index =
            get_counter_info_index(*record, counters, counter_info_indices);
        if(!counter_info_index)
        {
            continue;
        }

        const auto sample_index = get_sample_index(*record, samples, sample_indices);
        samples[sample_index].values.emplace_back(counter_value{
            .counter_info_index = *counter_info_index, .value = record->value });
    }
    if(samples.empty())
    {
        return;
    }

    const auto& info = dispatch_data->dispatch_info;
    trace_cache::get_buffer_storage().store(sample{
        .agent_id_handle         = info.agent_id.handle,
        .dispatch_id             = info.dispatch_id,
        .kernel_id               = info.kernel_id,
        .queue_id_handle         = info.queue_id.handle,
        .correlation_id_internal = dispatch_data->correlation_id.internal,
        .correlation_id_ancestor = dispatch_data->correlation_id.external.value,
        // TODO: wire HIP stream correlation for SPM dispatch callbacks.
        .stream_handle = 0,
        .data_loss     = data_loss,
        .counters      = std::move(counters),
        .samples       = std::move(samples),
    });
}

void
store_spm_records_noexcept(
    const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
    const rocprofiler_spm_counter_record_t** records, size_t record_count,
    bool data_loss) noexcept
{
    try
    {
        store_spm_records(dispatch_data, records, record_count, data_loss);
    } catch(const std::exception& error)
    {
        log_noexcept([&]() {
            LOG_WARNING("Dropping SPM record batch of {} record(s): {}", record_count,
                        error.what());
        });
    } catch(...)
    {
        log_noexcept([&]() {
            LOG_WARNING("Dropping SPM record batch of {} record(s): unknown error",
                        record_count);
        });
    }
}

/// SDK entry point for SPM records.
///
/// libaqlprofile invokes this from its own KFD thread, so a C++ exception must not
/// escape: unwinding through those frames is not guaranteed and would terminate the
/// profiled process. Buffered storage throws once it stops running, and the decode
/// buffers below allocate, so the whole body is contained here and a failed batch is
/// dropped instead.
// The parameter list is fixed by rocprofiler_spm_dispatch_counting_record_cb_t.
// NOLINTBEGIN(readability-function-size)
void
spm_record_callback(const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
                    const rocprofiler_spm_counter_record_t** records, size_t record_count,
                    rocprofiler_spm_record_flag_t flags,
                    rocprofiler_user_data_t /*userdata*/,
                    void* record_callback_args) noexcept
{
    if(dispatch_data == nullptr)
    {
        return;
    }

    // Dispatch-complete notifications do not carry SPM records, even if other
    // flags are set. Handle them before record/data-loss processing.
    if((flags & ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END) != 0)
    {
        return;
    }

    // Account for data loss before the record guards below. A loss batch can decode
    // to zero records, which is exactly when loss is most likely, so returning early
    // on an empty batch would silently under-report it.
    const auto data_loss = ((flags & ROCPROFILER_SPM_RECORD_FLAG_DATA_LOSS) != 0);
    if(data_loss)
    {
        auto* data = static_cast<client_data*>(record_callback_args);
        if(data != nullptr)
        {
            data->spm_data_loss_reports.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if(records == nullptr || record_count == 0 ||
       (flags & ROCPROFILER_SPM_RECORD_FLAG_DATA) == 0)
    {
        return;
    }

    store_spm_records_noexcept(dispatch_data, records, record_count, data_loss);
}
// NOLINTEND(readability-function-size)
}  // namespace

bool
configure_requested_runtime(client_data* data, const configuration& requested_config)
{
    if(data == nullptr)
    {
        LOG_WARNING("SPM runtime collection requested but client data is unavailable");
        return false;
    }

    if(!configure_agent_spm_configs(*data, requested_config))
    {
        return false;
    }

    if(data->spm_ctx.handle == k_invalid_context_handle)
    {
        auto status = rocprofiler_create_context(&data->spm_ctx);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            LOG_WARNING("Failed to create SPM context: {} ({})", static_cast<int>(status),
                        rocprofiler_get_status_string(status));
            destroy_spm_counter_configs(*data);
            return false;
        }
    }

    auto status = rocprofiler_spm_configure_callback_dispatch_service(
        data->spm_ctx, spm_dispatch_callback, data, spm_record_callback, data);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        log_sdk_status_failure("Failed to configure SPM callback dispatch service",
                               status);
        destroy_spm_counter_configs(*data);
        return false;
    }

    LOG_WARNING(
        "ROCm SPM counter collection is enabled as a beta feature. Kernel dispatches "
        "are serialized while SPM is active, so timings in the trace will differ from "
        "an uninstrumented run. SPM can also affect system stability and in rare cases "
        "trigger a GPU or system reset. See the SPM section of the Systems Profiler "
        "documentation for supported hardware and driver requirements.");
    LOG_DEBUG("Configured SPM callback dispatch service on spm_ctx={}",
              data->spm_ctx.handle);
    return true;
}

namespace
{
void
log_data_loss(client_data* data) noexcept
{
    if(data == nullptr)
    {
        return;
    }

    const auto data_loss_reports =
        data->spm_data_loss_reports.exchange(0, std::memory_order_relaxed);
    if(data_loss_reports > 0)
    {
        log_noexcept([&]() {
            LOG_WARNING("SPM data loss was reported in {} callback batch(es)",
                        data_loss_reports);
        });
    }
}
}  // namespace

void
finalize_runtime(client_data* data) noexcept
{
    if(data == nullptr)
    {
        return;
    }

    try
    {
        destroy_spm_counter_configs(*data);
    } catch(const std::exception& error)
    {
        log_noexcept([&]() {
            LOG_WARNING("Failed to finalize SPM counter configs: {}", error.what());
        });
    } catch(...)
    {
        log_noexcept([]() {
            LOG_WARNING("Failed to finalize SPM counter configs: unknown error");
        });
    }

    log_data_loss(data);
}
#else
bool
configure_requested_runtime(client_data*, const configuration&)
{
    LOG_WARNING("SPM runtime collection was requested, but this rocprofiler-sdk "
                "build does not provide the experimental SPM API. Build with a "
                "rocprofiler-sdk version that provides "
                "rocprofiler-sdk/experimental/spm.h.");
    return false;
}

void
finalize_runtime(client_data*) noexcept
{}
#endif

namespace
{
bool
configure_validated_runtime(client_data* data, const configuration& requested_config)
{
    if(!requested_config.requested())
    {
        return true;
    }

    if(!configure_requested_runtime(data, requested_config))
    {
        LOG_WARNING("Continuing without SPM counter collection");
    }
    return true;
}
}  // namespace

bool
configure_runtime(client_data* data, const configuration& requested_config,
                  const std::vector<std::string>& dispatch_counter_events,
                  const std::string&              gpu_perf_counter_events)
{
    if(!is_config_valid(requested_config, dispatch_counter_events,
                        gpu_perf_counter_events))
    {
        return false;
    }
    return configure_validated_runtime(data, requested_config);
}

bool
configure_runtime(client_data* data)
{
    const auto requested_config = configuration{
        .counter_events  = get_events(),
        .sample_interval = get_sample_interval(),
    };
    const auto dispatch_counter_events = config::get_rocm_counter_events();
    const auto gpu_perf_counter_events = config::get_gpu_perf_counters();

    if(!is_config_valid(requested_config, dispatch_counter_events,
                        gpu_perf_counter_events))
    {
        return false;
    }

    if(requested_config.requested() && config::get_use_rocpd())
    {
        LOG_WARNING("SPM samples are not written to the RocPD database in this "
                    "release; use Perfetto output for SPM results");
    }

    return configure_validated_runtime(data, requested_config);
}
}  // namespace rocprofsys::rocprofiler_sdk::spm
