// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Keep SPM request handling, validation, SDK runtime wiring, callbacks, and
// no-SDK runtime fallbacks here.

#include "library/rocprofiler-sdk/spm.hpp"

#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
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
//     <rocprofiler-sdk/experimental/spm.h>. Also used by fwd.hpp and
//     perfetto_processor.cpp for SPM-dependent data members/output.
//
//   ROCPROFSYS_DISABLE_SPM_RUNTIME
//     Defined only by the unit-test target so the SPM request/validation helpers
//     can be compiled and tested without pulling in SDK SPM runtime dependencies.
//
// ROCPROFSYS_COMPILE_SPM_RUNTIME resolves those inputs into the single local
// condition this file branches on. Keeping this derived gate avoids repeating the
// compound condition at each #if site.
#if ROCPROFSYS_USE_SPM && !defined(ROCPROFSYS_DISABLE_SPM_RUNTIME)
#    define ROCPROFSYS_COMPILE_SPM_RUNTIME 1
#else
#    define ROCPROFSYS_COMPILE_SPM_RUNTIME 0
#endif

#if ROCPROFSYS_COMPILE_SPM_RUNTIME
#    include "backends/rocprofiler_sdk/backend.hpp"
#    include "backends/rocprofiler_sdk/wrapper.hpp"
#    include "core/trace_cache/cache_manager.hpp"
#    include "core/trace_cache/sample_type.hpp"
#    include "core/utility.hpp"
#    include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#    include "library/rocprofiler-sdk/fwd.hpp"

#    include <rocprofiler-sdk/context.h>
#    include <rocprofiler-sdk/experimental/spm.h>
#    include <rocprofiler-sdk/rocprofiler.h>

#    include <array>
#    include <atomic>
#    include <functional>
#    include <numeric>
#    include <unordered_map>
#endif

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace spm
{
bool
request::requested() const noexcept
{
    return !events.empty();
}

bool
is_config_valid(const request&                  req,
                const std::vector<std::string>& dispatch_counter_events,
                const std::string&              gpu_perf_counter_events)
{
    // Backstop for direct library load paths. Tool initialization must reject
    // SPM requests when the required interval or mutual-exclusion constraints
    // are not satisfied.
    if(!req.requested())
    {
        if(req.sample_interval > 0)
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
    if(std::any_of(gpu_perf_counter_events.begin(), gpu_perf_counter_events.end(),
                   [](unsigned char itr) { return std::isspace(itr) == 0; }))
    {
        LOG_ERROR("Invalid SPM configuration: SPM counter collection is mutually "
                  "exclusive with ROCPROFSYS_GPU_PERF_COUNTERS");
        return false;
    }

    if(req.sample_interval == 0)
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
constexpr auto device_qualifier = std::string_view{ ":device=" };
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
    auto name = std::string{ event.substr(0, event.find(device_qualifier)) };
    utility::trim_str(name);
    return name;
}

requested_counter_vec_t
parse_requested_counters(const request& req)
{
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

        if(name.empty() || !device.has_value())
        {
            LOG_WARNING("Invalid SPM device qualifier '{}'. Expected COUNTER:device=N",
                        event);
            continue;
        }

        counters.push_back({ std::move(name), device.value() });
    }
    return counters;
}

requested_counter_vec_t
requested_counters_for_device(const requested_counter_vec_t& all_requested,
                              std::uint64_t                  device_id)
{
    auto requested = requested_counter_vec_t{};
    requested.reserve(all_requested.size());
    std::copy_if(all_requested.begin(), all_requested.end(),
                 std::back_inserter(requested), [device_id](const auto& itr) {
                     return !itr.device_id.has_value() ||
                            itr.device_id.value() == device_id;
                 });
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
}  // namespace detail

#if ROCPROFSYS_COMPILE_SPM_RUNTIME
namespace
{
constexpr auto invalid_context_handle = 0UL;

struct resolved_counter
{
    rocprofiler_counter_id_t                                    id           = {};
    std::vector<trace_cache::info::gpu_perf_counter_name_entry> name_entries = {};
};

using spm_available_config_vec_t = std::vector<rocprofiler_spm_available_configuration_t>;
using spm_counter_id_vec_t       = std::vector<rocprofiler_counter_id_t>;
using resolved_counter_vec_t     = std::vector<resolved_counter>;
using requested_counter_vec_t    = detail::requested_counter_vec_t;
namespace gpu_perf_counter       = pmc::collectors::gpu_perf_counter;
using counter_detail_backend     = ::rocprofsys::backends::rocprofiler_sdk::backend<
        ::rocprofsys::rocprofiler_sdk::backend>;

enum class spm_status
{
    success,
    skipped,
    failed,
};

struct counter_query_result
{
    spm_status             status   = spm_status::failed;
    resolved_counter_vec_t counters = {};
};

struct agent_config_result
{
    spm_status                      status = spm_status::failed;
    rocprofiler_counter_config_id_t config = {};
};

struct counter_config_data
{
    spm_counter_id_vec_t                                        counter_ids  = {};
    std::vector<trace_cache::info::gpu_perf_counter_name_entry> name_entries = {};
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

std::vector<trace_cache::info::gpu_perf_counter_name_entry>
spm_counter_name_entries(
    const std::vector<gpu_perf_counter::counter_metadata>& counter_details,
    std::uint64_t                                          device_id)
{
    auto entries = std::vector<trace_cache::info::gpu_perf_counter_name_entry>{};
    entries.reserve(counter_details.size());

    for(const auto& metadata : counter_details)
    {
        auto counter_name = gpu_perf_counter::make_qualified_name(metadata);
        auto track_name   = fmt::format("GPU SPM {} [{}]", counter_name, device_id);
        entries.push_back(
            { metadata.counter_id, std::move(counter_name), std::move(track_name) });
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
        return { spm_status::skipped, {} };
    }

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        LOG_WARNING("Failed to query SPM counters for agent {}: {} ({})", agent_id.handle,
                    static_cast<int>(status), rocprofiler_get_status_string(status));
        return { spm_status::failed, {} };
    }

    auto counters = resolved_counter_vec_t{};
    auto matched  = std::unordered_set<std::string>{};
    for(const auto& counter : supported)
    {
        auto details = counter_detail_backend::query_counter_details(counter);
        if(details.empty()) continue;

        auto name = details.front().name;
        if(requested_names.count(name) > 0)
        {
            auto name_entries = spm_counter_name_entries(details, device_id);
            counters.push_back({ counter, std::move(name_entries) });
            matched.emplace(std::move(name));
        }
    }

    if(matched.size() == requested_names.size())
        return { spm_status::success, std::move(counters) };

    for(const auto& name : requested_names)
    {
        if(matched.count(name) == 0)
        {
            LOG_WARNING("Requested SPM counter '{}' is not supported for device {} "
                        "(agent {})",
                        name, device_id, agent_id.handle);
        }
    }

    return { spm_status::skipped, {} };
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
        log_sdk_status_failure(
            fmt::format("Failed to create SPM counter config for device {} (agent {})",
                        device_id, agent_id.handle),
            status);
        return std::nullopt;
    }

    return config;
}

agent_config_result
create_agent_spm_config(rocprofiler_agent_id_t agent_id, std::uint64_t device_id,
                        const request& req, const requested_counter_vec_t& requested)
{
    const auto requested_names = detail::requested_counter_names(requested);
    auto counters = query_supported_spm_counters(agent_id, requested_names, device_id);
    if(counters.status != spm_status::success) return { counters.status, {} };

    if(!sample_interval_supported(agent_id, req))
    {
        LOG_WARNING("SPM sample interval {} SCLK cycles is not supported for device {} "
                    "(agent {})",
                    req.sample_interval, device_id, agent_id.handle);
        return { spm_status::failed, {} };
    }

    auto config_data = make_counter_config_data(counters.counters);
    auto config =
        create_sdk_spm_counter_config(agent_id, device_id, req, config_data.counter_ids);
    if(config)
    {
        trace_cache::get_metadata_registry().set_spm_counter_names(
            static_cast<std::uint32_t>(device_id), std::move(config_data.name_entries));
        return { spm_status::success, *config };
    }

    return { spm_status::failed, {} };
}

bool
configure_agent_spm_configs(client_data& data, const request& req)
{
    if(data.gpu_agents.empty())
    {
        LOG_WARNING("SPM runtime collection requested but no GPU agents are available");
        return false;
    }

    const auto all_requested = detail::parse_requested_counters(req);

    return data.agent_spm_counter_configs.wlock([&](auto& configs) {
        configs.clear();
        auto matched_agent  = false;
        auto skipped_agents = std::size_t{ 0 };
        for(const auto& agent : data.gpu_agents)
        {
            if(agent.agent == nullptr) continue;
            const auto device_id = agent.device_id;
            const auto requested =
                detail::requested_counters_for_device(all_requested, device_id);
            if(requested.empty())
            {
                LOG_DEBUG("No SPM counters requested for device {}", device_id);
                continue;
            }

            auto config = create_agent_spm_config(
                rocprofiler_agent_id_t{ agent.agent->handle }, device_id, req, requested);
            if(config.status == spm_status::skipped)
            {
                ++skipped_agents;
                continue;
            }
            if(config.status == spm_status::failed) return false;

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

    // Dispatch-complete notifications do not carry SPM records, even if other
    // flags are set. Handle them before record/data-loss processing.
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
    counters.reserve(record_count);
    samples.reserve(record_count);
    counter_info_indices.reserve(record_count);
    sample_indices.reserve(record_count);
    for(const auto* record :
        std::span<const rocprofiler_spm_counter_record_t*>{ records, record_count })
    {
        if(record == nullptr) continue;
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
        log_sdk_status_failure("Failed to configure SPM callback dispatch service",
                               status);
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

void
log_data_loss(client_data* data)
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
#else
bool
configure_runtime(client_data*, const request&)
{
    LOG_WARNING("SPM runtime collection was requested, but this rocprofiler-sdk "
                "build does not provide the experimental SPM API. Build with a "
                "rocprofiler-sdk version that provides "
                "rocprofiler-sdk/experimental/spm.h.");
    return false;
}

void
log_data_loss(client_data*)
{}
#endif
}  // namespace spm
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
