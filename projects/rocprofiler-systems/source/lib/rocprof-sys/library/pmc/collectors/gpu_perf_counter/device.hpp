// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/agent.hpp"
#include "library/pmc/collectors/gpu_perf_counter/derived_expression.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include "logger/debug.hpp"

#include "policies/rocprofiler-sdk/gpu_perf_counters/backend.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

template <policies::gpu_perf_counters::backend Backend>
class device
{
public:
    device(std::shared_ptr<Backend> backend, Backend::context_id_t context,
           std::shared_ptr<rocprofsys::agent> agent, device_counters counters)
    : m_backend_api{ std::move(backend) }
    , m_context{ context }
    , m_agent{ std::move(agent) }
    , m_counter_meta{ std::move(counters.requested) }
    {
        // Each counter may produce multiple dimension instances (e.g. per-WGP);
        // the headroom factor gives slack beyond the enumerated metadata count.
        // The minimum size guards against an under-sized buffer on sparse devices.
        m_record_buffer.resize(
            std::max<size_t>((m_counter_meta.size() + counters.inputs.size()) *
                                 k_record_buffer_headroom_factor,
                             k_record_buffer_min_size));
        build_record_roles(counters.inputs);
    }

    [[nodiscard]] bool is_supported() const noexcept { return !m_counter_meta.empty(); }

    [[nodiscard]] size_t get_index() const noexcept { return m_agent->device_type_index; }

    [[nodiscard]] const std::string& get_name() const noexcept { return m_agent->name; }

    [[nodiscard]] const std::string& get_product_name() const noexcept
    {
        return m_agent->product_name;
    }

    [[nodiscard]] const std::string& get_vendor_name() const noexcept
    {
        return m_agent->vendor_name;
    }

    [[nodiscard]] const std::vector<counter_metadata>& get_counter_metadata()
        const noexcept
    {
        return m_counter_meta;
    }

    [[nodiscard]] const metrics& sample_metrics(
        [[maybe_unused]] const enabled_metrics& enabled,
        [[maybe_unused]] std::uint64_t          timestamp)
    {
        m_result_cache.clear();

        // start_context requires HSA to be live. It is called eagerly in start(), but
        // tool_init runs before the application's hsa_init, so the first call may fail.
        // Retry here until it succeeds; once m_context_started is true this is a single
        // branch-predicted branch with no further work.
        if(!m_context_started)
        {
            start();
        }
        if(!m_context_started)
        {
            return m_result_cache;
        }

        auto rec_count = m_record_buffer.size();

        const auto status = m_backend_api->sample_device_counting_service(
            m_context, {}, Backend::flag_none, m_record_buffer.data(), &rec_count);

        if(status == Backend::status_hsa_not_loaded)
        {
            LOG_DEBUG("HSA not loaded for device {} (status={}). Ignoring error.",
                      m_agent->device_type_index, static_cast<int>(status));
            return m_result_cache;
        }

        if(status != Backend::status_success)
        {
            LOG_WARNING("Sample failed for device {} (status={})",
                        m_agent->device_type_index, static_cast<int>(status));
            return m_result_cache;
        }

        // SDK writes back the actual number of records filled; it must not exceed
        // the buffer capacity we provided.
        assert(rec_count <= m_record_buffer.size());

        m_result_cache.reserve(rec_count + m_interval_counters.size());
        std::ranges::fill(m_input_reductions, input_reduction{});
        for(size_t idx = 0; idx < rec_count; ++idx)
        {
            const auto& record = m_record_buffer[idx];

            typename Backend::counter_id_t config_id{};
            m_backend_api->query_record_counter_id(record, &config_id);
            auto&        role  = role_of(config_id.handle);
            const double value = role.sample_value(record.counter_value);
            if(role.is_reported)
            {
                m_result_cache.push_back({ config_id.handle, value });
            }
            if(role.input_slot)
            {
                m_input_reductions[*role.input_slot].add(value);
            }
        }

        for(auto& counter : m_interval_counters)
        {
            for(size_t i = 0; i < counter.slots.size(); ++i)
            {
                counter.inputs[i] = m_input_reductions[counter.slots[i]];
            }
            m_result_cache.push_back(
                { counter.counter_id, counter.expression.evaluate(counter.inputs) });
        }

        return m_result_cache;
    }

    void start()
    {
        if(m_context_started)
        {
            return;
        }

        try
        {
            m_backend_api->start_context(m_context);
            m_context_started = true;
            for(auto& [id, role] : m_record_roles)
            {
                role.previous.reset();
            }
            LOG_DEBUG("GPU PMC context started for device {}.",
                      m_agent->device_type_index);
        } catch(const std::exception& e)
        {
            // HSA may not be initialized yet at this call site. The hsa_init callback
            // registered in tool_init will call start() again once HSA is live.
            LOG_DEBUG("GPU PMC context start deferred for device {} ({}).",
                      m_agent->device_type_index, e.what());
        }
    }

    void stop()
    {
        try
        {
            m_backend_api->stop_context(m_context);
        } catch(const std::exception& e)
        {
            LOG_WARNING("Failed to stop context for device {} ({})",
                        m_agent->device_type_index, e.what());
        }
    }

private:
    // How a record's value enters a sample.
    struct record_role
    {
        bool                  is_constant{ false };  // value as is, not a delta
        bool                  is_reported{ false };  // reported under its own id
        std::optional<size_t> input_slot;            // index into m_input_reductions
        std::optional<double> previous;              // total at the previous sample

        // The SDK reports totals since start_context; a sample is the interval delta.
        [[nodiscard]] double sample_value(double total)
        {
            if(is_constant)
            {
                return total;
            }
            const double delta = previous ? total - *previous : total;
            previous           = total;
            return delta;
        }
    };

    // A requested derived counter that rocprof-sys evaluates on its inputs' interval
    // values; the SDK value of a non-linear expression of totals cannot be differenced.
    struct interval_counter
    {
        counter_id_t                 counter_id{ 0 };
        derived_expression           expression;
        std::vector<size_t>          slots;   // m_input_reductions index per input
        std::vector<input_reduction> inputs;  // per-sample scratch, one per input
    };

    void build_record_roles(const std::vector<counter_metadata>& inputs)
    {
        auto all_counters = m_counter_meta;
        all_counters.insert(all_counters.end(), inputs.begin(), inputs.end());
        auto const definitions = make_counter_definitions(all_counters);

        auto slots_by_name = std::unordered_map<std::string, size_t>{};
        for(const auto& meta : m_counter_meta)
        {
            add_requested(meta, definitions, slots_by_name);
        }
        for(const auto& meta : all_counters)
        {
            auto const slot = slots_by_name.find(meta.name);
            if(slot != slots_by_name.end() && !meta.is_derived)
            {
                auto& role       = m_record_roles[meta.counter_id];
                role.is_constant = meta.is_constant;
                role.input_slot  = slot->second;
            }
        }
        m_input_reductions.resize(slots_by_name.size());
    }

    void add_requested(const counter_metadata&                  meta,
                       const counter_definitions&               definitions,
                       std::unordered_map<std::string, size_t>& slots_by_name)
    {
        if(!meta.is_derived)
        {
            m_record_roles[meta.counter_id] = { .is_constant = meta.is_constant,
                                                .is_reported = true };
            return;
        }

        auto expression = derived_expression{ meta.expression, definitions };
        switch(expression.evaluation())
        {
            case derived_evaluation::constant:
                m_record_roles[meta.counter_id] = { .is_constant = true,
                                                    .is_reported = true };
                return;
            case derived_evaluation::linear:
                m_record_roles[meta.counter_id] = { .is_reported = true };
                return;
            case derived_evaluation::interval: break;
            default:
                throw std::logic_error(
                    fmt::format("derived_evaluation {} not handled",
                                static_cast<int>(expression.evaluation())));
        }

        // The SDK value of an interval counter is replaced by the evaluated one.
        m_record_roles[meta.counter_id] = {};
        auto slots                      = std::vector<size_t>{};
        for(const auto& name : expression.input_names())
        {
            slots.push_back(
                slots_by_name.try_emplace(name, slots_by_name.size()).first->second);
        }
        auto const input_count = slots.size();
        m_interval_counters.push_back(
            { .counter_id = meta.counter_id,
              .expression = std::move(expression),
              .slots      = std::move(slots),
              .inputs     = std::vector<input_reduction>(input_count) });
    }

    [[nodiscard]] record_role& role_of(counter_id_t counter_id)
    {
        auto const itr = m_record_roles.find(counter_id);
        if(itr == m_record_roles.end())
        {
            throw std::logic_error(fmt::format(
                "GPU {}: record of counter {} that is not in the counter config",
                m_agent->device_type_index, counter_id));
        }
        return itr->second;
    }

    static constexpr size_t k_record_buffer_headroom_factor = 2;
    static constexpr size_t k_record_buffer_min_size        = 256;

    std::shared_ptr<Backend>                        m_backend_api;
    Backend::context_id_t                           m_context;
    std::shared_ptr<rocprofsys::agent>              m_agent;
    std::vector<counter_metadata>                   m_counter_meta;
    std::vector<typename Backend::counter_record_t> m_record_buffer;
    metrics                                         m_result_cache;
    std::unordered_map<counter_id_t, record_role>   m_record_roles;
    std::vector<input_reduction>                    m_input_reductions;
    std::vector<interval_counter>                   m_interval_counters;
    bool                                            m_context_started = false;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
