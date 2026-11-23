// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/agent.hpp"
#include "library/pmc/collectors/gpu_perf_counter/derived_expression.hpp"
#include "library/pmc/collectors/gpu_perf_counter/expression_tree.hpp"
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

/// @brief The counters a device samples, as the device provider resolved them.
struct device_counters
{
    /// Counters the user requested; every instance is reported in each sample.
    std::vector<counter_metadata> requested;
    /// Counters that are read only to evaluate the interval derived counters in
    /// `requested`.
    std::vector<counter_metadata> inputs;
    /// The parsed expression of each derived counter in `requested`, by counter name.
    std::unordered_map<std::string, derived_expression> expressions;
    /// The slot each input counter's reduction has in `expressions`.
    input_slots slots;
};

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
        build_record_roles(counters);
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
            m_result_cache.push_back(
                { counter.counter_id, counter.expression.evaluate(m_input_reductions) });
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
        counter_id_t       counter_id{ 0 };
        derived_expression expression;  // reads m_input_reductions by slot
    };

    void build_record_roles(const device_counters& counters)
    {
        for(const auto& meta : m_counter_meta)
        {
            add_requested(meta, counters.expressions);
        }
        add_inputs(m_counter_meta, counters.slots);
        add_inputs(counters.inputs, counters.slots);
        m_input_reductions.resize(counters.slots.size());
    }

    void add_requested(
        const counter_metadata&                                    meta,
        const std::unordered_map<std::string, derived_expression>& expressions)
    {
        if(!meta.is_derived)
        {
            m_record_roles[meta.counter_id] = { .is_constant = meta.is_constant,
                                                .is_reported = true };
            return;
        }

        auto const itr = expressions.find(meta.name);
        if(itr == expressions.end())
        {
            throw std::logic_error(
                fmt::format("GPU {}: derived counter {} has no parsed expression",
                            m_agent->device_type_index, meta.name));
        }
        switch(itr->second.evaluation)
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
                                static_cast<int>(itr->second.evaluation)));
        }

        // The SDK value of an interval counter is replaced by the evaluated one.
        m_record_roles[meta.counter_id] = {};
        m_interval_counters.push_back(
            { .counter_id = meta.counter_id, .expression = itr->second });
    }

    // Gives every instance of a counter that an expression reads its input slot.
    void add_inputs(const std::vector<counter_metadata>& candidates,
                    const input_slots&                   slots)
    {
        for(const auto& meta : candidates)
        {
            auto const slot = slots.find(meta.name);
            if(slot != slots.end() && !meta.is_derived)
            {
                auto& role       = m_record_roles[meta.counter_id];
                role.is_constant = meta.is_constant;
                role.input_slot  = slot->second;
            }
        }
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
