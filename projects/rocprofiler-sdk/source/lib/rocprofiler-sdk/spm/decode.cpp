// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/spm/core.hpp"
#include "lib/rocprofiler-sdk/spm/decode.hpp"
#include "lib/rocprofiler-sdk/spm/interface.hpp"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace rocprofiler
{
namespace spm
{
std::mutex&
get_buffer_mut()
{
    static auto*& mut = common::static_object<std::mutex>::construct();
    return *CHECK_NOTNULL(mut);
}

void
decode_cb(uint64_t timestamp, uint64_t value, uint64_t index, int shader_engine, void* userdata)
{
    auto& samples   = *reinterpret_cast<spm_sample_vec*>(userdata);
    bool  is_global = (shader_engine < 0);
    if(is_global) shader_engine = 0;
    samples.emplace_back(spm_sample_t{timestamp, value, index, shader_engine, is_global});
}

/**
 * @brief Compute derived metric values from flat SPM samples.
 *
 * Groups samples by timestamp, builds the results_map expected by EvaluateAST,
 * evaluates each derived expression per timestamp group, and appends the
 * resulting derived records to out_records.
 */
void
compute_spm_derived_metrics(const spm_sample_vec&                          samples,
                            spm_desc_v0_t&                                 desc_v0,
                            size_t                                         xcc_id,
                            const spm_counter_config&                      config,
                            rocprofiler_dispatch_id_t                      dispatch_id,
                            rocprofiler_agent_id_t                         agent_id,
                            std::vector<rocprofiler_spm_counter_record_t>& out_records)
{
    if(config.derived.empty() || samples.empty()) return;

    // Group samples by timestamp → {metric_id → vector<rocprofiler_counter_record_t>}
    using record_map_t = std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_t>>;
    auto ts_groups     = std::map<uint64_t, record_map_t>{};

    for(const auto& s : samples)
    {
        auto& event = desc_v0.events()[s.index];

        auto instance_id = rocprofiler_counter_instance_id_t{};
        counters::set_dim_in_rec(
            instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_XCC, xcc_id);
        counters::set_dim_in_rec(
            instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_INSTANCE, event.instance);
        counters::set_counter_in_rec(instance_id, event.id);
        if(!s.is_global)
            counters::set_dim_in_rec(instance_id,
                                     rocprofiler::counters::ROCPROFILER_DIMENSION_SHADER_ENGINE,
                                     s.shader_engine);

        auto counter_id = counters::rec_to_counter_id(instance_id);
        auto metric_id  = counters::get_base_metric_from_counter_id(counter_id);

        ts_groups[s.timestamp][metric_id].push_back(
            rocprofiler_counter_record_t{.id            = instance_id,
                                         .counter_value = static_cast<double>(s.value),
                                         .dispatch_id   = dispatch_id,
                                         .user_data     = {},
                                         .agent_id      = agent_id});
    }

    for(const auto& [derived_metric, ast, req_counters] : config.derived)
    {
        for(const auto& [timestamp, metric_records] : ts_groups)
        {
            // Check all required non-constant base metrics are present
            bool all_present = true;
            for(const auto& base : req_counters)
            {
                if(base.event().empty() && base.expression().empty()) continue;
                if(metric_records.find(base.id()) == metric_records.end())
                {
                    all_present = false;
                    break;
                }
            }
            if(!all_present) continue;

            auto results_map = metric_records;

            if(!config.required_special_counters.empty())
            {
                counters::EvaluateAST::read_special_counters(
                    *config.agent, config.required_special_counters, results_map);
            }

            try
            {
                auto ast_copy = ast;
                auto cache =
                    std::vector<std::unique_ptr<std::vector<rocprofiler_counter_record_t>>>{};
                auto* result = ast_copy.evaluate(results_map, cache);
                if(!result || result->empty()) continue;

                ast_copy.set_out_id(*result);

                for(const auto& rec : *result)
                {
                    auto out_id = rocprofiler_counter_instance_id_t{};
                    counters::set_counter_in_rec(out_id, counters::rec_to_counter_id(rec.id));
                    counters::set_dim_in_rec(
                        out_id, rocprofiler::counters::ROCPROFILER_DIMENSION_XCC, xcc_id);
                    out_records.push_back(rocprofiler_spm_counter_record_t{
                        .size        = sizeof(rocprofiler_spm_counter_record_t),
                        .dispatch_id = dispatch_id,
                        .id          = out_id,
                        .agent_id    = agent_id,
                        .timestamp   = timestamp,
                        .value       = rec.counter_value});
                }
            } catch(std::runtime_error& e)
            {
                ROCP_ERROR << "SPM derived metric evaluation failed for " << derived_metric.name()
                           << ": " << e.what();
                continue;
            }
        }
    }
}

/** @brief  Callback for aqlprofile to return SPM data
 * buffer id - XCC of the data
 * flags - Indicates if there was a data loss
 */
void
aql_data_callback(size_t buffer_id, void* data, size_t data_size, int flags, void* userdata)
{
    auto* spm_packet = static_cast<hsa::SPMPacket*>(userdata);
    if(data_size == 0) return;

    auto& desc_v0 = *static_cast<rocprofiler::spm::spm_desc_v0_t*>(spm_packet->spm_desc.data);
    if(!desc_v0.valid()) return;

    {
        uint64_t count  = 0;
        auto     status = spm_packet->sym->spm_decode_query(
            spm_packet->aql_desc, AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT, &count);
        if(status != HSA_STATUS_SUCCESS) return;
        if(count != desc_v0.num_events) return;
    }

    auto samples = spm_sample_vec{};
    auto status  = spm_packet->sym->spm_decode_stream_v1(
        spm_packet->aql_desc, decode_cb, data, data_size, &samples);
    if(status != HSA_STATUS_SUCCESS) return;

    rocprofiler::buffer::instance* buf = nullptr;
    if(spm_packet->buffer) buf = buffer::get_buffer(spm_packet->buffer->handle);

    auto agent_id    = (rocprofiler::agent::get_rocprofiler_agent(spm_packet->GetAgent()))->id;
    auto dispatch_id = spm_packet->dispatch_data.dispatch_info.dispatch_id;

    // Retrieve config for derived metric evaluation and filtering
    auto* spm_cfg = CHECK_NOTNULL(get_spm_counter_config(spm_packet->config_id).get());

    // Build set of explicitly requested metric IDs for filtering
    auto requested_ids = std::unordered_set<uint64_t>{};
    for(const auto& m : spm_cfg->metrics)
        requested_ids.insert(m.id());

    // Convert flat samples to base SPM counter records, filtering to requested metrics
    auto basic_metrics_samples = std::vector<rocprofiler_spm_counter_record_t>{};
    for(const auto& s : samples)
    {
        auto& event = desc_v0.events()[s.index];

        auto instance_id = rocprofiler_counter_instance_id_t{};
        counters::set_dim_in_rec(
            instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_XCC, buffer_id);
        counters::set_dim_in_rec(
            instance_id, rocprofiler::counters::ROCPROFILER_DIMENSION_INSTANCE, event.instance);
        counters::set_counter_in_rec(instance_id, event.id);
        if(!s.is_global)
            counters::set_dim_in_rec(instance_id,
                                     rocprofiler::counters::ROCPROFILER_DIMENSION_SHADER_ENGINE,
                                     s.shader_engine);

        auto counter_id = counters::rec_to_counter_id(instance_id);
        auto metric_id  = counters::get_base_metric_from_counter_id(counter_id);

        if(requested_ids.count(metric_id) != 0u)
        {
            basic_metrics_samples.emplace_back(
                rocprofiler_spm_counter_record_t{.size = sizeof(rocprofiler_spm_counter_record_t),
                                                 .dispatch_id = dispatch_id,
                                                 .id          = instance_id,
                                                 .agent_id    = agent_id,
                                                 .timestamp   = s.timestamp,
                                                 .value       = static_cast<double>(s.value)});
        }
    }

    // Compute derived metrics from flat samples
    auto derived_metrics_samples = std::vector<rocprofiler_spm_counter_record_t>{};
    if(!spm_cfg->derived.empty())
    {
        compute_spm_derived_metrics(
            samples, desc_v0, buffer_id, *spm_cfg, dispatch_id, agent_id, derived_metrics_samples);
    }

    if(buf)
    {
        auto _lk = std::unique_lock{get_buffer_mut()};

        buf->emplace(ROCPROFILER_BUFFER_CATEGORY_COUNTERS,
                     ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER,
                     spm_packet->dispatch_data);
        for(const auto& itr : basic_metrics_samples)
        {
            buf->emplace(
                ROCPROFILER_BUFFER_CATEGORY_COUNTERS, ROCPROFILER_COUNTER_RECORD_VALUE, itr);
        }
        for(const auto& itr : derived_metrics_samples)
        {
            buf->emplace(
                ROCPROFILER_BUFFER_CATEGORY_COUNTERS, ROCPROFILER_COUNTER_RECORD_VALUE, itr);
        }
    }
    else if(spm_packet->record_cb)
    {
        auto record_ptrs = std::vector<const rocprofiler_spm_counter_record_t*>{};
        record_ptrs.reserve(basic_metrics_samples.size() + derived_metrics_samples.size());
        for(const auto& rec : basic_metrics_samples)
            record_ptrs.push_back(&rec);
        for(const auto& rec : derived_metrics_samples)
            record_ptrs.push_back(&rec);

        spm_packet->record_cb(&(spm_packet->dispatch_data),
                              record_ptrs.data(),
                              record_ptrs.size(),
                              (1 << ROCPROFILER_SPM_RECORD_FLAG_DATA) |
                                  (flags << ROCPROFILER_SPM_RECORD_FLAG_DATA_LOST),
                              spm_packet->user_data,
                              spm_packet->record_callback_args);
    }
}

}  // namespace spm
}  // namespace rocprofiler
