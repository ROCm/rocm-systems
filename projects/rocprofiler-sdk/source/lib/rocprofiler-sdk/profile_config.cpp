// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <glog/logging.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/evaluate_ast.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

extern "C" {
/**
 * @brief Create Profile Configuration.
 *
 * @param [in] agent Agent identifier
 * @param [in] counters_list List of GPU counters
 * @param [in] counters_count Size of counters list
 * @param [out] config_id Identifier for GPU counters group
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_create_profile_config(rocprofiler_agent_t              agent,
                                  rocprofiler_counter_id_t*        counters_list,
                                  size_t                           counters_count,
                                  rocprofiler_profile_config_id_t* config_id)
{
    std::shared_ptr<rocprofiler::counters::profile_config> config =
        std::make_shared<rocprofiler::counters::profile_config>();

    const auto& id_map = *CHECK_NOTNULL(rocprofiler::counters::getMetricIdMap());
    for(size_t i = 0; i < counters_count; i++)
    {
        auto& counter_id = counters_list[i];

        const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle);
        if(!metric_ptr) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        if(!rocprofiler::counters::checkValidMetric(std::string(agent.name), *metric_ptr))
        {
            return ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT;
        }
        config->metrics.push_back(*metric_ptr);
    }

    config->agent     = agent;
    config_id->handle = rocprofiler::counters::create_counter_profile(std::move(config));

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t ROCPROFILER_API
rocprofiler_destroy_profile_config(rocprofiler_profile_config_id_t config_id)
{
    rocprofiler::counters::destroy_counter_profile(config_id.handle);
    return ROCPROFILER_STATUS_SUCCESS;
}
}
