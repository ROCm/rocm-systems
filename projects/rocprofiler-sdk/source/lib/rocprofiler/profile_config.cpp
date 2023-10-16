#include <rocprofiler/rocprofiler.h>

#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/counters/core.hpp"
#include "lib/rocprofiler/counters/evaluate_ast.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"

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
    rocprofiler::counters::profile_config config;
    const auto&                           id_map = rocprofiler::counters::getMetricIdMap();

    for(size_t i = 0; i < counters_count; i++)
    {
        auto& counter_id = counters_list[i];

        const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle);
        if(!metric_ptr) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        config.metrics.push_back(*metric_ptr);

        auto agent_name = std::string(agent.name);
        auto req_counters =
            rocprofiler::counters::get_required_hardware_counters(agent_name, *metric_ptr);
        if(!req_counters) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        config.reqired_hw_counters.insert(req_counters->begin(), req_counters->end());

        const auto& asts      = rocprofiler::counters::get_ast_map();
        const auto* agent_map = rocprofiler::common::get_val(asts, agent_name);
        if(!agent_map) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        const auto* counter_ast = rocprofiler::common::get_val(*agent_map, metric_ptr->name());
        if(!counter_ast) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        config.asts.push_back(*counter_ast);
    }

    config.agent      = agent;
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
