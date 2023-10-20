#include <rocprofiler/rocprofiler.h>

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/counters/evaluate_ast.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"
#include "lib/rocprofiler/hsa/queue.hpp"
#include "lib/rocprofiler/hsa/queue_controller.hpp"

extern "C" {
/**
 * @brief Query Counter name.
 *
 * @param [in] counter_id
 * @param [out] name if nullptr, size will be returned
 * @param [out] size
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_counter_name(rocprofiler_counter_id_t counter_id, const char** name, size_t* size)
{
    const auto& id_map = rocprofiler::counters::getMetricIdMap();

    if(const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle))
    {
        *name = metric_ptr->name().c_str();
        *size = metric_ptr->name().size();
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
}

/**
 * @brief Query Counter Instances Count.
 *
 * @param [in] counter_id
 * @param [out] instance_count
 * @return rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_counter_instance_count(rocprofiler_agent_t      agent,
                                         rocprofiler_counter_id_t counter_id,
                                         size_t*                  instance_count)
{
    const auto& id_map     = rocprofiler::counters::getMetricIdMap();
    const auto* metric_ptr = rocprofiler::common::get_val(id_map, counter_id.handle);
    if(!metric_ptr) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    *instance_count = 0;
    // Special counters like KERNEL_DURATION are not real counters and wont
    // have any query info.
    if(!metric_ptr->special().empty())
    {
        *instance_count = 1;
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // Returns the set of hardware counters needed to evaluate the metric.
    // For derived metrics, this can be more than one counter. In that case,
    // we return the maximum instance count among all underlying counters.
    auto req_counters =
        rocprofiler::counters::get_required_hardware_counters(std::string(agent.name), *metric_ptr);
    if(!req_counters) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;

    // NOTE: to look up instance information, we require HSA be init'd. Reason
    // for this is the call to get instance information is an HSA call.
    const auto* maybe_agent = rocprofiler::common::get_val(
        rocprofiler::hsa::get_queue_controller().get_supported_agents(), agent.id.handle);
    if(!maybe_agent)
    {
        LOG(ERROR) << "HSA must be loaded to obtain instance information.";
        return ROCPROFILER_STATUS_ERROR;
    }

    for(const auto& counter : *req_counters)
    {
        if(!counter.special().empty())
        {
            *instance_count = std::max(size_t(1), *instance_count);
            continue;
        }
        auto query_info = rocprofiler::aql::get_query_info(maybe_agent->get_hsa_agent(), counter);
        *instance_count = std::max(static_cast<size_t>(query_info.instance_count), *instance_count);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}
/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent
 * @param [out] counters_list
 * @param [out] counters_count
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_t                 agent,
                                             rocprofiler_available_counters_cb_t cb,
                                             void*                               user_data)
{
    const auto& metrics = rocprofiler::counters::getMetricsForAgent(std::string(agent.name));
    std::vector<rocprofiler_counter_id_t> ids;
    ids.reserve(metrics.size());
    for(const auto& metric : metrics)
    {
        ids.push_back({.handle = metric.id()});
    }

    return cb(ids.data(), ids.size(), user_data);
}
}
