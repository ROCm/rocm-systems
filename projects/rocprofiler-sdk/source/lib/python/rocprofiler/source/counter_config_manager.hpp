// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "types.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace python
{
/**
 * @brief Manages counter configurations for different agents
 *
 * This class handles the creation and caching of counter configurations
 * for each GPU agent. Configurations are created lazily when first
 * requested for a specific agent.
 */
class CounterConfigManager
{
public:
    /**
     * @brief Construct a new CounterConfigManager
     * @param metric_names List of counter/metric names to collect
     */
    explicit CounterConfigManager(const std::vector<std::string>& metric_names);

    ~CounterConfigManager();

    // Non-copyable, non-movable
    CounterConfigManager(const CounterConfigManager&) = delete;
    CounterConfigManager& operator=(const CounterConfigManager&) = delete;
    CounterConfigManager(CounterConfigManager&&)                 = delete;
    CounterConfigManager& operator=(CounterConfigManager&&) = delete;

    /**
     * @brief Get or create a counter config for a specific agent
     * @param agent_id The agent to get the config for
     * @return The counter configuration ID
     */
    rocprofiler_counter_config_id_t get_config_for_agent(rocprofiler_agent_id_t agent_id);

    /**
     * @brief Resolve counter IDs for an agent given counter names
     * @param agent_id The agent to resolve counters for
     * @param counter_names The names of counters to resolve
     * @return Vector of resolved counter IDs
     */
    std::vector<rocprofiler_counter_id_t> resolve_counters_for_agent(
        rocprofiler_agent_id_t          agent_id,
        const std::vector<std::string>& counter_names);

    /**
     * @brief Get counter info from a counter ID
     * @param counter_id The counter ID to query
     * @return CounterInfo structure with counter details
     */
    static CounterInfo get_counter_info(rocprofiler_counter_id_t counter_id);

    /**
     * @brief Get dimension info for a counter
     * @param counter_id The counter ID
     * @return Vector of dimension info
     */
    std::vector<rocprofiler_counter_record_dimension_info_t> get_dimensions(
        rocprofiler_counter_id_t counter_id);

    /**
     * @brief Destroy all cached counter configurations
     */
    void destroy_all_configs();

    /**
     * @brief Get the list of metric names being collected
     */
    const std::vector<std::string>& metric_names() const { return metric_names_; }

private:
    std::vector<std::string>  metric_names_;
    mutable std::shared_mutex cache_mutex_;

    // Cache: agent_id.handle -> counter_config_id
    std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> config_cache_;

    // Cache: counter_id.handle -> dimension info
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_record_dimension_info_t>>
        dimension_cache_;
};

}  // namespace python
}  // namespace rocprofiler
