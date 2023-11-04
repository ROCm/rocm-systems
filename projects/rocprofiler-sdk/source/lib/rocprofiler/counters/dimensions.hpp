#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lib/rocprofiler/counters/id_decode.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"

namespace rocprofiler
{
namespace counters
{
class MetricDimension
{
public:
    MetricDimension(std::string                                name,
                    uint64_t                                   dim_size,
                    rocprofiler_profile_counter_instance_types type)
    : name_(std::move(name))
    , size_(dim_size)
    , type_(type){};

    const std::string&                         name() const { return name_; }
    uint64_t                                   size() const { return size_; }
    rocprofiler_profile_counter_instance_types type() const { return type_; }
    bool                                       operator==(const MetricDimension& dim) const
    {
        return std::tie(name_, size_, type_) == std::tie(dim.name_, dim.size_, dim.type_);
    }

private:
    std::string                                name_;
    uint64_t                                   size_;
    rocprofiler_profile_counter_instance_types type_;
};

/*
{
    AgentBlockDimensionsMap = {
        "gfx906":{},
        "gfx908":{
            "TCC": [dim_1, dim_2 ... dim_n]
            "SQ": [dim_1, dim_2 ... dim_n]
            "TCP": [dim_1, dim_2 ... dim_n]
        }
    }
}
*/

// // map block_name -> MetricDimension
// using BlockDimensionMap = std::unordered_map<std::string, std::vector<MetricDimension>>;
// // map agent_name -> BlockDimensionMap
// using AgentBlockDimensionsMap = std::unordered_map<std::string, BlockDimensionMap&>;

// // map dimension_id -> MetricDimension
// using DimensionIdMap = std::unordered_map<uint64_t, MetricDimension>;

// // get the complete AgentBlockDimensionsMap
// const AgentBlockDimensionsMap&
// getAgentBlockDimensionsMap();

// // get specific dimensions for an agent, block_name
// const std::vector<MetricDimension>&
// getBlockDimension(const std::string&                         agent,
//                   std::string                                block_name,
//                   rocprofiler_profile_counter_instance_types dim);

// get all dimensions for an agent, block_name
std::vector<MetricDimension>
getBlockDimensions(const std::string& agent, const counters::Metric&);

// // get a specific dimension by id
// const MetricDimension&
// getDimensionById(uint64_t id);

}  // namespace counters
}  // namespace rocprofiler