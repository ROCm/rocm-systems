// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/sdk_pmc/types.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::pmc::collectors::sdk_pmc
{

/**
 * @brief SDK PMC sample type for trace cache serialization.
 *
 * Stores a snapshot of GPU hardware counter values collected via
 * rocprofiler_sample_device_counting_service().
 */
struct sample
{
    uint32_t              device_id = 0;
    uint64_t              timestamp = 0;
    std::vector<uint64_t> counter_ids;
    std::vector<double>   values;

    sample() = default;
    sample(uint32_t dev_id, uint64_t ts, std::vector<uint64_t> ids,
           std::vector<double> vals)
    : device_id(dev_id)
    , timestamp(ts)
    , counter_ids(std::move(ids))
    , values(std::move(vals))
    {}
};

}  // namespace rocprofsys::pmc::collectors::sdk_pmc
