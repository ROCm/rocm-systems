#pragma once

#include <cstdint>

namespace rocm_timesync
{

class timesync_db
{
public:
    virtual ~timesync_db() = default;

    virtual bool write(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        uint64_t system_timestamp) = 0;

    virtual bool lookup(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        uint64_t& system_timestamp) = 0;

};

} // namespace rocm_timesync
