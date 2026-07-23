#pragma once

#include <cstdint>

namespace rocm_timesync
{

typedef enum {
    TIMESYNC_PRECISION_LOW,
    TIMESYNC_PRECISION_HIGH
} ts_precision_t;

int timesync_init(ts_precision_t precision);
int timesync_deinit();
int timesync_translate(uint32_t agent_kfd_gpu_id, uint64_t agent_timestamp, uint64_t *system_timestamp);

} // namespace rocm_timesync
