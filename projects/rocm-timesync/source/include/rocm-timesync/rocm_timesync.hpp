#pragma once

#include <cstdint>
#include <string>
#include <variant>

namespace rocm_timesync
{

typedef enum {
    TIMESYNC_PRECISION_LOW = 0,
    TIMESYNC_PRECISION_HIGH
} ts_precision_t;

struct ts_client_config_t {
    std::string config_file;
    ts_precision_t precision;
};

int timesync_client_init(const ts_client_config_t& cfg);
int timesync_client_deinit();
int timesync_client_translate(uint32_t agent_kfd_gpu_id, uint64_t agent_timestamp, uint64_t& system_timestamp);

} // namespace rocm_timesync
