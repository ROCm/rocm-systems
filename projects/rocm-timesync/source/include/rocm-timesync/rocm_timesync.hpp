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

struct ts_db_influx_t {
    std::string host{"localhost"};
    uint16_t port{8086};
    std::string database;
    // TODO: credentials, etc.
};

using ts_db_config_t = std::variant<ts_db_influx_t>;

struct ts_config_t {
    ts_precision_t precision;
    ts_db_config_t db_config;
};

int timesync_init(const ts_config_t& cfg);
int timesync_deinit();
int timesync_translate(uint32_t agent_kfd_gpu_id, uint64_t agent_timestamp, uint64_t *system_timestamp);

} // namespace rocm_timesync
