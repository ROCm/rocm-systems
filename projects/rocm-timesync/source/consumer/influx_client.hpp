#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "timesync_db.hpp"

namespace rocm_timesync
{

class influx_client : public timesync_db
{
public:
    influx_client(std::string host,
                  uint16_t port,
                  std::string database);

    bool write(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        uint64_t system_timestamp) override;

    bool lookup(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        uint64_t& system_timestamp) override;

    std::string query(std::string_view influxql);

private:
    std::string make_write_url() const;
    std::string make_query_url() const;

    std::string host_;
    uint16_t port_;
    std::string database_;
};

} // namespace rocm_timesync
