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

    bool write(const entry_t& entry) override;
    bool write_batch(const std::vector<entry_t>& entries) override;
    bool lookup_newest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& k_points) override;

private:
    std::string make_write_url() const;
    std::string make_query_url() const;
    std::string query(std::string_view influxql);

    bool lookup_before(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        timesync_point& point) override;

    bool lookup_after(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        timesync_point& point) override;

    std::string host_;
    uint16_t port_;
    std::string database_;
};

} // namespace rocm_timesync
