#pragma once

#include <unordered_map>
#include <vector>
#include <deque>
#include <mutex>
#include <memory>
#include <shared_mutex>
#include <cstdint>

#include "timesync_db.hpp"

namespace rocm_timesync
{

class memory_client : public timesync_db
{
public:
    memory_client(int64_t max_entries_per_gpu);
    bool write(const entry_t& entry) override;
    bool write_batch(const std::vector<entry_t>& entries) override;
    bool lookup_oldest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) override;
    bool lookup_newest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) override;

private:
    bool lookup_before(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        timesync_point& point);

    bool lookup_after(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        timesync_point& point);

    struct mem_points {
        std::deque<timesync_point> points;
        std::mutex mtx;
    };

    int64_t max_entries_per_gpu_;
    mutable std::shared_mutex data_mtx_;
    std::unordered_map<uint32_t, std::shared_ptr<mem_points>> data_;

    using gpu_ptr = std::shared_ptr<mem_points>;

    gpu_ptr get_gpu(uint32_t gpu_id)
    {
        std::shared_lock lock(data_mtx_);

        auto it = data_.find(gpu_id);
        return it == data_.end() ? nullptr : it->second;
    }

    gpu_ptr get_or_create_gpu(uint32_t gpu_id)
    {
        std::unique_lock lock(data_mtx_);

        auto& ptr = data_[gpu_id];
        if (!ptr)
            ptr = std::make_shared<mem_points>();

        return ptr;
    }
};

} // namespace rocm_timesync
