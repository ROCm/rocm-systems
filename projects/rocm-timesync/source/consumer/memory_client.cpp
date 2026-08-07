#include "memory_client.hpp"

#include <algorithm>

namespace rocm_timesync
{

memory_client::memory_client(
    uint32_t max_entries_per_gpu)
: max_entries_per_gpu_(max_entries_per_gpu)
{}

bool
memory_client::write(const entry_t& entry)
{
    auto gpu = get_or_create_gpu(entry.gpu_id);
    std::lock_guard<std::mutex> lock(gpu->mtx);
    auto& points = gpu->points;

    points.push_back(entry.point);

    if(max_entries_per_gpu_ > 0 && points.size() > max_entries_per_gpu_)
        points.pop_front();

    return true;
}

bool
memory_client::write_batch(const std::vector<entry_t>& entries)
{
    std::unordered_map<uint32_t, std::vector<timesync_point>> groups;

    for (const auto& entry : entries)
        groups[entry.gpu_id].push_back(entry.point);

    for (auto& [gpu_id, points_to_add] : groups)
    {
        auto gpu = get_or_create_gpu(gpu_id);
        std::lock_guard<std::mutex> lock(gpu->mtx);
        auto& points = gpu->points;

        points.insert(
            points.end(),
            points_to_add.begin(),
            points_to_add.end()
        );

        if (max_entries_per_gpu_ > 0 && points.size() > max_entries_per_gpu_)
            points.erase(points.begin(), points.begin() + (points.size() - max_entries_per_gpu_));
    }

    return true;
}

bool
memory_client::lookup_newest_k(
    uint32_t gpu_id,
    uint64_t k,
    std::vector<timesync_point>& k_points)
{
    auto gpu = get_gpu(gpu_id);
    if (gpu == nullptr)
        return false;

    std::lock_guard<std::mutex> lock(gpu->mtx);
    const auto& points = gpu->points;

    auto to_copy = std::min(k, points.size());
    k_points.assign(points.end() - to_copy, points.end());
    return true;
}

bool
memory_client::lookup_before(
    uint32_t gpu_id,
    uint64_t gpu_timestamp,
    timesync_point& point)
{

    auto gpu = get_gpu(gpu_id);
    if (gpu == nullptr)
        return false;

    std::lock_guard<std::mutex> lock(gpu->mtx);
    const auto& points = gpu->points;
    
    auto it = std::lower_bound(
        points.begin(),
        points.end(),
        gpu_timestamp,
        [](const timesync_point& point, uint64_t timestamp) {
            return point.gpu_timestamp < timestamp;
        });

    if(it != points.end() && it->gpu_timestamp == gpu_timestamp)
    {
        point = *it;
        return true;
    }

    if(it == points.begin())
        return false;

    --it;

    point = *it;
    return true;
}

bool
memory_client::lookup_after(
    uint32_t gpu_id,
    uint64_t gpu_timestamp,
    timesync_point& point)
{

    auto gpu = get_gpu(gpu_id);
    if (gpu == nullptr)
        return false;

    std::lock_guard<std::mutex> lock(gpu->mtx);
    const auto& points = gpu->points;
    
    auto it = std::lower_bound(
        points.begin(),
        points.end(),
        gpu_timestamp,
        [](const timesync_point& point, uint64_t timestamp) {
            return point.gpu_timestamp < timestamp;
        });

    if(it == points.end())
        return false;

    point = *it;
    return true;
}

} // namespace rocm_timesync
