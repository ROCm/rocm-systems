#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <memory>
#include <mutex>

namespace rocm_timesync
{

class timesync_db
{
public:
    struct timesync_point {
        uint64_t gpu_timestamp;
        uint64_t system_timestamp;
    };

    struct entry_t {
        uint32_t gpu_id;
        timesync_point point;
    };

    virtual ~timesync_db() = default;
    virtual bool write(const entry_t& entry) = 0;
    virtual bool write_batch(
        const std::vector<entry_t>& entries)
    {
        for (const auto& entry : entries)
            if (!write(entry)) return false;
        return true;
    }

    virtual bool lookup(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        uint64_t& system_timestamp)
    {
        timesync_point before, after;

        //
        // We must have at least one point before.
        //
        if (!lookup_before(gpu_id, gpu_timestamp, before))
            return false;

        //std::cerr << "gpu_timestamp=" << gpu_timestamp << " before=" << before.gpu_timestamp << std::endl;

        //
        // We must have at least one point after.
        //
        if (!lookup_after(gpu_id, gpu_timestamp, after))
            return false;

        //std::cerr << "gpu_timestamp=" << gpu_timestamp << " after=" << after.gpu_timestamp << std::endl;

        //
        // Exact match.
        //
        if (before.gpu_timestamp == after.gpu_timestamp)
        {
            system_timestamp = before.system_timestamp;
            return true;
        }

        //
        // Linear interpolation.
        //
        system_timestamp = linear_interpolation(gpu_timestamp, before, after);
        return true;
    }

    virtual bool lookup_or_extrapolate(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        uint64_t& system_timestamp)
    {
        if (lookup(gpu_id, gpu_timestamp, system_timestamp))
            return true;

        //
        // We must have at least two points.
        //
        std::vector<timesync_point> newest_two;
        if (!lookup_newest_k(gpu_id, 2, newest_two))
            return false;

        if (newest_two.size() != 2)
            return false;

        auto& p0 = newest_two[0];
        auto& p1 = newest_two[1];

        //
        // It is possible that a new timestamp arrives in between the call to
        // lookup() failing and the subsequent retrieval of the two most recent
        // timestamps.
        //
        // That event in turn makes standard interpolation possible if the point to
        // be trnanslated falls in the range
        //
        // Check for that case here.
        //
        if (gpu_timestamp >= p0.gpu_timestamp && gpu_timestamp <= p1.gpu_timestamp) {
            system_timestamp = linear_interpolation(gpu_timestamp, p0, p1);
            return true;
        }

        //
        // gpu_timestamp must be newer than our most recent points
        //
        assert(gpu_timestamp >= p1.gpu_timestamp);

        //
        // Linear extrapolation.
        //
        system_timestamp = linear_extrapolation(gpu_timestamp, p0, p1);
        return true;
    }

    virtual bool lookup_newest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) = 0;

private:
    virtual bool lookup_before(uint32_t gpu_id, uint64_t gpu_timestamp, timesync_point& point) = 0;
    virtual bool lookup_after(uint32_t gpu_id, uint64_t gpu_timestamp, timesync_point& point) = 0;

    uint64_t linear_interpolation(uint64_t gpu_timestamp, timesync_point& before, timesync_point& after)
    {
        const uint64_t gpu_delta = gpu_timestamp - before.gpu_timestamp;
        const uint64_t gpu_range = after.gpu_timestamp - before.gpu_timestamp;
        const uint64_t system_range = after.system_timestamp - before.system_timestamp;
        const __uint128_t scaled = static_cast<__uint128_t>(gpu_delta) * system_range;

        return before.system_timestamp + static_cast<uint64_t>(scaled / gpu_range);
    }

    uint64_t linear_extrapolation(uint64_t gpu_timestamp, timesync_point& p0, timesync_point& p1)
    {
        const uint64_t gpu_delta = gpu_timestamp - p1.gpu_timestamp;
        const uint64_t gpu_range = p1.gpu_timestamp - p0.gpu_timestamp;
        const uint64_t system_range = p1.system_timestamp - p0.system_timestamp;
        const __uint128_t scaled = static_cast<__uint128_t>(gpu_delta) * system_range;

        return p1.system_timestamp + static_cast<uint64_t>(scaled / gpu_range);
    }
};

class cached_timesync_db : public timesync_db
{
public:
    cached_timesync_db(std::unique_ptr<timesync_db> cache, std::unique_ptr<timesync_db> backing)
    : cache_(std::move(cache))
    , backing_(std::move(backing)) {}

    bool write(const entry_t& entry) override
    {
        //std::lock_guard lock(mtx_);
        (void)cache_->write(entry);
        return backing_->write(entry);
    }

    bool write_batch(const std::vector<entry_t>& entries) override
    {
        //std::lock_guard lock(mtx_);
        (void)cache_->write_batch(entries);
        return backing_->write_batch(entries);
    }

    bool lookup(uint32_t gpu_id, uint64_t gpu_timestamp, uint64_t& system_timestamp) override
    {
        //std::lock_guard lock(mtx_);
        if (cache_->lookup(gpu_id, gpu_timestamp, system_timestamp)) {
            //std::cerr << "MEM_CLIENT HIT gpu_timestamp=" << gpu_timestamp << std::endl;
            return true;
        }
        //std::cerr << "MEM_CLIENT MISS gpu_timestamp=" << gpu_timestamp << std::endl;

        printf("%luth backing query\n", ++backing_queries_);

        if (backing_->lookup(gpu_id, gpu_timestamp, system_timestamp)) {
            //std::cerr << "INFLUX_CLIENT HIT gpu_timestamp=" << gpu_timestamp << std::endl;
            return true;
        }
        //std::cerr << "INFLUX_CLIENT MISS gpu_timestamp=" << gpu_timestamp << std::endl;
        return false;
    }

    bool lookup_newest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) override
    {
        //std::lock_guard lock(mtx_);
        if (cache_->lookup_newest_k(gpu_id, k, points))
            return true;
        return backing_->lookup_newest_k(gpu_id, k, points);
    }

private:
    bool lookup_before(uint32_t gpu_id, uint64_t gpu_timestamp, timesync_point& point) override
    {
        std::abort();
        return false;
    }

    bool lookup_after(uint32_t gpu_id, uint64_t gpu_timestamp, timesync_point& point) override
    {
        std::abort();
        return false;
    }

    //std::mutex mtx_;

    std::unique_ptr<timesync_db> cache_;
    std::unique_ptr<timesync_db> backing_;
    uint64_t backing_queries_ = 0;

};

} // namespace rocm_timesync
