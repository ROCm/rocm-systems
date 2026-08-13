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

    enum class LookupResult {
        Exact,
        Interpolated,
        ExtrapolatedBackward,
        ExtrapolatedForward,
        ErrTooOld,
        ErrTooNew,
        ErrUnknown
    };

    static constexpr bool IsSuccess(LookupResult r)
    {
        return r == LookupResult::Exact || r == LookupResult::Interpolated ||
            r == LookupResult::ExtrapolatedBackward || r == LookupResult::ExtrapolatedForward;
    }

    static constexpr bool IsError(LookupResult r)
    {
        return !IsSuccess(r);
    }

    virtual ~timesync_db() = default;
    virtual bool write(const entry_t& entry) = 0;
    virtual bool write_batch(
        const std::vector<entry_t>& entries)
    {
        for (const auto& entry : entries)
            if (!write(entry)) return false;
        return true;
    }

    virtual LookupResult lookup(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        uint64_t& system_timestamp)
    {
        timesync_point before, after;

        //
        // We must have at least one point before.
        //
        if (!lookup_before(gpu_id, gpu_timestamp, before))
            return LookupResult::ErrTooOld;

        //std::cerr << "gpu_timestamp=" << gpu_timestamp << " before=" << before.gpu_timestamp << std::endl;

        //
        // We must have at least one point after.
        //
        if (!lookup_after(gpu_id, gpu_timestamp, after))
            return LookupResult::ErrTooNew;

        //std::cerr << "gpu_timestamp=" << gpu_timestamp << " after=" << after.gpu_timestamp << std::endl;

        //
        // Exact match.
        //
        if (before.gpu_timestamp == after.gpu_timestamp)
        {
            system_timestamp = before.system_timestamp;
            return LookupResult::Exact;
        }

        //
        // Linear interpolation.
        //
        system_timestamp = linear_interpolation(gpu_timestamp, before, after);
        return LookupResult::Interpolated;
    }

    virtual LookupResult lookup_or_extrapolate(
        uint32_t gpu_id,
        uint64_t gpu_timestamp,
        uint64_t& system_timestamp)
    {
        //
        // lookup() will attempt to find the tightest range of timestamps
        // surrounding the 'gpu_timestamp' and interpolate based on them.
        //
        auto res = lookup(gpu_id, gpu_timestamp, system_timestamp);
        if (IsSuccess(res))
            return res;

        if (res == LookupResult::ErrTooOld) {
            std::cerr << "WARNING: received request to translate point that is older than anything we have tracked" << std::endl;
        }

        //
        // We must have at least two points to extrapolate from.
        // Either choose the oldest or newest 2, depending on whether
        // we're extrapolating backwards or forwards
        //
        std::vector<timesync_point> pair;

        if (res == LookupResult::ErrTooOld) {
            if (!lookup_oldest_k(gpu_id, 2, pair))
                return LookupResult::ErrUnknown;
        } else if (res == LookupResult::ErrTooNew) {
            if (!lookup_newest_k(gpu_id, 2, pair))
                return LookupResult::ErrUnknown;
        } else {
            return res;
        }

        if (pair.size() != 2)
            return LookupResult::ErrUnknown;

        auto& p0 = pair[0];
        auto& p1 = pair[1];

        //
        // It is possible that a new timestamp arrives in between the call to
        // lookup() failing and the subsequent retrieval of the two most recent
        // timestamps.
        //
        // That event in turn makes standard interpolation possible if the point to
        // be translated falls in the range
        //
        // Check for that case here.
        //
        if (gpu_timestamp >= p0.gpu_timestamp && gpu_timestamp <= p1.gpu_timestamp) {
            system_timestamp = linear_interpolation(gpu_timestamp, p0, p1);
            return LookupResult::Interpolated;
        }

        //
        // Linear extrapolation.
        //
        system_timestamp = linear_extrapolation(gpu_timestamp, p0, p1);
        return (res == LookupResult::ErrTooOld) ? LookupResult::ExtrapolatedBackward : LookupResult::ExtrapolatedForward;
    }

    virtual bool lookup_oldest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) = 0;
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
        const int64_t gpu_delta = gpu_timestamp - p1.gpu_timestamp; // can be negative
        const uint64_t gpu_range = p1.gpu_timestamp - p0.gpu_timestamp;
        const uint64_t system_range = p1.system_timestamp - p0.system_timestamp;
        const __uint128_t scaled = static_cast<__int128_t>(gpu_delta) * system_range;

        return p1.system_timestamp + static_cast<int64_t>(scaled / gpu_range);
    }
};

class cached_timesync_db : public timesync_db
{
public:
    cached_timesync_db(std::unique_ptr<timesync_db> cache, std::unique_ptr<timesync_db> backing)
    : cache_(std::move(cache))
    , backing_(std::move(backing)) {}

    ~cached_timesync_db() {
        std::cout << "cache_queries: " << (cache_queries_ - cache_misses_) << "/" << cache_queries_ << std::endl;
        std::cout << "backing_queries: " << (backing_queries_ - backing_misses_) << "/" << backing_queries_ << std::endl;
    }

    bool write(const entry_t& entry) override
    {
        (void)cache_->write(entry);
        return backing_->write(entry);
    }

    bool write_batch(const std::vector<entry_t>& entries) override
    {
        (void)cache_->write_batch(entries);
        return backing_->write_batch(entries);
    }

    LookupResult lookup(uint32_t gpu_id, uint64_t gpu_timestamp, uint64_t& system_timestamp) override
    {
        ++cache_queries_;
        auto ret = cache_->lookup(gpu_id, gpu_timestamp, system_timestamp);
        if (IsSuccess(ret))
            return ret;
        ++cache_misses_;

        // there is no point in querying the backing store if we missed on too new of a request
        if (ret == LookupResult::ErrTooNew)
            return ret;

        ++backing_queries_;
        ret = backing_->lookup(gpu_id, gpu_timestamp, system_timestamp);
        if (IsSuccess(ret))
            return ret;
        ++backing_misses_;

        std::cerr << "INFLUX_CLIENT MISS gpu_timestamp=" << gpu_timestamp << std::endl;
        return ret;
    }

    bool lookup_oldest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) override
    {
        return backing_->lookup_oldest_k(gpu_id, k, points);
    }

    bool lookup_newest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) override
    {
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

    std::unique_ptr<timesync_db> cache_;
    std::unique_ptr<timesync_db> backing_;

    uint64_t cache_queries_ = 0;
    uint64_t cache_misses_ = 0;
    uint64_t backing_queries_ = 0;
    uint64_t backing_misses_ = 0;
};

} // namespace rocm_timesync
