#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace rocm_timesync
{

struct db_stats
{
    uint64_t writes = 0;
    uint64_t lookup_queries = 0;
    uint64_t lookup_exact = 0;
    uint64_t lookup_interpolated = 0;
    uint64_t lookup_miss_too_old = 0;
    uint64_t lookup_miss_too_new = 0;
    uint64_t lookup_extrapolated_backward = 0;
    uint64_t lookup_extrapolated_forward = 0;
    uint64_t lookup_error = 0;

    std::unordered_map<std::string, uint64_t> extra;
};

struct timesync_stats {
    std::unordered_map<std::string, db_stats> dbs;
};

inline void to_json(json& j, const db_stats& s)
{
    j = {
        {"writes", s.writes},
        {"lookup_queries", s.lookup_queries},
        {"lookup_exact", s.lookup_exact},
        {"lookup_interpolated", s.lookup_interpolated},
        {"lookup_miss_too_old", s.lookup_miss_too_old},
        {"lookup_miss_too_new", s.lookup_miss_too_new},
        {"lookup_extrapolated_backward", s.lookup_extrapolated_backward},
        {"lookup_extrapolated_forward", s.lookup_extrapolated_forward},
        {"lookup_error", s.lookup_error},
        {"extra", s.extra}
    };
}

inline void to_json(json& j, const timesync_stats& s)
{
    j = json::object();

    for (const auto& [name, stats] : s.dbs)
        j[name] = stats;
}

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
        MissTooOld,
        MissTooNew,
        Error
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
        for (const auto& entry : entries) {
            if (!write(entry)) return false;
            ++stats_.writes;
        }
        return true;
    }

    virtual LookupResult lookup(uint32_t gpu_id, uint64_t gpu_timestamp, uint64_t& system_timestamp)
    {
        timesync_point before, after;

        // record the query
        ++stats_.lookup_queries;

        //
        // We must have at least one point before.
        //
        if (!lookup_before(gpu_id, gpu_timestamp, before))
            return record_result(LookupResult::MissTooOld);

        //
        // We must have at least one point after.
        //
        if (!lookup_after(gpu_id, gpu_timestamp, after))
            return record_result(LookupResult::MissTooNew);

        //
        // Exact match.
        //
        if (before.gpu_timestamp == after.gpu_timestamp)
        {
            system_timestamp = before.system_timestamp;
            return record_result(LookupResult::Exact);
        }

        //
        // Linear interpolation.
        //
        system_timestamp = linear_interpolation(gpu_timestamp, before, after);
        return record_result(LookupResult::Interpolated);
    }

    virtual LookupResult extrapolate(LookupResult lookup_res, uint32_t gpu_id, uint64_t gpu_timestamp, uint64_t& system_timestamp)
    {
        //
        // We must have at least two points to extrapolate from.
        //
        std::vector<timesync_point> pair;

        if (lookup_res == LookupResult::MissTooOld) {
            if (!lookup_oldest_k(gpu_id, 2, pair))
                return record_result(LookupResult::Error);
        } else if (lookup_res == LookupResult::MissTooNew) {
            if (!lookup_newest_k(gpu_id, 2, pair))
                return record_result(LookupResult::Error);
        } else {
            return record_result(LookupResult::Error);
        }

        if (pair.size() != 2)
            return record_result(LookupResult::Error);

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
            return record_result(LookupResult::Interpolated);
        }

        //
        // Linear extrapolation.
        //
        system_timestamp = linear_extrapolation(gpu_timestamp, p0, p1);
        return (lookup_res == LookupResult::MissTooOld) ?
            record_result(LookupResult::ExtrapolatedBackward) :
            record_result(LookupResult::ExtrapolatedForward);
    }

    LookupResult lookup_or_extrapolate(uint32_t gpu_id, uint64_t gpu_timestamp, uint64_t& system_timestamp)
    {
        //
        // lookup() will attempt to find the tightest range of timestamps
        // surrounding the 'gpu_timestamp' and interpolate based on them.
        //
        auto res = lookup(gpu_id, gpu_timestamp, system_timestamp);
        if (IsSuccess(res))
            return res;

        if (res == LookupResult::MissTooOld) {
            std::cerr << "WARNING: received request to translate point that is older than anything we have tracked" << std::endl;
        }

        //
        // extrapolate() will use the 2 nearest known timestamps as
        // references and extrapolate to 'gpu_timestamp' based on them.
        //
        return extrapolate(res, gpu_id, gpu_timestamp, system_timestamp);
    }

    virtual timesync_stats stats() const
    {
        return {
            { { "db", stats_ } }
        };
    }

    virtual bool lookup_oldest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) = 0;
    virtual bool lookup_newest_k(uint32_t gpu_id, uint64_t k, std::vector<timesync_point>& points) = 0;

protected:
    void record_write()
    {
        ++stats_.writes;
    }

    void record_write_batch(uint64_t batch_size)
    {
        stats_.writes += batch_size;
    }

private:
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

    LookupResult record_result(LookupResult res)
    {
        // record how the query was handled
        switch (res) {
        case LookupResult::Exact:
            ++stats_.lookup_exact;
            break;
        case LookupResult::Interpolated:
            ++stats_.lookup_interpolated;
            break;
        case LookupResult::ExtrapolatedBackward:
            ++stats_.lookup_extrapolated_backward;
            break;
        case LookupResult::ExtrapolatedForward:
            ++stats_.lookup_extrapolated_forward;
            break;
        case LookupResult::MissTooOld:
            ++stats_.lookup_miss_too_old;
            break;
        case LookupResult::MissTooNew:
            ++stats_.lookup_miss_too_new;
            break;
        case LookupResult::Error:
        default:
            ++stats_.lookup_error;
            break;
        }

        return res;
    }

    virtual bool lookup_before(uint32_t gpu_id, uint64_t gpu_timestamp, timesync_point& point) = 0;
    virtual bool lookup_after(uint32_t gpu_id, uint64_t gpu_timestamp, timesync_point& point) = 0;

    struct db_stats stats_;
};

class cached_timesync_db : public timesync_db
{
public:
    cached_timesync_db(std::unique_ptr<timesync_db> cache, std::unique_ptr<timesync_db> backing)
    : cache_(std::move(cache))
    , backing_(std::move(backing)) {}

    ~cached_timesync_db() {}

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
        auto ret = cache_->lookup(gpu_id, gpu_timestamp, system_timestamp);
        if (IsSuccess(ret))
            return ret;

        // there is no point in querying the backing store if we missed on too new of a request
        if (ret == LookupResult::MissTooNew)
            return ret;

        return backing_->lookup(gpu_id, gpu_timestamp, system_timestamp);
    }

    LookupResult extrapolate(LookupResult lookup_res, uint32_t gpu_id, uint64_t gpu_timestamp, uint64_t& system_timestamp) override
    {
        // use backing for old timestamps; use cache for new timestamps
        if (lookup_res == LookupResult::MissTooOld)
            return backing_->extrapolate(lookup_res, gpu_id, gpu_timestamp, system_timestamp);
        else if (lookup_res == LookupResult::MissTooNew)
            return cache_->extrapolate(lookup_res, gpu_id, gpu_timestamp, system_timestamp);
        else
            return LookupResult::Error;
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

    virtual timesync_stats stats() const override
    {
        return {
            {
                { "cache", cache_->stats().dbs.at("db") },
                { "db", backing_->stats().dbs.at("db") }
            }
        };
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
};

} // namespace rocm_timesync
