// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "gsl_assert.h"
#include "stack_entry.h"
#include "stats.h"
#include "synchronized.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch_trace_collector::detail
{

// Identifies one autograd node. PyTorch draws sequence numbers from a
// thread-local counter that restarts at zero on each thread, so the owning
// thread is part of the identity.
struct SnapshotKey
{
    std::int64_t  seq_nr    = 0;
    std::uint64_t thread_id = 0;

    bool operator==(const SnapshotKey& other) const noexcept
    {
        return seq_nr == other.seq_nr && thread_id == other.thread_id;
    }
};

// Sequence numbers are dense and thread ids are small, so both fields are mixed
// to spread keys across buckets and shards.
inline std::size_t hash_snapshot_key(const SnapshotKey& key) noexcept
{
    constexpr std::uint64_t kGoldenRatio = 0x9e3779b97f4a7c15ULL;

    std::uint64_t value = static_cast<std::uint64_t>(key.seq_nr) + kGoldenRatio * key.thread_id;
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<std::size_t>(value);
}

}  // namespace torch_trace_collector::detail

template<>
struct std::hash<torch_trace_collector::detail::SnapshotKey>
{
    std::size_t operator()(const torch_trace_collector::detail::SnapshotKey& key) const noexcept
    {
        return torch_trace_collector::detail::hash_snapshot_key(key);
    }
};

namespace torch_trace_collector::detail
{

using rocprofiler_compute_tool::common::synchronized_t;

// Sharded store mapping an autograd node to a forward-stack snapshot, with
// per-shard LRU eviction.
class SnapshotStore
{
public:
    static constexpr std::size_t kNumShards    = 64;
    static constexpr std::size_t kShardSoftCap = 10000;

    static std::size_t shard_index(const SnapshotKey& key) noexcept
    {
        return hash_snapshot_key(key) % kNumShards;
    }

    void save(std::int64_t seq_nr, std::uint64_t thread_id, const std::vector<StackEntry>& stack)
    {
        const SnapshotKey key = {seq_nr, thread_id};
        shard_for(key).wlock(
            [&](Shard& shard)
            {
                auto it = shard.snapshots.find(key);
                if (it != shard.snapshots.end())
                {
                    // Nested forward ops can report the same sequence number; the
                    // most recent save wins.
                    it->second = stack;
                    lru_touch(shard, key);
                    inc(g_stats.snapshots_overwritten);
                    inc(g_stats.snapshots_saved);
                    return;
                }
                while (shard.snapshots.size() >= kShardSoftCap)
                {
                    evict_oldest(shard);
                }
                shard.snapshots.emplace(key, stack);
                lru_touch(shard, key);
                inc(g_stats.snapshots_saved);
            });
    }

    bool consume(std::int64_t seq_nr, std::uint64_t thread_id, std::vector<StackEntry>* out_stack)
    {
        const SnapshotKey key = {seq_nr, thread_id};
        return shard_for(key).wlock(
            [&](Shard& shard)
            {
                auto it = shard.snapshots.find(key);
                if (it == shard.snapshots.end())
                    return false;
                Expects(shard.lru_idx.find(key) != shard.lru_idx.end());
                *out_stack = std::move(it->second);
                shard.snapshots.erase(it);
                lru_remove(shard, key);
                inc(g_stats.snapshots_consumed);
                return true;
            });
    }

    std::size_t pending() const
    {
        std::size_t total = 0;
        for (const auto& guarded_shard : shards_)
        {
            total += guarded_shard.rlock([](const Shard& shard) { return shard.snapshots.size(); });
        }
        return total;
    }

    void clear()
    {
        for (auto& guarded_shard : shards_)
        {
            guarded_shard.wlock(
                [](Shard& shard)
                {
                    shard.snapshots.clear();
                    shard.lru_order.clear();
                    shard.lru_idx.clear();
                });
        }
    }

private:
    // Every key in snapshots also holds a place in lru_order and an iterator to
    // it in lru_idx, so eviction always has a key to remove.
    struct Shard
    {
        std::unordered_map<SnapshotKey, std::vector<StackEntry>>          snapshots;
        std::list<SnapshotKey>                                            lru_order;
        std::unordered_map<SnapshotKey, std::list<SnapshotKey>::iterator> lru_idx;
    };

    synchronized_t<Shard>& shard_for(const SnapshotKey& key) { return shards_[shard_index(key)]; }

    static void lru_remove(Shard& shard, const SnapshotKey& key)
    {
        auto it = shard.lru_idx.find(key);
        if (it == shard.lru_idx.end())
            return;
        shard.lru_order.erase(it->second);
        shard.lru_idx.erase(it);
    }

    static void lru_touch(Shard& shard, const SnapshotKey& key)
    {
        lru_remove(shard, key);
        shard.lru_order.push_back(key);
        auto tail = shard.lru_order.end();
        --tail;
        shard.lru_idx.emplace(key, tail);
    }

    static void evict_oldest(Shard& shard)
    {
        // save() evicts until the shard is under its cap, so an empty order
        // would not terminate.
        Expects(!shard.lru_order.empty());
        const SnapshotKey oldest = shard.lru_order.front();
        shard.lru_order.pop_front();
        shard.lru_idx.erase(oldest);
        shard.snapshots.erase(oldest);
        inc(g_stats.snapshots_dropped);
    }

    std::array<synchronized_t<Shard>, kNumShards> shards_;
};

inline SnapshotStore g_snapshots;

}  // namespace torch_trace_collector::detail
