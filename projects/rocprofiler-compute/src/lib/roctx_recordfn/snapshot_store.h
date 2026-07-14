// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"
#include "stats.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace roctx_recordfn::detail
{

// Sharded store mapping seqNr to a forward-stack snapshot, with per-shard
// LRU eviction.
class SnapshotStore
{
public:
    static constexpr std::size_t kNumShards    = 64;
    static constexpr std::size_t kShardSoftCap = 10000;

    void save(std::int64_t seq_nr, const std::vector<StackEntry>& stack)
    {
        Shard&                      shard = shard_for(seq_nr);
        std::lock_guard<std::mutex> guard(shard.mutex);
        auto                        it = shard.snapshots.find(seq_nr);
        if (it != shard.snapshots.end())
        {
            it->second = stack;
            lru_touch(shard, seq_nr);
            inc(g_stats.snapshots_saved);
            return;
        }
        while (shard.snapshots.size() >= kShardSoftCap)
        {
            evict_oldest(shard);
        }
        shard.snapshots.emplace(seq_nr, stack);
        lru_touch(shard, seq_nr);
        inc(g_stats.snapshots_saved);
    }

    bool consume(std::int64_t seq_nr, std::vector<StackEntry>* out_stack)
    {
        Shard&                      shard = shard_for(seq_nr);
        std::lock_guard<std::mutex> guard(shard.mutex);
        auto                        it = shard.snapshots.find(seq_nr);
        if (it == shard.snapshots.end())
            return false;
        *out_stack = std::move(it->second);
        shard.snapshots.erase(it);
        lru_remove(shard, seq_nr);
        inc(g_stats.snapshots_consumed);
        return true;
    }

    std::size_t pending()
    {
        std::size_t total = 0;
        for (auto& shard : shards_)
        {
            std::lock_guard<std::mutex> guard(shard.mutex);
            total += shard.snapshots.size();
        }
        return total;
    }

    void clear()
    {
        for (auto& shard : shards_)
        {
            std::lock_guard<std::mutex> guard(shard.mutex);
            shard.snapshots.clear();
            shard.lru_order.clear();
            shard.lru_idx.clear();
        }
    }

private:
    struct Shard
    {
        std::mutex                                                          mutex;
        std::unordered_map<std::int64_t, std::vector<StackEntry>>           snapshots;
        std::list<std::int64_t>                                             lru_order;
        std::unordered_map<std::int64_t, std::list<std::int64_t>::iterator> lru_idx;
    };

    Shard& shard_for(std::int64_t seq_nr)
    {
        return shards_[static_cast<std::size_t>(seq_nr) % kNumShards];
    }

    static void lru_remove(Shard& shard, std::int64_t seq_nr)
    {
        auto it = shard.lru_idx.find(seq_nr);
        if (it == shard.lru_idx.end())
            return;
        shard.lru_order.erase(it->second);
        shard.lru_idx.erase(it);
    }

    static void lru_touch(Shard& shard, std::int64_t seq_nr)
    {
        lru_remove(shard, seq_nr);
        shard.lru_order.push_back(seq_nr);
        auto tail = shard.lru_order.end();
        --tail;
        shard.lru_idx.emplace(seq_nr, tail);
    }

    static void evict_oldest(Shard& shard)
    {
        if (shard.lru_order.empty())
            return;
        const std::int64_t oldest = shard.lru_order.front();
        shard.lru_order.pop_front();
        shard.lru_idx.erase(oldest);
        shard.snapshots.erase(oldest);
        inc(g_stats.snapshots_dropped);
    }

    std::array<Shard, kNumShards> shards_;
};

inline SnapshotStore g_snapshots;

}  // namespace roctx_recordfn::detail
