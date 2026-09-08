// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "snapshot_store.h"

#include "gsl_assert.h"

#include <utility>

namespace torch_trace_collector::detail
{

synchronized_t<SnapshotStore::Shard>& SnapshotStore::shard_for(const SnapshotKey& key)
{
    return shards_[shard_index(key)];
}

void SnapshotStore::lru_remove(Shard& shard, const SnapshotKey& key)
{
    auto it = shard.lru_idx.find(key);
    if (it == shard.lru_idx.end())
        return;
    shard.lru_order.erase(it->second);
    shard.lru_idx.erase(it);
}

void SnapshotStore::lru_touch(Shard& shard, const SnapshotKey& key)
{
    lru_remove(shard, key);
    shard.lru_order.push_back(key);
    auto tail = shard.lru_order.end();
    --tail;
    shard.lru_idx.emplace(key, tail);
}

void SnapshotStore::evict_oldest(Shard& shard)
{
    Expects(!shard.lru_order.empty());
    const SnapshotKey oldest = shard.lru_order.front();
    shard.lru_order.pop_front();
    shard.lru_idx.erase(oldest);
    shard.snapshots.erase(oldest);
    stats_.snapshots_dropped.fetch_add(1, std::memory_order_relaxed);
}

void SnapshotStore::save(std::int64_t seq_nr, std::uint64_t thread_id, const std::vector<StackEntry>& stack)
{
    const SnapshotKey key = {seq_nr, thread_id};
    shard_for(key).wlock(
        [&](Shard& shard)
        {
            auto it = shard.snapshots.find(key);
            if (it != shard.snapshots.end())
            {
                it->second = stack;
                lru_touch(shard, key);
                stats_.snapshots_overwritten.fetch_add(1, std::memory_order_relaxed);
                stats_.snapshots_saved.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            while (shard.snapshots.size() >= Shard::kMaxEntriesPerShard)
            {
                evict_oldest(shard);
            }
            shard.snapshots.emplace(key, stack);
            lru_touch(shard, key);
            stats_.snapshots_saved.fetch_add(1, std::memory_order_relaxed);
        });
}

bool SnapshotStore::consume(std::int64_t seq_nr, std::uint64_t thread_id, std::vector<StackEntry>* out_stack)
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
            stats_.snapshots_consumed.fetch_add(1, std::memory_order_relaxed);
            return true;
        });
}

std::size_t SnapshotStore::pending() const
{
    std::size_t total = 0;
    for (const auto& guarded_shard : shards_)
    {
        total += guarded_shard.rlock([](const Shard& shard) { return shard.snapshots.size(); });
    }
    return total;
}

void SnapshotStore::clear()
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

}  // namespace torch_trace_collector::detail
