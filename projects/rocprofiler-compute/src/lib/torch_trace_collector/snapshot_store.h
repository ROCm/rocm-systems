// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"
#include "stats.h"
#include "synchronized.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

namespace torch_trace_collector::detail
{

// Forward-stack snapshot key: sequence number and RecordFunction thread id.
struct SnapshotKey
{
    std::int64_t  seq_nr    = 0;
    std::uint64_t thread_id = 0;

    bool operator==(const SnapshotKey& other) const noexcept
    {
        return seq_nr == other.seq_nr && thread_id == other.thread_id;
    }
};

constexpr std::size_t hash_snapshot_key(const SnapshotKey& key) noexcept
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
    static constexpr std::size_t kNumShards          = 64;
    static constexpr std::size_t kMaxEntriesPerShard = 10000;

    explicit SnapshotStore(Stats& stats)
        : stats_(stats)
    {
    }

    static constexpr std::size_t shard_index(const SnapshotKey& key) noexcept
    {
        return hash_snapshot_key(key) % kNumShards;
    }

    void save(std::int64_t seq_nr, std::uint64_t thread_id, const std::vector<StackEntry>& stack);
    bool consume(std::int64_t seq_nr, std::uint64_t thread_id, std::vector<StackEntry>* out_stack);
    std::size_t pending() const;
    void        clear();

private:
    struct Shard
    {
        std::unordered_map<SnapshotKey, std::vector<StackEntry>>          snapshots;
        std::list<SnapshotKey>                                            lru_order;
        std::unordered_map<SnapshotKey, std::list<SnapshotKey>::iterator> lru_idx;
    };

    synchronized_t<Shard>& shard_for(const SnapshotKey& key);
    static void            lru_remove(Shard& shard, const SnapshotKey& key);
    static void            lru_touch(Shard& shard, const SnapshotKey& key);
    void                   evict_oldest(Shard& shard);

    Stats&                                        stats_;
    std::array<synchronized_t<Shard>, kNumShards> shards_;
};

}  // namespace torch_trace_collector::detail
