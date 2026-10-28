// streaming_pipeline.cpp
//
// Implementation of cpu_snap::sampling::internal::streaming_pipeline. See the
// header for the contract.

#include "cpu_snap/sampling/streaming_pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <sys/types.h>
#include <utility>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/dl_module_watcher.hpp"
#include "cpu_snap/sampling/flat_tid_map.hpp"
#include "cpu_snap/sampling/frame_trie.hpp"
#include "cpu_snap/sampling/temporal_diff_filter.hpp"
#include "cpu_snap/sampling/time.hpp"
#include "cpu_snap/core/chunk_framer.hpp"
#include "cpu_snap/core/records.hpp"

namespace cpu_snap::sampling::internal
{

namespace
{

cpu_snap::core::chunk_framer::config
make_framer_cfg(const streaming_pipeline::config& cfg)
{
    return cpu_snap::core::chunk_framer::config{
        .chunk_bytes   = cfg.chunk_bytes,
        .flags         = cfg.flags,
        .pid           = cfg.pid,
        .start_unix_ns = cfg.start_unix_ns,
        .process_name  = cfg.process_name,
    };
}

}  // namespace

streaming_pipeline::streaming_pipeline(config cfg)
: m_cfg(std::move(cfg))
, m_trie{}
, m_diff{ m_trie }
, m_framer(make_framer_cfg(m_cfg), m_cfg.sink)
, m_maps{ m_cfg.clock }
{
    if(!m_cfg.sink)
    {
        throw config_error("streaming_pipeline: sink_fn must not be null");
    }
    if(m_cfg.maps_poll_interval == 0)
    {
        m_cfg.maps_poll_interval = 1;
    }
    if(m_cfg.checkpoint_interval == 0)
    {
        m_cfg.checkpoint_interval = 1;
    }

    m_trie.set_max_nodes(m_cfg.trie_max_nodes);
    if(m_cfg.module_iterator_for_tests)
    {
        m_maps.set_iterator_for_tests(m_cfg.module_iterator_for_tests);
    }

    m_framer.start();

    try
    {
        m_framer.write_split(m_maps.snapshot());
    } catch(...)
    {
        ++m_internal_errors;
    }
}

streaming_pipeline::~streaming_pipeline() { finalize(); }

void
streaming_pipeline::on_sample(pid_t tid, std::uint64_t timestamp_ns,
                              std::span<const std::uint64_t> frames) noexcept
{
    if(m_cfg.total_counter) m_cfg.total_counter->fetch_add(1, std::memory_order_relaxed);
    if(m_finalized) return;

    ++m_sample_calls;
    record_sample(tid, timestamp_ns, frames);
    resync_if_needed();
}

void
streaming_pipeline::on_lost(std::uint64_t lost_count) noexcept
{
    if(m_cfg.lost_counter) m_cfg.lost_counter->fetch_add(lost_count, std::memory_order_relaxed);
    if(m_finalized) return;

    try
    {
        m_lost_total += lost_count;
        while(lost_count > 0)
        {
            const auto delta = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(lost_count,
                                        std::numeric_limits<std::uint32_t>::max()));
            cpu_snap::core::lost_record lr;
            lr.wall_ns          = wall_ns_now();
            lr.tid              = 0;
            lr.lost_count_delta = delta;
            m_framer.write(lr);
            lost_count -= delta;
        }
    } catch(...)
    {
        ++m_internal_errors;
    }
}

void
streaming_pipeline::on_tid_exit(pid_t tid) noexcept
{
    if(m_finalized) return;

    try
    {
        m_diff.forget_tid(tid);
        m_tid_counters.erase(tid);
    } catch(...)
    {
        ++m_internal_errors;
    }
}

void
streaming_pipeline::finalize() noexcept
{
    if(m_finalized) return;
    m_finalized = true;

    m_resync_retry_at = 0;
    resync_if_needed();

    try
    {
        emit_checkpoint();

        auto deltas = m_maps.poll();
        for(auto& delta : deltas)
        {
            m_framer.write(delta);
        }

        cpu_snap::core::eof_record eof;
        eof.wall_ns               = wall_ns_now();
        eof.total_samples_emitted = m_samples_emitted.load(std::memory_order_relaxed);
        m_framer.write(eof);

        m_framer.flush();
    } catch(...)
    {
        ++m_internal_errors;
    }
}

void
streaming_pipeline::abandon() noexcept
{
    m_finalized = true;
}

std::uint64_t streaming_pipeline::samples_emitted() const noexcept
{
    return m_samples_emitted.load(std::memory_order_relaxed);
}

std::uint64_t streaming_pipeline::new_nodes_emitted() const noexcept
{
    return m_new_nodes_emitted.load(std::memory_order_relaxed);
}

std::uint64_t streaming_pipeline::lost_total() const noexcept
{
    return m_lost_total.load(std::memory_order_relaxed);
}

std::uint64_t streaming_pipeline::checkpoints_emitted() const noexcept
{
    return m_checkpoints_emitted.load(std::memory_order_relaxed);
}

std::uint64_t streaming_pipeline::internal_errors() const noexcept
{
    return m_internal_errors.load(std::memory_order_relaxed);
}

std::uint64_t streaming_pipeline::dropped_samples() const noexcept
{
    return m_dropped.load(std::memory_order_relaxed);
}

streaming_pipeline::stats streaming_pipeline::snapshot() const noexcept
{
    return { m_internal_errors.load(std::memory_order_relaxed),
             m_dropped.load(std::memory_order_relaxed) };
}

void
streaming_pipeline::record_sample(pid_t tid, std::uint64_t timestamp_ns,
                                  std::span<const std::uint64_t> frames) noexcept
{
    temporal_diff_filter::diff_result result;
    try
    {
        result = m_diff.diff(tid, frames);
    } catch(...)
    {
        ++m_internal_errors;
        record_dropped_sample(tid);
        return;
    }

    try
    {
        emit_pending_nodes();

        cpu_snap::core::sample_record sample;
        sample.timestamp_ns = timestamp_ns;
        sample.tid          = static_cast<std::uint32_t>(tid);
        sample.leaf_node_id = result.leaf_id;
        m_framer.write(sample);
        const auto emitted = ++m_samples_emitted;
        ++m_tid_counters[tid].samples;

        if(emitted % m_cfg.maps_poll_interval == 0)
        {
            auto deltas = m_maps.poll();
            for(auto& delta : deltas)
            {
                m_framer.write(delta);
            }
        }
        if(emitted % m_cfg.checkpoint_interval == 0)
        {
            emit_checkpoint();
        }
    } catch(...)
    {
        ++m_internal_errors;
    }
}

void
streaming_pipeline::emit_pending_nodes()
{
    for(; m_next_node_id <= m_trie.last_assigned_id(); ++m_next_node_id)
    {
        const auto node = m_trie.node_at(m_next_node_id);
        cpu_snap::core::new_node_record nn;
        nn.node_id   = node.id;
        nn.parent_id = node.parent_id;
        nn.ip        = node.ip;
        m_framer.write(nn);
        ++m_new_nodes_emitted;
    }
}

// Nodes interned by a failed diff() still exist in the trie and in the
// per-tid path, so they must reach the reader before any later sample
// that reuses them.
void
streaming_pipeline::record_dropped_sample(pid_t tid) noexcept
{
    try
    {
        emit_pending_nodes();
        ++m_dropped;
        ++m_tid_counters[tid].lost;
        cpu_snap::core::lost_record lr;
        lr.wall_ns          = wall_ns_now();
        lr.tid              = static_cast<std::uint32_t>(tid);
        lr.lost_count_delta = 1;
        m_framer.write(lr);
    } catch(...)
    {
        ++m_internal_errors;
    }
}

void
streaming_pipeline::note_sink_errors() noexcept
{
    const auto errors = m_framer.sink_errors();
    if(errors == m_seen_sink_errors) return;
    m_seen_sink_errors = errors;
    m_resync_pending   = true;
}

// A discarded chunk may have carried NEW_NODE records, so replay the whole
// trie; the reader accepts identical redefinitions. Retries back off
// exponentially so a persistently failing sink does not replay per sample.
void
streaming_pipeline::resync_if_needed() noexcept
{
    try
    {
        note_sink_errors();
        if(!m_resync_pending || m_sample_calls < m_resync_retry_at) return;

        m_resync_pending = false;
        m_next_node_id   = 1;
        emit_pending_nodes();
        note_sink_errors();

        if(m_resync_pending)
        {
            m_resync_backoff  = std::min(m_resync_backoff * 2, k_max_resync_backoff);
            m_resync_retry_at = m_sample_calls + m_resync_backoff;
        } else
        {
            m_resync_backoff = 1;
        }
    } catch(...)
    {
        ++m_internal_errors;
        m_resync_pending = true;
    }
}

void
streaming_pipeline::emit_checkpoint() noexcept
{
    try
    {
        cpu_snap::core::checkpoint_record cp;
        cp.wall_ns = wall_ns_now();
        cp.entries.reserve(m_tid_counters.size());
        m_tid_counters.for_each(
            [&](pid_t tid, const per_tid_counters& counters) {
                cpu_snap::core::per_tid_checkpoint_entry entry;
                entry.tid     = static_cast<std::uint32_t>(tid);
                entry.leaf_id = m_diff.current_leaf_for_tid(tid);
                entry.samples = counters.samples;
                entry.lost    = counters.lost;
                cp.entries.push_back(entry);
            });
        m_framer.write_split(cp);
        ++m_checkpoints_emitted;
    } catch(...)
    {
        ++m_internal_errors;
    }
}

std::uint64_t
streaming_pipeline::wall_ns_now() const noexcept
{
    return cpu_snap::sampling::utils::clock_now_ns(m_cfg.clock);
}

}  // namespace cpu_snap::sampling::internal
