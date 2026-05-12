// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/wall_clock_span_trace.hpp"
#include <cstdint>

#include "core/config.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/thread_info.hpp"

#include <pthread.h>

#include <atomic>
#include <vector>

namespace rocprofsys
{
namespace
{
std::atomic<std::uint64_t> g_next_span_id{ 1 };

struct span_frame
{
    tim::hash_value_t name_hash;
    std::uint64_t     span_id;
};

using span_stack_t = std::vector<span_frame>;

thread_local span_stack_t g_span_stack;

std::uint64_t
record_timestamp_ns()
{
    return static_cast<std::uint64_t>(comp::wall_clock::record());
}

std::uint64_t
thread_seq_value()
{
    std::uint64_t seq = 0;
    const auto&   ti  = thread_info::get(std::this_thread::get_id());
    if(ti.has_value() && ti->index_data.has_value())
        seq = static_cast<std::uint64_t>(ti->index_data->sequent_value);
    return seq;
}

void
emit_begin(std::uint64_t span_id, std::uint64_t parent_span_id, std::string_view label,
           std::string_view category)
{
    trace_cache::get_buffer_storage().store(trace_cache::wall_clock_span_begin_sample(
        span_id, parent_span_id, thread_seq_value(),
        static_cast<std::uint64_t>(pthread_self()), record_timestamp_ns(),
        std::string{ label }, std::string{ category }));
}

void
emit_end(std::uint64_t span_id)
{
    trace_cache::get_buffer_storage().store(trace_cache::wall_clock_span_end_sample(
        span_id, thread_seq_value(), record_timestamp_ns()));
}

}  // namespace

void
wall_clock_span_push_region(tim::hash_value_t name_hash, std::string_view label,
                            std::string_view category)
{
    if(!config::get_use_timemory()) return;
    if(get_state() != State::Active) return;

    const std::uint64_t parent_sid =
        g_span_stack.empty() ? 0ULL : g_span_stack.back().span_id;

    const std::uint64_t span_id = g_next_span_id.fetch_add(1, std::memory_order_relaxed);

    g_span_stack.push_back(span_frame{ name_hash, span_id });

    emit_begin(span_id, parent_sid, label, category);
}

void
wall_clock_span_pop_region(tim::hash_value_t name_hash)
{
    if(!config::get_use_timemory()) return;
    if(get_state() != State::Active) return;

    if(g_span_stack.empty()) return;

    std::size_t idx = static_cast<std::size_t>(-1);
    for(std::size_t i = g_span_stack.size(); i > 0; --i)
    {
        if(g_span_stack[i - 1].name_hash == name_hash)
        {
            idx = i - 1;
            break;
        }
    }

    if(idx == static_cast<std::size_t>(-1)) return;

    while(g_span_stack.size() > idx + 1)
    {
        emit_end(g_span_stack.back().span_id);
        g_span_stack.pop_back();
    }

    emit_end(g_span_stack.back().span_id);
    g_span_stack.pop_back();
}

void
wall_clock_span_flush_open_regions()
{
    if(!config::get_use_timemory()) return;

    while(!g_span_stack.empty())
    {
        emit_end(g_span_stack.back().span_id);
        g_span_stack.pop_back();
    }
}

}  // namespace rocprofsys
