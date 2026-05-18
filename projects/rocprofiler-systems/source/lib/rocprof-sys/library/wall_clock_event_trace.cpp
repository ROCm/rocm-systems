// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/wall_clock_event_trace.hpp"

#include "core/config.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/sample_type.hpp"

#include <timemory/process/threading.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace wall_clock_event_trace
{
namespace
{
struct frame_t
{
    std::uint64_t exec_id = 0;
    std::string   name;
};

struct state_t
{
    std::mutex                                             mtx;
    std::atomic<std::uint64_t>                             next_exec_id{ 1 };
    std::atomic<std::uint64_t>                             next_record_seq{ 1 };
    std::unordered_map<std::int64_t, std::vector<frame_t>> stacks;
    std::unordered_map<std::int64_t, std::uint64_t>        open_pthread_create;
    /// ROCprofiler \c correlation_id::internal for an open callback scope -> wall_clock
    /// \c exec_id (used to parent buffered GPU samples under the submitting API region).
    std::unordered_map<std::uint64_t, std::uint64_t> open_correlation_exec;

    void reset()
    {
        std::scoped_lock<std::mutex> lk{ mtx };
        next_exec_id.store(1, std::memory_order_relaxed);
        next_record_seq.store(1, std::memory_order_relaxed);
        stacks.clear();
        open_pthread_create.clear();
        open_correlation_exec.clear();
    }

    static std::pair<std::uint64_t, std::uint64_t> now_steady_wall_ns()
    {
        const auto sn = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
        const auto wn = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        return { static_cast<std::uint64_t>(sn), static_cast<std::uint64_t>(wn) };
    }

    void emit(std::uint64_t steady_ns, std::uint64_t wall_ns, std::uint64_t thread_id_u,
              trace_cache::wall_clock_scope_event_kind kind, std::uint64_t exec_id,
              std::uint64_t parent_exec_id, std::uint32_t depth,
              std::uint64_t correlation_id, std::string name)
    {
        const std::uint64_t rs = next_record_seq.fetch_add(1, std::memory_order_relaxed);
        trace_cache::get_buffer_storage().store(trace_cache::wall_clock_event_sample{
            steady_ns, wall_ns, thread_id_u, static_cast<std::uint8_t>(kind), exec_id,
            parent_exec_id, depth, correlation_id, rs, std::move(name) });
    }
};

state_t&
get_state()
{
    static state_t* s = new state_t{};
    return *s;
}
}  // namespace

void
session_reset()
{
    get_state().reset();
}

std::int64_t
stack_thread_id()
{
    return tim::threading::get_sys_tid();
}

void
push_region(std::int64_t thread_id, const std::string& name)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    const auto                   tw = state_t::now_steady_wall_ns();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    const std::uint64_t parent_exec = [&]() -> std::uint64_t {
        auto it = st.stacks.find(thread_id);
        if(it == st.stacks.end() || it->second.empty()) return 0;
        return it->second.back().exec_id;
    }();

    const std::uint32_t depth = [&]() -> std::uint32_t {
        auto it = st.stacks.find(thread_id);
        if(it == st.stacks.end()) return 0;
        return static_cast<std::uint32_t>(it->second.size());
    }();

    const std::uint64_t exec_id = st.next_exec_id.fetch_add(1, std::memory_order_relaxed);
    st.emit(tw.first, tw.second, static_cast<std::uint64_t>(thread_id),
            trace_cache::wall_clock_scope_event_kind::enter, exec_id, parent_exec, depth,
            0, name);
    st.stacks[thread_id].push_back(frame_t{ exec_id, name });
}

void
pop_region(std::int64_t thread_id, std::string_view name)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    const auto                   tw = state_t::now_steady_wall_ns();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    auto it = st.stacks.find(thread_id);
    if(it == st.stacks.end() || it->second.empty()) return;

    if(it->second.back().name != name) return;

    const std::uint64_t exec_id = it->second.back().exec_id;
    st.emit(tw.first, tw.second, static_cast<std::uint64_t>(thread_id),
            trace_cache::wall_clock_scope_event_kind::exit, exec_id, 0, 0, 0,
            std::string{});
    it->second.pop_back();
}

void
push_region_ts(std::int64_t thread_id, const std::string& name, std::uint64_t steady_ns,
               std::uint64_t wall_ns, std::uint64_t rocprofiler_correlation_internal)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    const std::uint64_t parent_exec = [&]() -> std::uint64_t {
        auto it = st.stacks.find(thread_id);
        if(it == st.stacks.end() || it->second.empty()) return 0;
        return it->second.back().exec_id;
    }();

    const std::uint32_t depth = [&]() -> std::uint32_t {
        auto it = st.stacks.find(thread_id);
        if(it == st.stacks.end()) return 0;
        return static_cast<std::uint32_t>(it->second.size());
    }();

    const std::uint64_t exec_id = st.next_exec_id.fetch_add(1, std::memory_order_relaxed);
    st.emit(steady_ns, wall_ns, static_cast<std::uint64_t>(thread_id),
            trace_cache::wall_clock_scope_event_kind::enter, exec_id, parent_exec, depth,
            0, name);
    st.stacks[thread_id].push_back(frame_t{ exec_id, name });
    if(rocprofiler_correlation_internal != 0)
        st.open_correlation_exec[rocprofiler_correlation_internal] = exec_id;
}

void
pop_region_ts(std::int64_t thread_id, std::string_view name, std::uint64_t steady_ns,
              std::uint64_t wall_ns, std::uint64_t rocprofiler_correlation_internal)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    auto it = st.stacks.find(thread_id);
    if(it == st.stacks.end() || it->second.empty()) return;

    if(it->second.back().name != name) return;

    const std::uint64_t exec_id = it->second.back().exec_id;
    st.emit(steady_ns, wall_ns, static_cast<std::uint64_t>(thread_id),
            trace_cache::wall_clock_scope_event_kind::exit, exec_id, 0, 0, 0,
            std::string{});
    it->second.pop_back();
    if(rocprofiler_correlation_internal != 0)
        st.open_correlation_exec.erase(rocprofiler_correlation_internal);
}

void
emit_buffered_wall_clock_interval(std::int64_t thread_id, const std::string& name,
                                  std::uint64_t beg_ns, std::uint64_t end_ns,
                                  std::uint64_t rocprofiler_ancestor_internal)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    std::uint64_t parent_exec = 0;
    if(rocprofiler_ancestor_internal != 0)
    {
        auto pit = st.open_correlation_exec.find(rocprofiler_ancestor_internal);
        if(pit != st.open_correlation_exec.end()) parent_exec = pit->second;
    }
    if(parent_exec == 0)
    {
        auto sit = st.stacks.find(thread_id);
        if(sit != st.stacks.end() && !sit->second.empty())
            parent_exec = sit->second.back().exec_id;
    }

    const std::uint32_t depth = [&]() -> std::uint32_t {
        auto sit = st.stacks.find(thread_id);
        if(sit == st.stacks.end()) return 0;
        return static_cast<std::uint32_t>(sit->second.size());
    }();

    const std::uint64_t exec_id = st.next_exec_id.fetch_add(1, std::memory_order_relaxed);
    st.emit(beg_ns, beg_ns, static_cast<std::uint64_t>(thread_id),
            trace_cache::wall_clock_scope_event_kind::enter, exec_id, parent_exec, depth,
            0, name);
    st.emit(end_ns, end_ns, static_cast<std::uint64_t>(thread_id),
            trace_cache::wall_clock_scope_event_kind::exit, exec_id, 0, 0, 0,
            std::string{});
}

void
push_pthread_create(std::int64_t parent_thread_id, const std::string& name)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    const auto                   tw = state_t::now_steady_wall_ns();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    const std::uint64_t parent_exec = [&]() -> std::uint64_t {
        auto ps_it = st.stacks.find(parent_thread_id);
        if(ps_it == st.stacks.end() || ps_it->second.empty()) return 0;
        return ps_it->second.back().exec_id;
    }();

    const std::uint32_t depth = [&]() -> std::uint32_t {
        auto ps_it = st.stacks.find(parent_thread_id);
        if(ps_it == st.stacks.end()) return 0;
        return static_cast<std::uint32_t>(ps_it->second.size());
    }();

    const std::uint64_t exec_id = st.next_exec_id.fetch_add(1, std::memory_order_relaxed);
    st.emit(tw.first, tw.second, static_cast<std::uint64_t>(parent_thread_id),
            trace_cache::wall_clock_scope_event_kind::enter, exec_id, parent_exec, depth,
            0, name);
    st.stacks[parent_thread_id].push_back(frame_t{ exec_id, name });
    st.open_pthread_create[parent_thread_id] = exec_id;
}

void
pop_pthread_create(std::int64_t parent_thread_id, std::string_view name)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    const auto                   tw = state_t::now_steady_wall_ns();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    st.open_pthread_create.erase(parent_thread_id);

    auto ps_it = st.stacks.find(parent_thread_id);
    if(ps_it == st.stacks.end() || ps_it->second.empty()) return;

    if(ps_it->second.back().name != name) return;

    const std::uint64_t exec_id = ps_it->second.back().exec_id;
    st.emit(tw.first, tw.second, static_cast<std::uint64_t>(parent_thread_id),
            trace_cache::wall_clock_scope_event_kind::exit, exec_id, 0, 0, 0,
            std::string{});
    ps_it->second.pop_back();
}

void
push_start_thread(std::int64_t parent_thread_id, std::int64_t child_thread_id,
                  const std::string& name)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    const auto                   tw = state_t::now_steady_wall_ns();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    std::uint64_t parent_exec = 0;
    auto          pit         = st.open_pthread_create.find(parent_thread_id);
    if(pit != st.open_pthread_create.end()) parent_exec = pit->second;

    const std::uint32_t depth = [&]() -> std::uint32_t {
        auto cs_it = st.stacks.find(child_thread_id);
        if(cs_it == st.stacks.end()) return 0;
        return static_cast<std::uint32_t>(cs_it->second.size());
    }();

    const std::uint64_t exec_id = st.next_exec_id.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t corr    = static_cast<std::uint64_t>(parent_thread_id);
    st.emit(tw.first, tw.second, static_cast<std::uint64_t>(child_thread_id),
            trace_cache::wall_clock_scope_event_kind::enter, exec_id, parent_exec, depth,
            corr, name);
    st.stacks[child_thread_id].push_back(frame_t{ exec_id, name });
}

void
pop_start_thread(std::int64_t child_thread_id, std::string_view name)
{
    if(!config::get_use_timemory()) return;

    auto&                        st = get_state();
    const auto                   tw = state_t::now_steady_wall_ns();
    std::scoped_lock<std::mutex> lk{ st.mtx };

    auto cs_it = st.stacks.find(child_thread_id);
    if(cs_it == st.stacks.end() || cs_it->second.empty()) return;

    if(cs_it->second.back().name != name) return;

    const std::uint64_t exec_id = cs_it->second.back().exec_id;
    st.emit(tw.first, tw.second, static_cast<std::uint64_t>(child_thread_id),
            trace_cache::wall_clock_scope_event_kind::exit, exec_id, 0, 0, 0,
            std::string{});
    cs_it->second.pop_back();
}

}  // namespace wall_clock_event_trace
}  // namespace rocprofsys
