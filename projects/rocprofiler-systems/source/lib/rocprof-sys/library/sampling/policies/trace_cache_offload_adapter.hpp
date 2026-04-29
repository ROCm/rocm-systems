// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// trace_cache_offload_adapter — production EmitterPolicy.
//
// Drains ring buffer contents into an in-memory per-tid store. No /tmp file I/O.
// Records sit here until shutdown(tid) drives parse+resolve+emit through
// real_trace_cache_sink (the trace_cache emission point owned by the sink
// policy).
//
// This header is intentionally lightweight: it only includes sampling data types
// and logger so it can be included in test TUs without libunwind / AMD-SMI deps.

#include "sampling/data/backtrace_record.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

#include "logger/debug.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocprofsys::sampling
{

class trace_cache_offload_adapter
{
public:
    // EmitterPolicy::write — drain ring buffer for tid into the in-memory store.
    // FatalErrorPolicy is wired here for interface parity; write cannot fail in
    // this in-memory implementation.
    template <size_t N, class FatalErrorPolicy>
    void write(int64_t tid, sample_ring_buffer<N>& buf, FatalErrorPolicy& /*fatal*/)
    {
        std::vector<backtrace_record> batch;
        batch.reserve(buf.count());
        while(auto opt = buf.pop())
            batch.push_back(*opt);

        if(batch.empty()) return;

        LOG_DEBUG("[trace_cache_offload_adapter] write: tid={} records={}", tid,
                  batch.size());

        std::lock_guard<std::mutex> lk{ m_mutex };
        auto&                       store = m_store[tid];
        store.insert(store.end(), batch.begin(), batch.end());
    }

    // EmitterPolicy::read — return all records for tid (called from post_process()).
    [[nodiscard]] std::vector<backtrace_record> read(int64_t tid)
    {
        std::lock_guard<std::mutex> lk{ m_mutex };
        auto                        it = m_store.find(tid);
        if(it == m_store.end()) return {};
        return it->second;
    }

    // EmitterPolicy::tids — return all tids that have stored records.
    [[nodiscard]] std::vector<int64_t> tids() const
    {
        std::lock_guard<std::mutex> lk{ m_mutex };
        std::vector<int64_t>        result;
        result.reserve(m_store.size());
        for(auto const& kv : m_store)
            result.push_back(kv.first);
        return result;
    }

    // EmitterPolicy::reset — clear all stored records.
    // noexcept: tests/unit/trace_cache_offload_adapter_test.cpp asserts this contract.
    // std::mutex::lock can theoretically throw std::system_error, but not on a properly
    // constructed POSIX mutex; the contract reflects that real-world guarantee.
    void reset() noexcept
    {
        std::lock_guard<std::mutex> lk{ m_mutex };
        m_store.clear();
    }

    // Direct store injection seam used by post_process tests (avoids ring-buffer detour).
    void inject(int64_t tid, backtrace_record const& rec)
    {
        std::lock_guard<std::mutex> lk{ m_mutex };
        m_store[tid].push_back(rec);
    }

    // Remove all records for tid after emit_resolved_to_trace_cache() has processed them.
    // Prevents double-emission when post_process() iterates offload_.tids() (Variant 2).
    // noexcept: tests/unit/emit_resolved_hook_test.cpp asserts this contract (see also
    // reset()).
    void erase(int64_t tid) noexcept
    {
        std::lock_guard<std::mutex> lk{ m_mutex };
        m_store.erase(tid);
    }

private:
    mutable std::mutex                                         m_mutex;
    std::unordered_map<int64_t, std::vector<backtrace_record>> m_store;
};

}  // namespace rocprofsys::sampling
