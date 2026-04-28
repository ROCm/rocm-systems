// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/data/backtrace_record.hpp"
#include "sampling/src/sample_ring_buffer.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofsys::sampling::test
{

struct in_memory_offload
{
    template <size_t N>
    void write(int64_t tid, sample_ring_buffer<N>& buf) noexcept
    {
        while(true)
        {
            auto opt = buf.pop();
            if(!opt.has_value()) break;
            m_store[tid].push_back(*opt);
        }
    }

    [[nodiscard]] std::vector<backtrace_record> read(int64_t tid)
    {
        call_log.push_back("read(" + std::to_string(tid) + ")");
        auto it = m_store.find(tid);
        if(it == m_store.end()) return {};
        return it->second;
    }

    [[nodiscard]] std::vector<int64_t> tids() const
    {
        std::vector<int64_t> result;
        result.reserve(m_store.size());
        for(auto const& kv : m_store)
            result.push_back(kv.first);
        return result;
    }

    void reset() noexcept
    {
        call_log.push_back("reset()");
        reset_count++;
        m_store.clear();
    }

    // Direct store injection for post_process tests (avoids ring-buffer detour).
    void inject(int64_t tid, backtrace_record const& rec) { m_store[tid].push_back(rec); }

    std::unordered_map<int64_t, std::vector<backtrace_record>> m_store;
    std::vector<std::string>                                   call_log;
    int                                                        reset_count{ 0 };
};

}  // namespace rocprofsys::sampling::test
