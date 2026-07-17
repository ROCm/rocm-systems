// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/kfd/results_map.hpp"

namespace rocprofiler
{
namespace kfd
{
void
ResultsMap::deposit(const correlation_key& key, const kfd_timing_result& result)
{
    // emplace = insert-if-absent (first-writer-wins). A correlation_key is
    // unique per in-flight dispatch, so a collision should not occur; if one does,
    // keep the first (real) pairing rather than overwrite with a later spurious one.
    m_data.wlock([&](auto& map) { map.emplace(key, result); });
}

std::optional<kfd_timing_result>
ResultsMap::take(const correlation_key& key)
{
    return m_data.wlock([&](auto& map) -> std::optional<kfd_timing_result> {
        auto it = map.find(key);
        if(it == map.end()) return std::nullopt;
        auto result = it->second;
        map.erase(it);
        return result;
    });
}

size_t
ResultsMap::evict_stale(uint64_t now_ns, uint64_t max_age_ns)
{
    return m_data.wlock([&](auto& map) -> size_t {
        size_t evicted = 0;
        for(auto it = map.begin(); it != map.end();)
        {
            // Guard against a deposit stamped slightly in the future relative to
            // now_ns (clock skew): only evict when now_ns is strictly ahead.
            if(now_ns > it->second.deposited_at_ns &&
               (now_ns - it->second.deposited_at_ns) > max_age_ns)
            {
                it = map.erase(it);
                ++evicted;
            }
            else
            {
                ++it;
            }
        }
        return evicted;
    });
}

size_t
ResultsMap::size() const
{
    return m_data.rlock([&](const auto& map) { return map.size(); });
}
}  // namespace kfd
}  // namespace rocprofiler
