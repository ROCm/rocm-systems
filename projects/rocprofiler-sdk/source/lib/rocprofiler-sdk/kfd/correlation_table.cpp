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

#include "lib/rocprofiler-sdk/kfd/correlation_table.hpp"

namespace rocprofiler
{
namespace kfd
{
void
CorrelationTable::insert(const correlation_key& key, const correlation_entry& entry)
{
    // emplace = insert-if-absent (first-writer-wins). See ResultsMap::deposit.
    m_data.wlock([&](auto& map) { map.emplace(key, entry); });
}

std::optional<correlation_entry>
CorrelationTable::take(const correlation_key& key)
{
    return m_data.wlock([&](auto& map) -> std::optional<correlation_entry> {
        auto it = map.find(key);
        if(it == map.end()) return std::nullopt;
        auto entry = it->second;
        map.erase(it);
        return entry;
    });
}

void
CorrelationTable::erase(const correlation_key& key)
{
    m_data.wlock([&](auto& map) { map.erase(key); });
}

size_t
CorrelationTable::size() const
{
    return m_data.rlock([&](const auto& map) { return map.size(); });
}
}  // namespace kfd
}  // namespace rocprofiler
