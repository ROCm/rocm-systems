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

#include "lib/rocprofiler-sdk/range_replay/digest.hpp"

#include <algorithm>

namespace rocprofiler
{
namespace range_replay
{
namespace digest
{
uint64_t
hash_bytes(const void* data, size_t size)
{
    constexpr uint64_t offset_basis = 14695981039346656037ULL;
    constexpr uint64_t prime        = 1099511628211ULL;

    if(data == nullptr) return offset_basis;

    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t    hash  = offset_basis;
    for(size_t i = 0; i < size; ++i)
    {
        hash ^= bytes[i];
        hash *= prime;
    }
    return hash;
}

size_t
count_divergent(const region_digests_t& lhs, const region_digests_t& rhs)
{
    if(lhs.size() != rhs.size()) return std::max(lhs.size(), rhs.size());

    size_t divergent = 0;
    for(size_t i = 0; i < lhs.size(); ++i)
        if(lhs[i] != rhs[i]) ++divergent;
    return divergent;
}
}  // namespace digest
}  // namespace range_replay
}  // namespace rocprofiler
