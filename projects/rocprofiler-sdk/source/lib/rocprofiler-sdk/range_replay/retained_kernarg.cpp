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

#include "lib/rocprofiler-sdk/range_replay/retained_kernarg.hpp"
#include "lib/common/synchronized.hpp"

#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace range_replay
{
namespace
{
using retained_map_t = std::unordered_map<rocprofiler_agent_id_t, kernarg_block_t>;

// Leaked on purpose: freeing a block calls into the HSA runtime, which may already be torn down
// when static destructors run.
common::Synchronized<retained_map_t>&
retained_blocks()
{
    static auto* blocks = new common::Synchronized<retained_map_t>{};
    return *blocks;
}
}  // namespace

kernarg_block_t
take_retained_kernarg_block(rocprofiler_agent_id_t agent)
{
    return retained_blocks().wlock([agent](retained_map_t& blocks) {
        auto itr = blocks.find(agent);
        if(itr == blocks.end()) return kernarg_block_t{};
        auto block = itr->second;
        blocks.erase(itr);
        return block;
    });
}

void
retain_kernarg_block(rocprofiler_agent_id_t agent, kernarg_block_t block)
{
    auto displaced = retained_blocks().wlock(
        [agent, block](retained_map_t& blocks) { return std::exchange(blocks[agent], block); });
    free_kernarg_block(displaced);
}

void
free_kernarg_block(kernarg_block_t& block)
{
    if(block.base != nullptr && block.free_fn != nullptr) block.free_fn(block.base);
    block = kernarg_block_t{};
}

size_t
retained_kernarg_bytes(rocprofiler_agent_id_t agent)
{
    return retained_blocks().rlock([agent](const retained_map_t& blocks) {
        auto itr = blocks.find(agent);
        return (itr == blocks.end()) ? size_t{0} : itr->second.capacity;
    });
}
}  // namespace range_replay
}  // namespace rocprofiler
