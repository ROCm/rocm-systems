// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk-att/att_decoder.h"
#include "lib/rocprofiler-sdk-att/att_lib_wrapper.hpp"
#include "lib/rocprofiler-sdk-att/code.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace att_wrapper
{
struct LineWaitcnt
{
    int              line_number{0};
    std::vector<int> dependencies{};
};

struct WaitcntList
{
    using isa_map_t = std::map<pcinfo_t, std::unique_ptr<CodeLine>>;
    using wave_t    = att_wave_data_t;

    WaitcntList() = default;

    static const WaitcntList& Get(int gfxip, const wave_t& wave, isa_map_t& isa_map)
    {
        auto it = _cache.find(wave.traceID);
        if(it != _cache.end()) return *it->second;

        auto ptr = std::make_unique<WaitcntList>();

        if(gfxip == 9)
            ptr->mem_unroll = gfx9_construct(wave, isa_map);
        else if(gfxip == 10 || gfxip == 11)
            ptr->mem_unroll = gfx10_construct(wave, isa_map);
        else if(gfxip == 12)
            ptr->mem_unroll = gfx12_construct(wave, isa_map);
        else
            throw std::runtime_error("Invalid gfxip: " + std::to_string(gfxip));

        return *_cache.emplace(wave.traceID, std::move(ptr)).first->second;
    }

    static std::vector<LineWaitcnt> gfx9_construct(const wave_t& wave, isa_map_t& isa_map);
    static std::vector<LineWaitcnt> gfx10_construct(const wave_t& wave, isa_map_t& isa_map);
    static std::vector<LineWaitcnt> gfx12_construct(const wave_t& wave, isa_map_t& isa_map);

    std::vector<LineWaitcnt> mem_unroll{};

private:
    static std::map<size_t, std::unique_ptr<WaitcntList>> _cache;
};

class MemoryCounter
{
public:
    enum Ordering
    {
        MEMORY_SEQUENTIAL = 0,
        MEMORY_PARALLEL
    };

    MemoryCounter(std::string _name)
    : name(_name)
    {}

    int64_t extract_waitcnt(const std::string& str) const;

    std::vector<int> join_and_reset(int64_t offset, std::vector<int>& flats);

    std::optional<std::vector<int>> handle_mem_op(const std::string& inst,
                                                  std::vector<int>&  flat_list);

    const std::string name;
    Ordering          order = Ordering::MEMORY_SEQUENTIAL;
    std::vector<int>  list{};
};

}  // namespace att_wrapper
}  // namespace rocprofiler
