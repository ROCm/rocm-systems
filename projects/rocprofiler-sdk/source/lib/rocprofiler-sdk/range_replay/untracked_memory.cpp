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

#include "lib/rocprofiler-sdk/range_replay/untracked_memory.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <fmt/format.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

namespace rocprofiler
{
namespace range_replay
{
namespace untracked_memory
{
namespace
{
// Anything below the first page or above the canonical 47-bit user range is a scalar, not an
// address a kernel can dereference.
constexpr uint64_t min_candidate = 0x10000;
constexpr uint64_t max_candidate = 0x00007fffffffffffULL;

// Writable mappings of this process that are ordinary host memory. Device mappings through
// /dev/kfd or /dev/dri are left out: HSA classifies the ones that matter.
std::vector<address_span_t>
read_writable_host_mappings()
{
    auto out  = std::vector<address_span_t>{};
    auto file = std::ifstream{"/proc/self/maps"};
    auto line = std::string{};
    while(std::getline(file, line))
    {
        unsigned long long begin    = 0;
        unsigned long long end      = 0;
        char               perms[8] = {};
        if(std::sscanf(line.c_str(), "%llx-%llx %7s", &begin, &end, perms) != 3) continue;
        if(perms[0] != 'r' || perms[1] != 'w') continue;
        if(line.find("/dev/kfd") != std::string::npos || line.find("/dev/dri") != std::string::npos)
            continue;
        out.push_back({begin, end});
    }
    std::sort(
        out.begin(), out.end(), [](const auto& l, const auto& r) { return l.begin < r.begin; });
    return out;
}

bool
in_spans(const std::vector<address_span_t>& sorted, uint64_t address)
{
    auto itr = std::upper_bound(
        sorted.begin(), sorted.end(), address, [](uint64_t a, const address_span_t& s) {
            return a < s.begin;
        });
    if(itr == sorted.begin()) return false;
    --itr;
    return address >= itr->begin && address < itr->end;
}

// The mode the runtime bound, which HSA_XNACK alone does not determine: XNACK can already be on
// without it, and the request to turn it on can fail.
bool
xnack_enabled()
{
    auto* core = hsa::get_core_table();
    auto  on   = false;
    return core != nullptr && core->hsa_system_get_info_fn != nullptr &&
           core->hsa_system_get_info_fn(HSA_AMD_SYSTEM_INFO_XNACK_ENABLED, &on) ==
               HSA_STATUS_SUCCESS &&
           on;
}

pointer_facts_t::type_t
convert_type(hsa_amd_pointer_type_t type)
{
    switch(type)
    {
        case HSA_EXT_POINTER_TYPE_UNKNOWN: return pointer_facts_t::type_t::unknown;
        case HSA_EXT_POINTER_TYPE_HSA: return pointer_facts_t::type_t::hsa;
        case HSA_EXT_POINTER_TYPE_LOCKED: return pointer_facts_t::type_t::locked;
        case HSA_EXT_POINTER_TYPE_RESERVED_ADDR: return pointer_facts_t::type_t::reserved;
        case HSA_EXT_POINTER_TYPE_HSA_VMEM: return pointer_facts_t::type_t::vmem;
        default: return pointer_facts_t::type_t::other;
    }
}
}  // namespace

const char*
to_string(kind_t kind)
{
    switch(kind)
    {
        case kind_t::host_pinned: return "host-pinned memory (hipHostMalloc)";
        case kind_t::managed: return "managed memory (hipMallocManaged)";
        case kind_t::virtual_mem: return "a virtual-memory allocation (hipMallocAsync, hipMemMap)";
        case kind_t::device_other:
            return "device memory the snapshot does not track (fine-grained, uncached, peer)";
        case kind_t::locked_host: return "locked host memory (hipHostRegister)";
        case kind_t::host_pageable: return "pageable host memory";
    }
    return "untracked memory";
}

pointer_facts_fn_t
make_hsa_pointer_facts()
{
    auto mappings = std::make_shared<std::optional<std::vector<address_span_t>>>();

    return [mappings](uint64_t address) {
        auto facts = pointer_facts_t{};

        auto* ext  = hsa::get_amd_ext_table();
        auto* core = hsa::get_core_table();
        if(ext != nullptr && ext->hsa_amd_pointer_info_fn != nullptr)
        {
            auto info = hsa_amd_pointer_info_t{};
            info.size = sizeof(info);
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            auto* ptr = reinterpret_cast<void*>(address);
            if(ext->hsa_amd_pointer_info_fn(ptr, &info, nullptr, nullptr, nullptr) ==
               HSA_STATUS_SUCCESS)
            {
                facts.type = convert_type(info.type);
                if(facts.type == pointer_facts_t::type_t::hsa && info.agentOwner.handle != 0 &&
                   core != nullptr && core->hsa_agent_get_info_fn != nullptr)
                {
                    auto device = hsa_device_type_t{};
                    if(core->hsa_agent_get_info_fn(
                           info.agentOwner, HSA_AGENT_INFO_DEVICE, &device) == HSA_STATUS_SUCCESS)
                        facts.owner_is_cpu = (device == HSA_DEVICE_TYPE_CPU);
                }
            }
        }

        // Without XNACK a kernel cannot fault in pageable host memory, and host pointers carried
        // by value in a struct argument would read as false hits. With XNACK the GPU can touch any
        // host address.
        if(facts.type == pointer_facts_t::type_t::unknown && xnack_enabled())
        {
            if(!mappings->has_value()) *mappings = read_writable_host_mappings();
            facts.in_writable_host_mapping = in_spans(**mappings, address);
        }
        return facts;
    };
}

std::vector<address_span_t>
covered_spans(const kernel_replay::memory_snapshot::device_snapshot_t& snapshot)
{
    auto out = std::vector<address_span_t>{};
    out.reserve(snapshot.blocks.size());
    for(const auto& block : snapshot.blocks)
    {
        const auto begin = reinterpret_cast<uint64_t>(block.gpu_addr);
        out.push_back({begin, begin + block.host_copy.size()});
    }
    std::sort(
        out.begin(), out.end(), [](const auto& l, const auto& r) { return l.begin < r.begin; });
    return out;
}

std::optional<kind_t>
classify_pointer(const pointer_facts_t& facts)
{
    using type_t = pointer_facts_t::type_t;
    switch(facts.type)
    {
        case type_t::hsa: return facts.owner_is_cpu ? kind_t::host_pinned : kind_t::device_other;
        case type_t::reserved: return kind_t::managed;
        case type_t::vmem: return kind_t::virtual_mem;
        case type_t::locked: return kind_t::locked_host;
        case type_t::other: return kind_t::device_other;
        case type_t::unknown:
            if(facts.in_writable_host_mapping) return kind_t::host_pageable;
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<hit_t>
find_untracked(const uint8_t*                     kernarg,
               size_t                             size,
               const std::vector<address_span_t>& covered,
               const pointer_facts_fn_t&          facts,
               std::unordered_set<uint64_t>&      seen)
{
    if(kernarg == nullptr) return std::nullopt;

    const auto limit = size >= implicit_arg_block_bytes ? size - implicit_arg_block_bytes : size;
    for(size_t off = 0; off + sizeof(uint64_t) <= limit; off += sizeof(uint64_t))
    {
        auto value = uint64_t{0};
        std::memcpy(&value, kernarg + off, sizeof(value));
        if(value < min_candidate || value > max_candidate) continue;
        if(in_spans(covered, value)) continue;
        if(!seen.insert(value).second) continue;

        if(const auto kind = classify_pointer(facts(value)); kind)
            return hit_t{off, value, *kind};
    }
    return std::nullopt;
}
}  // namespace untracked_memory

void
decline_untracked(range_context_t& ctx)
{
    if(!ctx.snapshot_taken || !ctx.record.eligible()) return;
    if(ctx.untracked_scanned >= ctx.record.dispatch_count()) return;
    decline_untracked(ctx, untracked_memory::make_hsa_pointer_facts());
}

void
decline_untracked(range_context_t& ctx, const untracked_memory::pointer_facts_fn_t& facts)
{
    namespace um = untracked_memory;
    if(!ctx.snapshot_taken || !ctx.record.eligible()) return;

    const auto& dispatches = ctx.record.dispatches();
    if(ctx.untracked_scanned == 0) ctx.covered = um::covered_spans(ctx.snapshot);

    for(; ctx.untracked_scanned < dispatches.size(); ++ctx.untracked_scanned)
    {
        const auto& kernarg = dispatches[ctx.untracked_scanned].kernarg;
        const auto  hit =
            um::find_untracked(kernarg.data(), kernarg.size(), ctx.covered, facts, ctx.seen);
        if(!hit) continue;

        ROCP_WARNING << fmt::format("range replay: range {}: dispatch {} kernarg offset {} points "
                                    "at 0x{:x}, {}, which the snapshot does not restore; the range "
                                    "is not replayed",
                                    ctx.record.range_id(),
                                    ctx.untracked_scanned,
                                    hit->kernarg_offset,
                                    hit->address,
                                    um::to_string(hit->kind));
        ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
        return;
    }
}
}  // namespace range_replay
}  // namespace rocprofiler
