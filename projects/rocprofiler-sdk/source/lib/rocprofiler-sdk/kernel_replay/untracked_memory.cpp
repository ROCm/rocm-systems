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

#include "lib/rocprofiler-sdk/kernel_replay/untracked_memory.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <fmt/format.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace kernel_replay
{
namespace untracked_memory
{
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
        // vvar, vdso and vsyscall hold scalars and host-only pointers, not memory a kernel is
        // handed as data, and a by-value scalar whose bit pattern lands in their range would read
        // as a false hit. The thread stacks are kept: a kernel argument may well point at a host
        // variable, and missing that is the failure this module exists to catch.
        if(line.find("[vvar") != std::string::npos || line.find("[vdso") != std::string::npos ||
           line.find("[vsyscall") != std::string::npos)
            continue;
        out.push_back({begin, end});
    }
    std::sort(
        out.begin(), out.end(), [](const auto& l, const auto& r) { return l.begin < r.begin; });
    return out;
}

namespace
{
// Anything below the first page or above the canonical 47-bit user range is a scalar, not an
// address a kernel can dereference.
constexpr uint64_t min_candidate = 0x10000;
constexpr uint64_t max_candidate = 0x00007fffffffffffULL;

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

pointer_facts_t::type_t
convert_type(hsa_amd_pointer_type_t type)
{
    switch(type)
    {
        case HSA_EXT_POINTER_TYPE_UNKNOWN: return pointer_facts_t::type_t::unknown;
        case HSA_EXT_POINTER_TYPE_HSA: return pointer_facts_t::type_t::hsa;
        case HSA_EXT_POINTER_TYPE_LOCKED:
            return pointer_facts_t::type_t::locked;
// hsa_amd_pointer_info gained these two types at HSA AMD interface 1.12. Older
// installed headers (e.g. the 6.2 release-compatibility builds) do not declare them
// at all, so they are compiled out there; an unnamed type already classifies as
// other, which is the safe degraded behavior because such a pointer is then never
// treated as memory the snapshot restores.
#if defined(HSA_AMD_INTERFACE_VERSION_MAJOR) &&                                                    \
    (HSA_AMD_INTERFACE_VERSION_MAJOR > 1 ||                                                        \
     (HSA_AMD_INTERFACE_VERSION_MAJOR == 1 && HSA_AMD_INTERFACE_VERSION_MINOR >= 12))
        case HSA_EXT_POINTER_TYPE_RESERVED_ADDR: return pointer_facts_t::type_t::reserved;
        case HSA_EXT_POINTER_TYPE_HSA_VMEM: return pointer_facts_t::type_t::vmem;
#endif
        default: return pointer_facts_t::type_t::other;
    }
}

bool
default_copy(void* dst, const void* src, size_t size)
{
    auto* core = hsa::get_core_table();
    return core != nullptr && core->hsa_memory_copy_fn != nullptr &&
           core->hsa_memory_copy_fn(dst, src, size) == HSA_STATUS_SUCCESS;
}

void*
as_pointer(uint64_t address)
{
    return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

// Bytes of `offset`, `size` rounded out to whole aligned words inside the segment, so that a word
// overlapping the range is read. Returns false when the range holds no whole word.
bool
word_range(size_t segment_size, uint64_t offset, uint64_t size, size_t& begin, size_t& end)
{
    constexpr auto word = sizeof(uint64_t);
    if(offset >= segment_size) return false;
    begin = static_cast<size_t>(offset) & ~(word - 1);
    end   = std::min(static_cast<size_t>(offset) + size + (word - 1), segment_size) & ~(word - 1);
    return begin + word <= end;
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
        case kind_t::device_other: return "device memory the snapshot does not track";
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
            if(ext->hsa_amd_pointer_info_fn(
                   as_pointer(address), &info, nullptr, nullptr, nullptr) == HSA_STATUS_SUCCESS)
            {
                facts.type = convert_type(info.type);
                facts.fine_grained =
                    (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) != 0;
                facts.agent_base = reinterpret_cast<uint64_t>(info.agentBaseAddress);
                facts.size       = info.sizeInBytes;
                facts.owner      = info.agentOwner.handle;
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

        // HSA reports neither pageable host memory nor hipHostRegister'd memory with an owner or
        // extent, so fall back to the process's writable mappings. This is not gated on XNACK:
        // hipHostRegister maps memory into the GPU address space even with XNACK off, so a kernel
        // can write it. vvar, vdso and vsyscall are excluded because a by-value scalar's bit
        // pattern can land in them.
        if(facts.type == pointer_facts_t::type_t::unknown)
        {
            if(!mappings->has_value()) *mappings = read_writable_host_mappings();
            facts.in_writable_host_mapping = in_spans(**mappings, address);
        }
        return facts;
    };
}

bool
is_idempotent_runtime_kernel(std::string_view kernel_name)
{
    // The code object registry names a kernel after its descriptor, which carries a ".kd" suffix.
    constexpr std::string_view descriptor_suffix{".kd"};
    if(kernel_name.size() > descriptor_suffix.size() &&
       kernel_name.compare(kernel_name.size() - descriptor_suffix.size(),
                           descriptor_suffix.size(),
                           descriptor_suffix) == 0)
        kernel_name.remove_suffix(descriptor_suffix.size());

    // ROCclr's blit kernels. They are handed HIP's staging and scratch memory rather than
    // application data, and writing the same bytes again in a later pass changes nothing, so they
    // replay without the checks. The runtime's other kernels are deliberately absent:
    // streamOpsIncrement, streamOpsWait, scheduler, initHeap and friends change state every time
    // they run, and a kernel not listed here takes the strict path.
    constexpr auto kernels = std::array<std::string_view, 13>{
        "__amd_rocclr_copyBuffer",
        "__amd_rocclr_copyBufferAligned",
        "__amd_rocclr_copyBufferRect",
        "__amd_rocclr_copyBufferRectAligned",
        "__amd_rocclr_copyBufferBatch",
        "__amd_rocclr_copyImage",
        "__amd_rocclr_copyImage1DA",
        "__amd_rocclr_copyImageToBuffer",
        "__amd_rocclr_copyBufferToImage",
        "__amd_rocclr_fillImage",
        "__amd_rocclr_fillBufferAligned",
        "__amd_rocclr_fillBufferAligned2D",
        "__amd_rocclr_fillBufferUnAligned",
    };
    return std::find(kernels.begin(), kernels.end(), kernel_name) != kernels.end();
}

std::vector<address_span_t>
covered_spans(const memory_snapshot::device_snapshot_t& snapshot)
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

region_t
region_of(kind_t kind, const pointer_facts_t& facts)
{
    auto out = region_t{};
    out.kind = kind;
    if(kind == kind_t::device_other && facts.type == pointer_facts_t::type_t::hsa &&
       facts.fine_grained && facts.agent_base != 0 && facts.size != 0)
    {
        out.base  = facts.agent_base;
        out.size  = facts.size;
        out.owner = facts.owner;
    }
    return out;
}

std::vector<hit_t>
scan_kernarg(const uint8_t*                                    kernarg,
             size_t                                            segment_size,
             const std::vector<code_object::kernarg_region_t>* pointer_regions,
             const std::vector<address_span_t>&                covered,
             const pointer_facts_fn_t&                         facts,
             std::unordered_set<uint64_t>&                     seen)
{
    auto hits = std::vector<hit_t>{};
    if(kernarg == nullptr) return hits;

    const auto whole = std::vector<code_object::kernarg_region_t>{
        code_object::kernarg_region_t{0u,
                                      static_cast<uint32_t>(std::min<size_t>(
                                          segment_size, std::numeric_limits<uint32_t>::max()))}};
    const auto& regions = pointer_regions != nullptr ? *pointer_regions : whole;

    for(const auto& region : regions)
    {
        auto begin = size_t{};
        auto end   = size_t{};
        if(!word_range(segment_size, region.offset, region.size, begin, end)) continue;
        for(size_t off = begin; off < end; off += sizeof(uint64_t))
        {
            auto value = uint64_t{0};
            std::memcpy(&value, kernarg + off, sizeof(value));
            if(value < min_candidate || value > max_candidate) continue;
            if(in_spans(covered, value)) continue;
            if(!seen.insert(value).second) continue;

            const auto f = facts(value);
            if(const auto kind = classify_pointer(f); kind)
                hits.push_back(hit_t{off, value, region_of(*kind, f)});
        }
    }
    return hits;
}

region_set_t::region_set_t()
: m_copy{default_copy}
{}

region_set_t::region_set_t(copy_fn_t copy)
: m_copy{std::move(copy)}
{}

region_set_t::save_status_t
region_set_t::save(const region_t& region)
{
    if(region.size == 0) return save_status_t::not_restorable;
    if(contains(region.base)) return save_status_t::already_saved;
    if(region.size > max_total_bytes - m_bytes) return save_status_t::too_large;

    auto saved   = saved_t{};
    saved.region = region;
    try
    {
        saved.data.resize(region.size);
    } catch(const std::bad_alloc&)
    {
        return save_status_t::copy_failed;
    }
    if(!m_copy(saved.data.data(), as_pointer(region.base), region.size))
        return save_status_t::copy_failed;

    try
    {
        m_saved.emplace_back(std::move(saved));
    } catch(const std::bad_alloc&)
    {
        return save_status_t::copy_failed;
    }
    m_bytes += region.size;
    return save_status_t::saved;
}

bool
region_set_t::contains(uint64_t address) const
{
    return std::any_of(m_saved.begin(), m_saved.end(), [address](const saved_t& s) {
        return address >= s.region.base && address < s.region.base + s.region.size;
    });
}

bool
region_set_t::restore() const
{
    return std::all_of(m_saved.begin(), m_saved.end(), [this](const saved_t& s) {
        return m_copy(as_pointer(s.region.base), s.data.data(), s.data.size());
    });
}

bool
save_dispatch_memory(region_set_t&                                     saved,
                     const uint8_t*                                    kernarg,
                     size_t                                            segment_size,
                     const std::vector<code_object::kernarg_region_t>* pointer_regions,
                     const memory_snapshot::device_snapshot_t&         snapshot,
                     hsa_agent_t                                       agent,
                     const pointer_facts_fn_t&                         facts)
{
    auto seen = std::unordered_set<uint64_t>{};
    for(const auto& hit :
        scan_kernarg(kernarg, segment_size, pointer_regions, covered_spans(snapshot), facts, seen))
    {
        const auto  where   = fmt::format("kernel replay: kernarg offset {} points at 0x{:x}, {}",
                                       hit.kernarg_offset,
                                       hit.address,
                                       to_string(hit.region.kind));
        const bool  foreign = hit.region.size != 0 && hit.region.owner != agent.handle;
        const char* why     = nullptr;
        switch(foreign ? region_set_t::save_status_t::not_restorable : saved.save(hit.region))
        {
            case region_set_t::save_status_t::saved:
                ROCP_INFO << fmt::format(
                    "{}; saved {} bytes at 0x{:x}", where, hit.region.size, hit.region.base);
                continue;
            case region_set_t::save_status_t::already_saved: continue;
            case region_set_t::save_status_t::not_restorable:
                if(foreign)
                    why = "it belongs to another agent";
                else if(hit.region.kind == kind_t::virtual_mem ||
                        hit.region.kind == kind_t::host_pageable)
                    why = "its extent is not known";
                else
                    why = "it is not device memory of this agent";
                break;
            case region_set_t::save_status_t::too_large:
                why = "saving it would exceed the per-dispatch limit";
                break;
            case region_set_t::save_status_t::copy_failed: why = "copying it failed"; break;
        }
        LOG_FIRST_N(WARNING, 1) << fmt::format(
            "{}, which the snapshot does not restore and {}; running this dispatch once without "
            "replay",
            where,
            why);
        return false;
    }
    return true;
}

bool
save_dispatch_memory(region_set_t&                             saved,
                     const hsa_kernel_dispatch_packet_t&       packet,
                     const memory_snapshot::device_snapshot_t& snapshot,
                     hsa_agent_t                               agent)
{
    const auto kernel = code_object::locate_kernel(packet.kernel_object);
    if(!kernel)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel replay: the dispatched kernel is not in the code "
                                   "object registry, so the memory its arguments point at cannot "
                                   "be checked; running this dispatch once without replay";
        return false;
    }

    if(kernel->name != nullptr && is_idempotent_runtime_kernel(*kernel->name)) return true;

    // Without metadata every word of the segment is a candidate, including the runtime's hidden
    // arguments -- its hostcall and printf buffers, its heap -- which point at memory the snapshot
    // does not restore, so such a dispatch may run once for that alone.
    const auto* regions = kernel->pointer_regions.get();
    if(regions == nullptr)
        LOG_FIRST_N(WARNING, 1) << fmt::format(
            "kernel replay: no argument metadata for {} (kernarg segment {} bytes), so every word "
            "of it is checked for pointers including the runtime's hidden arguments; a dispatch of "
            "this kernel may run once without replay",
            kernel->name != nullptr ? kernel->name->c_str() : "<unnamed kernel>",
            kernel->segment_size);

    return save_dispatch_memory(saved,
                                static_cast<const uint8_t*>(packet.kernarg_address),
                                kernel->segment_size,
                                regions,
                                snapshot,
                                agent,
                                make_hsa_pointer_facts());
}
}  // namespace untracked_memory
}  // namespace kernel_replay
}  // namespace rocprofiler
