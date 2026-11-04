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
host_mappings_t
read_writable_host_mappings()
{
    auto file = std::ifstream{"/proc/self/maps"};
    return read_writable_host_mappings(file);
}

host_mappings_t
read_writable_host_mappings(std::istream& file)
{
    auto out  = host_mappings_t{};
    auto line = std::string{};
    while(std::getline(file, line))
    {
        unsigned long long begin    = 0;
        unsigned long long end      = 0;
        char               perms[8] = {};
        if(std::sscanf(line.c_str(), "%llx-%llx %7s", &begin, &end, perms) != 3) continue;
        // Write permission is enough. A PROT_WRITE mapping is `-w-` here, and this check never
        // reads the page: it only asks whether a pointer lands in host memory the snapshot misses.
        if(perms[1] != 'w') continue;
        if(line.find("/dev/kfd") != std::string::npos || line.find("/dev/dri") != std::string::npos)
            continue;
        // vvar, vdso and vsyscall hold scalars and host-only pointers, not memory a kernel is
        // handed as data, and a by-value scalar whose bit pattern lands in their range would read
        // as a false hit. The thread stacks are kept: a kernel argument may well point at a host
        // variable, and missing that is the failure this module exists to catch.
        if(line.find("[vvar") != std::string::npos || line.find("[vdso") != std::string::npos ||
           line.find("[vsyscall") != std::string::npos)
            continue;
        out.spans.push_back({begin, end});
    }
    if(!file.eof() || file.bad())
    {
        out.spans.clear();
        return out;
    }
    std::sort(out.spans.begin(), out.spans.end(), [](const auto& l, const auto& r) {
        return l.begin < r.begin;
    });
    out.ok = true;
    return out;
}

namespace
{
// Anything below the first page, or above the 56-bit user range ROCr accepts, is a scalar rather
// than an address a kernel can dereference. That range includes addresses above the 47-bit
// canonical limit, such as an SVM aperture at 0x800000010000.
constexpr uint64_t min_candidate = 0x10000;
constexpr uint64_t max_candidate = (uint64_t{1} << 56) - 1;

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

void*
as_pointer(uint64_t address)
{
    return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

// Bounds of a kernarg region that contains at least one complete pointer-sized word.
bool
region_bounds(size_t segment_size, uint64_t offset, uint64_t size, size_t& begin, size_t& end)
{
    constexpr auto word = sizeof(uint64_t);
    if(offset > segment_size) return false;
    begin = static_cast<size_t>(offset);
    if(size > segment_size - begin || size < word) return false;
    end = begin + static_cast<size_t>(size);
    return true;
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
        case kind_t::unclassified: return "memory whose classification is unavailable";
    }
    return "untracked memory";
}

pointer_facts_fn_t
make_hsa_pointer_facts()
{
    auto mappings = std::make_shared<std::optional<host_mappings_t>>();

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
                facts.pointer_info_succeeded = true;
                facts.type                   = convert_type(info.type);
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
        if(facts.pointer_info_succeeded && facts.type == pointer_facts_t::type_t::unknown)
        {
            if(!mappings->has_value()) *mappings = read_writable_host_mappings();
            facts.host_mappings_ok = mappings->value().ok;
            if(facts.host_mappings_ok)
                facts.in_writable_host_mapping = in_spans(mappings->value().spans, address);
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
    if(!facts.pointer_info_succeeded) return kind_t::unclassified;
    switch(facts.type)
    {
        case type_t::hsa: return facts.owner_is_cpu ? kind_t::host_pinned : kind_t::device_other;
        case type_t::reserved: return kind_t::managed;
        case type_t::vmem: return kind_t::virtual_mem;
        case type_t::locked: return kind_t::locked_host;
        case type_t::other: return kind_t::device_other;
        case type_t::unknown:
            if(!facts.host_mappings_ok) return kind_t::unclassified;
            if(facts.in_writable_host_mapping) return kind_t::host_pageable;
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<hit_t>
scan_kernarg(const uint8_t*                                    kernarg,
             size_t                                            segment_size,
             const std::vector<code_object::kernarg_region_t>* pointer_regions,
             const std::vector<address_span_t>&                covered,
             const pointer_facts_fn_t&                         facts)
{
    if(kernarg == nullptr) return std::nullopt;

    const auto whole = std::vector<code_object::kernarg_region_t>{
        code_object::kernarg_region_t{0u,
                                      static_cast<uint32_t>(std::min<size_t>(
                                          segment_size, std::numeric_limits<uint32_t>::max()))}};
    const auto& regions = pointer_regions != nullptr ? *pointer_regions : whole;

    for(const auto& region : regions)
    {
        auto begin = size_t{};
        auto end   = size_t{};
        if(!region_bounds(segment_size, region.offset, region.size, begin, end)) continue;
        for(size_t off = begin; off <= end - sizeof(uint64_t); off += sizeof(uint64_t))
        {
            auto value = uint64_t{0};
            std::memcpy(&value, kernarg + off, sizeof(value));
            if(value < min_candidate || value > max_candidate) continue;
            if(in_spans(covered, value)) continue;
            if(const auto kind = classify_pointer(facts(value)); kind)
                return hit_t{off, value, *kind};
        }
    }
    return std::nullopt;
}

bool
can_replay_dispatch(const uint8_t*                                    kernarg,
                    size_t                                            segment_size,
                    const std::vector<code_object::kernarg_region_t>* pointer_regions,
                    const memory_snapshot::device_snapshot_t&         snapshot,
                    const pointer_facts_fn_t&                         facts)
{
    try
    {
        if(const auto hit =
               scan_kernarg(kernarg, segment_size, pointer_regions, covered_spans(snapshot), facts))
        {
            LOG_FIRST_N(WARNING, 1) << fmt::format(
                "kernel replay: kernarg offset {} points at 0x{:x}, {}; "
                "{}; running this dispatch once without replay",
                hit->kernarg_offset,
                hit->address,
                to_string(hit->kind),
                hit->kind == kind_t::unclassified ? "it could not be classified"
                                                  : "the snapshot does not restore it");
            return false;
        }
        return true;
    } catch(const std::bad_alloc&)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel replay: could not allocate memory for the untracked "
                                   "memory check; running this dispatch once without replay";
        return false;
    }
}

bool
can_replay_dispatch(const hsa_kernel_dispatch_packet_t&       packet,
                    const memory_snapshot::device_snapshot_t& snapshot)
{
    try
    {
        const auto kernel = code_object::locate_kernel(packet.kernel_object);
        if(!kernel)
        {
            LOG_FIRST_N(WARNING, 1)
                << "kernel replay: the dispatched kernel is not in the code "
                   "object registry, so the memory its arguments point at cannot "
                   "be checked; running this dispatch once without replay";
            return false;
        }

        if(kernel->name != nullptr && is_idempotent_runtime_kernel(*kernel->name)) return true;

        // Without metadata every word of the segment is a candidate, including the runtime's hidden
        // arguments -- its hostcall and printf buffers, its heap -- which point at memory the
        // snapshot does not restore, so such a dispatch may run once for that alone.
        const auto* regions = kernel->pointer_regions.get();
        if(regions == nullptr)
            LOG_FIRST_N(WARNING, 1)
                << fmt::format("kernel replay: no argument metadata for {} (kernarg segment {} "
                               "bytes), so every word "
                               "of it is checked for pointers including the runtime's hidden "
                               "arguments; a dispatch of "
                               "this kernel may run once without replay",
                               kernel->name != nullptr ? kernel->name->c_str() : "<unnamed kernel>",
                               kernel->segment_size);

        return can_replay_dispatch(static_cast<const uint8_t*>(packet.kernarg_address),
                                   kernel->segment_size,
                                   regions,
                                   snapshot,
                                   make_hsa_pointer_facts());
    } catch(const std::bad_alloc&)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel replay: could not allocate memory for the untracked "
                                   "memory check; running this dispatch once without replay";
        return false;
    }
}
}  // namespace untracked_memory
}  // namespace kernel_replay
}  // namespace rocprofiler
