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

#include <gtest/gtest.h>

#include <sys/mman.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace msnp = ::rocprofiler::kernel_replay::memory_snapshot;
namespace um   = ::rocprofiler::kernel_replay::untracked_memory;
namespace co   = ::rocprofiler::code_object;

namespace
{
using type_t = um::pointer_facts_t::type_t;

constexpr uint64_t tracked_addr = 0x7f0000100000;
constexpr uint64_t vmem_addr    = 0x7f0000200000;

um::pointer_facts_t
facts_of(type_t type, bool owner_is_cpu = false, bool in_host_mapping = false)
{
    auto f                     = um::pointer_facts_t{};
    f.type                     = type;
    f.owner_is_cpu             = owner_is_cpu;
    f.in_writable_host_mapping = in_host_mapping;
    f.pointer_info_succeeded   = true;
    f.host_mappings_ok         = true;
    return f;
}

bool
covers(const std::vector<um::address_span_t>& mappings, uint64_t address)
{
    for(const auto& mapping : mappings)
        if(address >= mapping.begin && address < mapping.end) return true;
    return false;
}

// Address at which the named mapping of this process starts, or nullopt when it has none.
std::optional<uint64_t>
mapping_start(const std::string& name)
{
    auto file = std::ifstream{"/proc/self/maps"};
    auto line = std::string{};
    while(std::getline(file, line))
    {
        if(line.find(name) == std::string::npos) continue;
        return static_cast<uint64_t>(std::strtoull(line.c_str(), nullptr, 16));
    }
    return std::nullopt;
}

std::vector<uint8_t>
kernarg_with(const std::vector<uint64_t>& words, size_t trailing = 0)
{
    auto out = std::vector<uint8_t>(words.size() * sizeof(uint64_t) + trailing, 0);
    std::memcpy(out.data(), words.data(), words.size() * sizeof(uint64_t));
    return out;
}

bool
can_replay(const std::vector<uint64_t>& words, const um::pointer_facts_fn_t& facts)
{
    const auto kernarg = kernarg_with(words);
    return um::can_replay_dispatch(
        kernarg.data(), kernarg.size(), nullptr, msnp::device_snapshot_t{}, facts);
}
}  // namespace

TEST(kernel_replay_untracked_memory, classifies_memory_the_snapshot_does_not_restore)
{
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::hsa, true)), um::kind_t::host_pinned);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::hsa, false)), um::kind_t::device_other);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::reserved)), um::kind_t::managed);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::vmem)), um::kind_t::virtual_mem);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::locked)), um::kind_t::locked_host);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::unknown, false, true)),
              um::kind_t::host_pageable);
    EXPECT_EQ(um::classify_pointer(facts_of(type_t::unknown)), std::nullopt)
        << "not memory a kernel can reach";
    auto unavailable_query                   = facts_of(type_t::unknown, false, true);
    unavailable_query.pointer_info_succeeded = false;
    EXPECT_EQ(um::classify_pointer(unavailable_query), um::kind_t::unclassified);
    unavailable_query.pointer_info_succeeded = true;
    unavailable_query.host_mappings_ok       = false;
    EXPECT_EQ(um::classify_pointer(unavailable_query), um::kind_t::unclassified);
}

TEST(kernel_replay_untracked_memory, ignores_scalars_and_covered_memory)
{
    const auto covered = std::vector<um::address_span_t>{{tracked_addr, tracked_addr + 4096}};
    const auto facts   = [](uint64_t) { return facts_of(type_t::hsa, true); };

    // A small scalar, a pointer into the snapshot, then a pointer the passes would write through.
    auto kernarg = kernarg_with({42, tracked_addr + 8, vmem_addr});

    // Read only as far as the first two words, it holds nothing the snapshot does not restore.
    const auto first_two = std::vector<co::kernarg_region_t>{{0u, 16u}};
    EXPECT_FALSE(
        um::scan_kernarg(kernarg.data(), kernarg.size(), &first_two, covered, facts).has_value());
    // Read whole, the third word is found.
    const auto hits = um::scan_kernarg(kernarg.data(), kernarg.size(), nullptr, covered, facts);
    ASSERT_TRUE(hits.has_value());
    EXPECT_EQ(hits->kernarg_offset, 2 * sizeof(uint64_t));
    EXPECT_EQ(hits->address, vmem_addr);
}

TEST(kernel_replay_untracked_memory, only_what_metadata_calls_a_pointer_is_queried)
{
    // A by-value scalar whose bits land in the range a kernel could dereference must not be taken
    // for a pointer: nothing writes through it, and a value that happens to sit over memory the
    // snapshot does not restore would run the dispatch once for nothing.
    constexpr auto scalar_value = uint64_t{0x40000000};
    auto           queried      = std::vector<uint64_t>{};
    const auto     facts        = [&queried](uint64_t address) {
        queried.push_back(address);
        if(address == scalar_value) return facts_of(type_t::unknown, false, true);
        if(address == vmem_addr) return facts_of(type_t::vmem);
        return facts_of(type_t::unknown);
    };

    const auto kernarg = kernarg_with({scalar_value, vmem_addr});
    const auto regions = std::vector<co::kernarg_region_t>{co::kernarg_region_t{8u, 8u}};

    const auto hits = um::scan_kernarg(kernarg.data(), kernarg.size(), &regions, {}, facts);
    ASSERT_TRUE(hits.has_value());
    EXPECT_EQ(hits->kernarg_offset, 8u);
    EXPECT_EQ(hits->kind, um::kind_t::virtual_mem);
    EXPECT_EQ(std::count(queried.begin(), queried.end(), scalar_value), 0)
        << "the scalar was never classified";

    // Without metadata the whole segment is the candidate set, scalar included.
    const auto blind = um::scan_kernarg(kernarg.data(), kernarg.size(), nullptr, {}, facts);
    ASSERT_TRUE(blind.has_value());
    EXPECT_EQ(blind->address, scalar_value);
}

TEST(kernel_replay_untracked_memory, reads_unaligned_pointer_regions_at_their_metadata_offsets)
{
    auto kernarg = std::vector<uint8_t>(sizeof(uint32_t) + sizeof(uint64_t), 0);
    std::memcpy(kernarg.data() + sizeof(uint32_t), &vmem_addr, sizeof(vmem_addr));
    const auto regions = std::vector<co::kernarg_region_t>{{sizeof(uint32_t), sizeof(uint64_t)}};
    const auto facts   = [](uint64_t address) {
        return address == vmem_addr ? facts_of(type_t::vmem) : facts_of(type_t::unknown);
    };

    const auto hits = um::scan_kernarg(kernarg.data(), kernarg.size(), &regions, {}, facts);
    ASSERT_TRUE(hits.has_value());
    EXPECT_EQ(hits->kernarg_offset, sizeof(uint32_t));
    EXPECT_EQ(hits->address, vmem_addr);
}

TEST(kernel_replay_untracked_memory, only_the_runtime_copy_and_fill_kernels_are_exempt)
{
    // Replaying a copy or a fill writes the same bytes again, which changes nothing. The code
    // object registry hands out the name of the kernel descriptor, which carries a ".kd" suffix.
    EXPECT_TRUE(um::is_idempotent_runtime_kernel("__amd_rocclr_copyBuffer"));
    EXPECT_TRUE(um::is_idempotent_runtime_kernel("__amd_rocclr_copyBuffer.kd"));
    EXPECT_TRUE(um::is_idempotent_runtime_kernel("__amd_rocclr_fillBufferAligned"));
    EXPECT_TRUE(um::is_idempotent_runtime_kernel("__amd_rocclr_copyBufferToImage"));

    // The runtime's other kernels change state every time they run, and a kernel that merely
    // shares their prefix belongs to the application.
    EXPECT_FALSE(um::is_idempotent_runtime_kernel("__amd_rocclr_streamOpsIncrement"));
    EXPECT_FALSE(um::is_idempotent_runtime_kernel("__amd_rocclr_streamOpsDecrement"));
    EXPECT_FALSE(um::is_idempotent_runtime_kernel("__amd_rocclr_scheduler"));
    EXPECT_FALSE(um::is_idempotent_runtime_kernel("__amd_rocclr_initHeap"));
    EXPECT_FALSE(um::is_idempotent_runtime_kernel("__amd_rocclr_myKernel"));
    EXPECT_FALSE(um::is_idempotent_runtime_kernel("vecAdd"));
    EXPECT_FALSE(um::is_idempotent_runtime_kernel("vecAdd.kd"));
    EXPECT_FALSE(um::is_idempotent_runtime_kernel(""));
}

TEST(kernel_replay_untracked_memory, host_mappings_hold_the_memory_a_kernel_could_be_handed)
{
    const auto result = um::read_writable_host_mappings();
    ASSERT_TRUE(result.ok);
    const auto& mappings = result.spans;
    ASSERT_FALSE(mappings.empty());
    const auto by_address = [](const um::address_span_t& lhs, const um::address_span_t& rhs) {
        return lhs.begin < rhs.begin;
    };
    EXPECT_TRUE(std::is_sorted(mappings.begin(), mappings.end(), by_address));

    // A kernel argument may point at a host variable, and the snapshot holds no host memory, so
    // the stack has to read as writable memory like any other mapping.
    auto local = uint64_t{0};
    auto heap  = std::make_unique<uint64_t[]>(1024);
    EXPECT_TRUE(covers(mappings, reinterpret_cast<uint64_t>(&local)))
        << "the thread stack is memory a kernel can be handed";
    EXPECT_TRUE(covers(mappings, reinterpret_cast<uint64_t>(heap.get())));

    // The pages the kernel maps itself hold scalars and host-only pointers, and a by-value argument
    // whose bit pattern lands in one would read as a hit.
    for(const auto* name : {"[vdso]", "[vvar]", "[vsyscall]"})
    {
        const auto start = mapping_start(name);
        if(!start) continue;  // not every kernel maps these
        EXPECT_FALSE(covers(mappings, *start)) << name << " holds nothing a kernel is handed";
    }

    // PROT_WRITE without PROT_READ is `-w-` in /proc/self/maps. The guard never reads the mapping,
    // and a pointer HSA does not know that lands there must still decline replay.
    constexpr auto page = size_t{4096};
    void* write_only    = ::mmap(nullptr, page, PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(write_only, MAP_FAILED);
    const auto updated = um::read_writable_host_mappings();
    EXPECT_TRUE(updated.ok);
    EXPECT_TRUE(covers(updated.spans, reinterpret_cast<uint64_t>(write_only)))
        << "a writable-only mapping is host memory a kernel can be handed";
    ::munmap(write_only, page);

    auto       empty_file = std::istringstream{};
    const auto empty      = um::read_writable_host_mappings(empty_file);
    EXPECT_TRUE(empty.ok);
    EXPECT_TRUE(empty.spans.empty());

    for(const auto state : {std::ios::failbit, std::ios::badbit})
    {
        auto file = std::istringstream{};
        file.setstate(state);
        const auto failed = um::read_writable_host_mappings(file);
        EXPECT_FALSE(failed.ok);
        EXPECT_TRUE(failed.spans.empty());
    }
}

TEST(kernel_replay_untracked_memory, addresses_in_the_56_bit_user_range_are_classified)
{
    // Just above the 47-bit canonical limit, and inside the 56-bit user range ROCr accepts.
    constexpr auto wide = uint64_t{0x800000010000};
    // One bit past that range is not a user address.
    constexpr auto past  = uint64_t{1} << 56;
    const auto     facts = [](uint64_t address) {
        if(address == wide) return facts_of(type_t::vmem);
        return facts_of(type_t::unknown);
    };

    const auto kernarg = kernarg_with({wide, past});
    const auto hits    = um::scan_kernarg(kernarg.data(), kernarg.size(), nullptr, {}, facts);
    ASSERT_TRUE(hits.has_value());
    EXPECT_EQ(hits->address, wide);
    EXPECT_EQ(hits->kind, um::kind_t::virtual_mem);
}

TEST(kernel_replay_untracked_memory, memory_outside_the_snapshot_runs_the_dispatch_once)
{
    for(const auto type :
        {type_t::hsa, type_t::locked, type_t::reserved, type_t::vmem, type_t::other})
        EXPECT_FALSE(can_replay({vmem_addr}, [type](uint64_t) { return facts_of(type); }));
    EXPECT_FALSE(can_replay({vmem_addr}, [](uint64_t) { return facts_of(type_t::hsa, true); }))
        << "host-pinned";
    EXPECT_FALSE(can_replay({vmem_addr}, [](uint64_t) {
        return facts_of(type_t::unknown, false, true);
    })) << "pageable host memory";

    for(const bool query_ok : {false, true})
    {
        EXPECT_FALSE(can_replay({vmem_addr}, [query_ok](uint64_t) {
            auto facts                   = facts_of(type_t::unknown);
            facts.pointer_info_succeeded = query_ok;
            facts.host_mappings_ok       = false;
            return facts;
        })) << (query_ok ? "host mapping query failed" : "pointer info query failed");
    }
    EXPECT_TRUE(can_replay({vmem_addr}, [](uint64_t) { return facts_of(type_t::unknown); }))
        << "successful UNKNOWN query with no writable host mapping";
}

TEST(kernel_replay_untracked_memory, allocation_failure_declines_replay)
{
    auto       kernarg = kernarg_with({vmem_addr});
    const auto facts   = [](uint64_t) -> um::pointer_facts_t { throw std::bad_alloc{}; };

    EXPECT_FALSE(um::can_replay_dispatch(
        kernarg.data(), kernarg.size(), nullptr, msnp::device_snapshot_t{}, facts));
}
