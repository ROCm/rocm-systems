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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace msnp = ::rocprofiler::kernel_replay::memory_snapshot;
namespace um   = ::rocprofiler::kernel_replay::untracked_memory;

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
    return f;
}

uint64_t
address_of(const std::vector<uint8_t>& buf)
{
    return reinterpret_cast<uint64_t>(buf.data());
}

constexpr auto this_agent  = hsa_agent_t{9};
constexpr auto other_agent = hsa_agent_t{10};

// Fine-grained device memory of `owner` for `buf`, a virtual-memory mapping at vmem_addr, unknown
// to HSA otherwise. Host-pinned instead of device memory when `pinned`.
um::pointer_facts_fn_t
facts_for(const std::vector<uint8_t>& buf, hsa_agent_t owner = this_agent, bool pinned = false)
{
    return [&buf, owner, pinned](uint64_t address) {
        const auto base = address_of(buf);
        if(address >= base && address < base + buf.size())
        {
            auto f         = facts_of(type_t::hsa, pinned);
            f.fine_grained = true;
            f.agent_base   = base;
            f.size         = buf.size();
            f.owner        = owner.handle;
            return f;
        }
        if(address == vmem_addr) return facts_of(type_t::vmem);
        return facts_of(type_t::unknown);
    };
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

um::region_set_t
host_copy_set(bool ok = true)
{
    return um::region_set_t{[ok](void* dst, const void* src, size_t size) {
        if(ok) std::memcpy(dst, src, size);
        return ok;
    }};
}

std::vector<uint8_t>
kernarg_with(const std::vector<uint64_t>& words, size_t trailing = 0)
{
    auto out = std::vector<uint8_t>(words.size() * sizeof(uint64_t) + trailing, 0);
    std::memcpy(out.data(), words.data(), words.size() * sizeof(uint64_t));
    return out;
}

bool
save(um::region_set_t& saved, const std::vector<uint64_t>& words, const um::pointer_facts_fn_t& f)
{
    const auto kernarg = kernarg_with(words);
    return um::save_dispatch_memory(
        saved, kernarg.data(), kernarg.size(), msnp::device_snapshot_t{}, this_agent, f);
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
}

TEST(kernel_replay_untracked_memory, ignores_scalars_and_covered_memory)
{
    const auto covered = std::vector<um::address_span_t>{{tracked_addr, tracked_addr + 4096}};
    const auto facts   = [](uint64_t) { return facts_of(type_t::hsa, true); };

    // A small scalar, then a pointer into the snapshot, then the hidden arguments holding a
    // pointer of their own past the end of what the kernel was written with.
    auto kernarg = kernarg_with({42, tracked_addr + 8, vmem_addr});

    auto seen = std::unordered_set<uint64_t>{};
    // The kernel's own arguments are a scalar and memory the snapshot restores.
    EXPECT_TRUE(um::scan_kernarg(kernarg.data(), 2 * sizeof(uint64_t), covered, facts, seen)
                    .empty());
    // The word after them is a pointer the passes would write through, and stopping early is what
    // hides it: where the kernel's arguments end has to come from the kernel, not from a guess.
    const auto hits = um::scan_kernarg(kernarg.data(), kernarg.size(), covered, facts, seen);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].kernarg_offset, 2 * sizeof(uint64_t));
    EXPECT_EQ(hits[0].address, vmem_addr);
}

TEST(kernel_replay_untracked_memory, scans_whatever_the_metadata_does_not_mark_hidden)
{
    // The hidden arguments are only appended to kernels that use them, and where they begin is a
    // property of each kernel, so the boundary has to come from the kernel's own metadata. Guessing
    // a fixed block at the end of the segment would hide pointers in the kernels whose arguments
    // run that long, which is the failure this module exists to catch.
    EXPECT_EQ(um::explicit_arg_size(288, std::nullopt), 288u) << "no metadata to read";
    EXPECT_EQ(um::explicit_arg_size(288, 288), 288u) << "a kernel with no hidden arguments";
    EXPECT_EQ(um::explicit_arg_size(264, 8), 8u) << "hidden arguments after one pointer";
    EXPECT_EQ(um::explicit_arg_size(64, 128), 64u) << "the segment bounds the scan";

    // A pointer in the last word of a 288-byte segment, the shape of a kernel that takes many
    // arguments and has no hidden ones.
    auto kernarg = std::vector<uint8_t>(288, 0);
    std::memcpy(kernarg.data() + 280, &vmem_addr, sizeof(vmem_addr));

    const auto facts = [](uint64_t address)
    { return address == vmem_addr ? facts_of(type_t::vmem) : facts_of(type_t::unknown); };

    auto seen = std::unordered_set<uint64_t>{};
    auto hits = um::scan_kernarg(
        kernarg.data(), um::explicit_arg_size(288, 288), {}, facts, seen);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].kernarg_offset, 280u);
    EXPECT_EQ(hits[0].region.kind, um::kind_t::virtual_mem);

    // The same segment with metadata placing the kernel's arguments in its first word.
    EXPECT_TRUE(
        um::scan_kernarg(kernarg.data(), um::explicit_arg_size(288, 8), {}, facts, seen).empty());
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
    const auto mappings = um::read_writable_host_mappings();
    ASSERT_FALSE(mappings.empty());
    const auto by_address =
        [](const um::address_span_t& lhs, const um::address_span_t& rhs)
    { return lhs.begin < rhs.begin; };
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
}

TEST(kernel_replay_untracked_memory, fine_grained_memory_of_this_agent_is_restored_between_passes)
{
    auto buf   = std::vector<uint8_t>(64, 1);
    auto saved = host_copy_set();

    ASSERT_TRUE(save(saved, {42, address_of(buf) + 16, address_of(buf)}, facts_for(buf)));
    ASSERT_EQ(saved.size(), 1u) << "one allocation, however many arguments point into it";
    EXPECT_EQ(saved.bytes(), buf.size());

    // A pass writes 2; the next pass must start from 1 again.
    std::fill(buf.begin(), buf.end(), 2);
    ASSERT_TRUE(saved.restore());
    EXPECT_EQ(buf, std::vector<uint8_t>(64, 1));
}

TEST(kernel_replay_untracked_memory, memory_that_cannot_be_saved_runs_the_dispatch_once)
{
    auto buf = std::vector<uint8_t>(64, 1);

    const auto runs_once = [](const um::pointer_facts_fn_t& facts, uint64_t address) {
        auto saved = host_copy_set();
        return !save(saved, {address}, facts) && saved.empty();
    };
    EXPECT_TRUE(runs_once(facts_for(buf, this_agent, true), address_of(buf))) << "host-pinned";
    EXPECT_TRUE(runs_once(facts_for(buf, other_agent), address_of(buf))) << "another agent's";
    EXPECT_TRUE(runs_once(facts_for(buf), vmem_addr)) << "extent unknown";

    auto failing = host_copy_set(false);
    EXPECT_FALSE(save(failing, {address_of(buf)}, facts_for(buf))) << "copy failed";
    EXPECT_TRUE(failing.empty());
}
