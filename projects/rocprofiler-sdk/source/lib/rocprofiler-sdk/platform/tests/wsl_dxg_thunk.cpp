// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// read_dxg_gpu_topology() driven by a fake librocdxg. The thunk function table
// and the dynamic-loader calls are both injected, so no dlopen, no real thunk,
// no HSA runtime and no GPU are involved and this runs anywhere.
//
// What is under test is the call sequence and its balance: every path that
// opened the thunk must close it, every path that acquired a snapshot must
// release it, and a thunk that does not export everything the read needs must
// be refused before any of it is called.

#include "lib/rocprofiler-sdk/platform/wsl/dxg_thunk.hpp"

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace
{
using rocprofiler::platform::wsl::DxgLoaderOps;
using rocprofiler::platform::wsl::DxgThunk;
using rocprofiler::platform::wsl::kHsaKmtStatusKernelAlreadyOpened;
using rocprofiler::platform::wsl::kHsaKmtStatusSuccess;
using rocprofiler::platform::wsl::read_dxg_gpu_topology;
using rocprofiler::platform::wsl::resolve_dxg_thunk;

constexpr int32_t kFailure = -1;

// The fake thunk's answers and its call log. A DxgThunk is a table of plain
// function pointers, so the trampolines below reach the fixture through this
// single pointer, set up by FakeThunk's constructor.
struct FakeThunkState
{
    std::vector<std::string> calls;

    int32_t  open_status    = kHsaKmtStatusSuccess;
    int32_t  acquire_status = kHsaKmtStatusSuccess;
    uint32_t num_nodes      = 0;

    // per-node answers, indexed by node id
    std::vector<int32_t>           node_status;
    std::vector<HsaNodeProperties> nodes;

    int count(const std::string& name) const
    {
        int n = 0;
        for(const auto& itr : calls)
            if(itr == name) ++n;
        return n;
    }
};

FakeThunkState* g_state = nullptr;

HsaNodeProperties
make_gpu_node()
{
    auto props             = HsaNodeProperties{};
    props.NumFComputeCores = 256;
    props.NumSIMDPerCU     = 4;
    props.NumShaderBanks   = 4;
    props.NumArrays        = 2;
    props.WaveFrontSize    = 32;
    props.DeviceId         = 0x744c;
    props.VendorId         = 0x1002;
    return props;
}

// A DxgThunk wired to g_state, plus the fixture that owns it.
struct FakeThunk
{
    FakeThunkState state;

    FakeThunk() { g_state = &state; }
    ~FakeThunk() { g_state = nullptr; }

    FakeThunk(const FakeThunk&)            = delete;
    FakeThunk& operator=(const FakeThunk&) = delete;

    static int32_t OpenKfd()
    {
        g_state->calls.emplace_back("open_kfd");
        return g_state->open_status;
    }

    static int32_t CloseKfd()
    {
        g_state->calls.emplace_back("close_kfd");
        return kHsaKmtStatusSuccess;
    }

    static int32_t AcquireSnapshot(HsaSystemProperties* props)
    {
        g_state->calls.emplace_back("acquire_snapshot");
        // The node count is read back out of this, so a caller that passed
        // nothing would walk zero nodes and look like an empty machine.
        EXPECT_NE(props, nullptr) << "acquire must be handed somewhere to write";
        if(props != nullptr) props->NumNodes = g_state->num_nodes;
        return g_state->acquire_status;
    }

    static int32_t ReleaseSnapshot()
    {
        g_state->calls.emplace_back("release_snapshot");
        return kHsaKmtStatusSuccess;
    }

    static int32_t GetNode(uint32_t node_id, HsaNodeProperties* out)
    {
        g_state->calls.emplace_back("get_node");
        EXPECT_NE(out, nullptr) << "the node read must be handed somewhere to write";

        if(node_id >= g_state->node_status.size()) return kFailure;
        if(g_state->node_status.at(node_id) != kHsaKmtStatusSuccess)
            return g_state->node_status.at(node_id);

        *out = g_state->nodes.at(node_id);
        return kHsaKmtStatusSuccess;
    }

    DxgThunk table() const
    {
        auto thunk             = DxgThunk{};
        thunk.open_kfd         = &OpenKfd;
        thunk.acquire_snapshot = &AcquireSnapshot;
        thunk.get_node         = &GetNode;
        thunk.release_snapshot = &ReleaseSnapshot;
        thunk.close_kfd        = &CloseKfd;
        return thunk;
    }

    // Publish `count` well-formed GPU nodes.
    void publish_gpu_nodes(uint32_t count)
    {
        state.num_nodes = count;
        for(uint32_t i = 0; i < count; ++i)
        {
            state.nodes.emplace_back(make_gpu_node());
            state.node_status.emplace_back(kHsaKmtStatusSuccess);
        }
    }
};

// A fake dynamic loader. Records the flags of every open so the
// RTLD_NOLOAD-then-plain-dlopen order can be checked, records every symbol
// asked for so the resolved set can be pinned exactly, and counts closes.
struct FakeLoader
{
    std::vector<int>         open_flags;
    std::vector<std::string> requested_symbols;
    int                      close_count      = 0;
    bool                     noload_finds_it  = true;
    bool                     plain_open_works = true;
    std::string              missing_symbol   = {};
    void*                    handle           = reinterpret_cast<void*>(0xd06);

    DxgLoaderOps ops()
    {
        auto out = DxgLoaderOps{};
        out.open = [this](const char*, int flags) -> void* {
            open_flags.emplace_back(flags);
            const bool is_noload = (flags & RTLD_NOLOAD) != 0;
            if(is_noload && !noload_finds_it) return nullptr;
            if(!is_noload && !plain_open_works) return nullptr;
            return handle;
        };
        out.sym = [this](void*, const char* name) -> void* {
            requested_symbols.emplace_back(name);
            if(missing_symbol == name) return nullptr;
            if(std::strcmp(name, "hsaKmtOpenKFD") == 0)
                return reinterpret_cast<void*>(&FakeThunk::OpenKfd);
            if(std::strcmp(name, "hsaKmtAcquireSystemProperties") == 0)
                return reinterpret_cast<void*>(&FakeThunk::AcquireSnapshot);
            if(std::strcmp(name, "hsaKmtGetNodeProperties") == 0)
                return reinterpret_cast<void*>(&FakeThunk::GetNode);
            if(std::strcmp(name, "hsaKmtReleaseSystemProperties") == 0)
                return reinterpret_cast<void*>(&FakeThunk::ReleaseSnapshot);
            if(std::strcmp(name, "hsaKmtCloseKFD") == 0)
                return reinterpret_cast<void*>(&FakeThunk::CloseKfd);
            return nullptr;
        };
        out.close = [this](void*) {
            ++close_count;
            return 0;
        };
        out.error = []() { return "fake loader error"; };
        return out;
    }
};

// The five KMT symbols the read cannot proceed without, in the order
// resolve_dxg_thunk() asks for them.
const std::vector<std::string> kResolvedSymbols = {"hsaKmtOpenKFD",
                                                   "hsaKmtAcquireSystemProperties",
                                                   "hsaKmtGetNodeProperties",
                                                   "hsaKmtReleaseSystemProperties",
                                                   "hsaKmtCloseKFD"};
}  // namespace

// --- the happy path --------------------------------------------------------

TEST(wsl_dxg_thunk, a_healthy_thunk_is_called_in_order_and_left_balanced)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(2);

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_EQ(nodes.size(), 2);
    EXPECT_EQ(nodes.at(0).node_id, 0);
    EXPECT_EQ(nodes.at(1).node_id, 1);
    EXPECT_EQ(fake.state.calls,
              (std::vector<std::string>{"open_kfd",
                                        "acquire_snapshot",
                                        "get_node",
                                        "get_node",
                                        "release_snapshot",
                                        "close_kfd"}));
}

// KERNEL_ALREADY_OPENED means another consumer - normally the HSA runtime -
// had the thunk open and we took an additional reference. It is a success and
// it still owes a close.
TEST(wsl_dxg_thunk, an_already_open_thunk_is_used_and_still_closed)
{
    auto fake              = FakeThunk{};
    fake.state.open_status = kHsaKmtStatusKernelAlreadyOpened;
    fake.publish_gpu_nodes(1);

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_EQ(nodes.size(), 1);
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
    EXPECT_EQ(fake.state.count("release_snapshot"), 1);
    EXPECT_EQ(fake.state.calls.back(), "close_kfd");
}

// CPU-only nodes are skipped, but the walk still visits every node and the
// teardown is unchanged.
TEST(wsl_dxg_thunk, cpu_only_nodes_are_skipped_not_published)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(2);
    fake.state.nodes.at(0).NumFComputeCores = 0;

    const auto nodes = read_dxg_gpu_topology(fake.table());

    ASSERT_EQ(nodes.size(), 1);
    EXPECT_EQ(nodes.front().node_id, 1);
    EXPECT_EQ(fake.state.count("get_node"), 2);
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
}

// --- refusals --------------------------------------------------------------

// A failed open owns nothing, so it must not close.
TEST(wsl_dxg_thunk, a_failed_open_does_not_close)
{
    auto fake              = FakeThunk{};
    fake.state.open_status = kFailure;
    fake.publish_gpu_nodes(2);

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(fake.state.calls, (std::vector<std::string>{"open_kfd"}));
}

// A failed acquire holds no snapshot, but the open above it still has to be
// balanced on the way out.
TEST(wsl_dxg_thunk, a_failed_acquire_closes_but_does_not_release)
{
    auto fake                 = FakeThunk{};
    fake.state.acquire_status = kFailure;
    fake.publish_gpu_nodes(2);

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(fake.state.calls,
              (std::vector<std::string>{"open_kfd", "acquire_snapshot", "close_kfd"}));
}

// One unreadable node does not abandon the others, and the snapshot and the
// open are still given back.
TEST(wsl_dxg_thunk, a_failed_node_query_skips_that_node_and_still_unwinds)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(3);
    fake.state.node_status.at(1) = kFailure;

    const auto nodes = read_dxg_gpu_topology(fake.table());

    ASSERT_EQ(nodes.size(), 2);
    EXPECT_EQ(nodes.at(0).node_id, 0);
    EXPECT_EQ(nodes.at(1).node_id, 2);
    EXPECT_EQ(fake.state.count("release_snapshot"), 1);
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
}

// An empty snapshot is not an error; it just publishes nothing.
TEST(wsl_dxg_thunk, an_empty_snapshot_is_released_and_closed)
{
    auto fake = FakeThunk{};

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(fake.state.calls,
              (std::vector<std::string>{
                  "open_kfd", "acquire_snapshot", "release_snapshot", "close_kfd"}));
}

// --- an incomplete function table ------------------------------------------

// Missing any one of the five must stop the read before it calls anything at
// all - not crash on a null pointer, and not abort.
TEST(wsl_dxg_thunk, a_thunk_missing_any_entry_point_is_never_called)
{
    for(int missing = 0; missing < 5; ++missing)
    {
        auto fake  = FakeThunk{};
        auto thunk = fake.table();

        switch(missing)
        {
            case 0: thunk.open_kfd = nullptr; break;
            case 1: thunk.acquire_snapshot = nullptr; break;
            case 2: thunk.get_node = nullptr; break;
            case 3: thunk.release_snapshot = nullptr; break;
            case 4: thunk.close_kfd = nullptr; break;
            default: break;
        }

        EXPECT_FALSE(thunk.complete()) << "entry point " << missing;
        EXPECT_TRUE(read_dxg_gpu_topology(thunk).empty()) << "entry point " << missing;
        EXPECT_TRUE(fake.state.calls.empty()) << "entry point " << missing;
    }
}

// --- the loader ------------------------------------------------------------

// The normal case during a profiling run: the HSA runtime already loaded the
// thunk, RTLD_NOLOAD finds it, and the reference taken is given back.
TEST(wsl_dxg_thunk, an_already_resident_library_is_reused_and_released)
{
    auto fake   = FakeThunk{};
    auto loader = FakeLoader{};
    fake.publish_gpu_nodes(1);

    const auto nodes = read_dxg_gpu_topology(loader.ops());

    EXPECT_EQ(nodes.size(), 1);
    ASSERT_EQ(loader.open_flags.size(), 1);
    EXPECT_TRUE((loader.open_flags.front() & RTLD_NOLOAD) != 0)
        << "the resident copy must be looked for before a fresh load";
    EXPECT_EQ(loader.close_count, 1);
}

// A pre-HSA consumer - rocprofv3-avail, tool initialization - is the first to
// touch the library, so RTLD_NOLOAD misses and a real load follows.
TEST(wsl_dxg_thunk, a_cold_process_falls_through_to_a_real_load)
{
    auto fake              = FakeThunk{};
    auto loader            = FakeLoader{};
    loader.noload_finds_it = false;
    fake.publish_gpu_nodes(1);

    const auto nodes = read_dxg_gpu_topology(loader.ops());

    EXPECT_EQ(nodes.size(), 1);
    ASSERT_EQ(loader.open_flags.size(), 2);
    EXPECT_TRUE((loader.open_flags.at(0) & RTLD_NOLOAD) != 0);
    EXPECT_TRUE((loader.open_flags.at(1) & RTLD_NOLOAD) == 0);
    EXPECT_EQ(loader.close_count, 1);
}

// No thunk at all: nothing was opened, so nothing is closed, and it is not
// fatal.
TEST(wsl_dxg_thunk, a_missing_library_closes_nothing)
{
    auto fake               = FakeThunk{};
    auto loader             = FakeLoader{};
    loader.noload_finds_it  = false;
    loader.plain_open_works = false;

    const auto nodes = read_dxg_gpu_topology(loader.ops());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(loader.close_count, 0);
    EXPECT_TRUE(fake.state.calls.empty());
}

// A thunk that resolves some symbols and not others. The handle it did open
// still has to be closed, and none of the entry points may be called.
TEST(wsl_dxg_thunk, a_thunk_missing_a_required_symbol_is_closed_without_being_called)
{
    for(const auto& symbol : kResolvedSymbols)
    {
        auto fake             = FakeThunk{};
        auto loader           = FakeLoader{};
        loader.missing_symbol = symbol;
        fake.publish_gpu_nodes(1);

        const auto nodes = read_dxg_gpu_topology(loader.ops());

        EXPECT_TRUE(nodes.empty()) << symbol;
        EXPECT_EQ(loader.close_count, 1) << symbol;
        EXPECT_TRUE(fake.state.calls.empty()) << symbol;
    }
}

// resolve_dxg_thunk() reports exactly which entry points it found, so the
// warning naming the missing symbol is driven by the same data the caller
// checks.
TEST(wsl_dxg_thunk, resolve_reports_a_complete_table_only_when_all_five_resolve)
{
    auto loader = FakeLoader{};

    EXPECT_TRUE(resolve_dxg_thunk(loader.handle, loader.ops()).complete());

    loader.missing_symbol = "hsaKmtGetNodeProperties";
    const auto partial    = resolve_dxg_thunk(loader.handle, loader.ops());
    EXPECT_FALSE(partial.complete());
    EXPECT_EQ(partial.get_node, nullptr);
    EXPECT_NE(partial.acquire_snapshot, nullptr) << "the other entry points still resolve";
}

// The resolved set is the whole KMT interface this topology read requires.
TEST(wsl_dxg_thunk, resolve_asks_for_exactly_the_entry_points_it_needs)
{
    auto loader = FakeLoader{};

    resolve_dxg_thunk(loader.handle, loader.ops());

    EXPECT_EQ(loader.requested_symbols, kResolvedSymbols);
}
