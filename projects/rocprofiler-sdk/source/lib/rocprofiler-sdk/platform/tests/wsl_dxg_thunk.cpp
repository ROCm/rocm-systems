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
// release it, and an old or broken thunk must be refused rather than trusted
// or aborted on.

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
using rocprofiler::platform::wsl::DxgNodeTopology;
using rocprofiler::platform::wsl::DxgThunk;
using rocprofiler::platform::wsl::kDxgNodeTopologyAbiVersion;
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

    // Leave the caller's record untouched on a successful get_node, the way a
    // stub that answers the call without implementing it would.
    bool get_node_writes_nothing = false;

    // per-node answers, indexed by node id
    std::vector<int32_t>         node_status;
    std::vector<DxgNodeTopology> nodes;

    int count(const std::string& name) const
    {
        int n = 0;
        for(const auto& itr : calls)
            if(itr == name) ++n;
        return n;
    }
};

FakeThunkState* g_state = nullptr;

DxgNodeTopology
make_gpu_node(uint32_t node_id)
{
    auto node             = DxgNodeTopology{};
    node.StructSize       = sizeof(DxgNodeTopology);
    node.AbiVersion       = kDxgNodeTopologyAbiVersion;
    node.NodeId           = node_id;
    node.NumFComputeCores = 256;
    node.NumSIMDPerCU     = 4;
    node.NumShaderBanks   = 4;
    node.NumArrays        = 2;
    node.WaveFrontSize    = 32;
    node.DeviceId         = 0x744c;
    node.VendorId         = 0x1002;
    return node;
}

// A DxgThunk wired to g_state, plus the fixture that owns it.
struct FakeThunk
{
    FakeThunkState state;

    FakeThunk() { g_state = &state; }
    ~FakeThunk() { g_state = nullptr; }

    FakeThunk(const FakeThunk&) = delete;
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

    static int32_t GetNode(uint32_t node_id, uint32_t size, DxgNodeTopology* out)
    {
        g_state->calls.emplace_back("get_node");
        EXPECT_EQ(size, sizeof(DxgNodeTopology))
            << "the caller must tell the thunk how much room it has";

        if(node_id >= g_state->node_status.size()) return kFailure;
        if(g_state->node_status.at(node_id) != kHsaKmtStatusSuccess)
            return g_state->node_status.at(node_id);
        if(g_state->get_node_writes_nothing) return kHsaKmtStatusSuccess;

        *out = g_state->nodes.at(node_id);
        return kHsaKmtStatusSuccess;
    }

    DxgThunk table() const
    {
        auto thunk             = DxgThunk{};
        thunk.get_node         = &GetNode;
        thunk.acquire_snapshot = &AcquireSnapshot;
        thunk.release_snapshot = &ReleaseSnapshot;
        thunk.open_kfd         = &OpenKfd;
        thunk.close_kfd        = &CloseKfd;
        return thunk;
    }

    // Publish `count` well-formed GPU nodes.
    void publish_gpu_nodes(uint32_t count)
    {
        state.num_nodes = count;
        for(uint32_t i = 0; i < count; ++i)
        {
            state.nodes.emplace_back(make_gpu_node(i));
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
    // The librocdxg that ships today: every hsaKmt* entry point resolves,
    // DxgGetNodeTopology does not. Set false to model it.
    bool        exports_topology = true;
    std::string missing_symbol   = {};
    void*       handle           = reinterpret_cast<void*>(0xd06);

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
            if(!exports_topology && std::strncmp(name, "Dxg", 3) == 0) return nullptr;
            if(std::strcmp(name, "DxgGetNodeTopology") == 0)
                return reinterpret_cast<void*>(&FakeThunk::GetNode);
            if(std::strcmp(name, "hsaKmtAcquireSystemProperties") == 0)
                return reinterpret_cast<void*>(&FakeThunk::AcquireSnapshot);
            if(std::strcmp(name, "hsaKmtReleaseSystemProperties") == 0)
                return reinterpret_cast<void*>(&FakeThunk::ReleaseSnapshot);
            if(std::strcmp(name, "hsaKmtOpenKFD") == 0)
                return reinterpret_cast<void*>(&FakeThunk::OpenKfd);
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

// Exactly the symbols resolve_dxg_thunk() may ask for, in the order it asks.
// Pinning the whole set is the point: whatever else a librocdxg happens to
// export - in particular the process-global ABI handshake the HSA runtime
// performs, whose side effect is to reconfigure the thunk for its caller -
// must not be looked up here, let alone called.
const std::vector<std::string> kResolvedSymbols = {"DxgGetNodeTopology",
                                                   "hsaKmtAcquireSystemProperties",
                                                   "hsaKmtReleaseSystemProperties",
                                                   "hsaKmtOpenKFD",
                                                   "hsaKmtCloseKFD"};
}  // namespace

// --- the happy path --------------------------------------------------------

TEST(wsl_dxg_thunk, a_healthy_thunk_is_called_in_order_and_left_balanced)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(2);

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_EQ(nodes.size(), 2);
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
    EXPECT_EQ(nodes.front().NodeId, 1);
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
    EXPECT_EQ(nodes.at(0).NodeId, 0);
    EXPECT_EQ(nodes.at(1).NodeId, 2);
    EXPECT_EQ(fake.state.count("release_snapshot"), 1);
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
}

// --- the per-record compatibility gate -------------------------------------
//
// Each reply carries the ABI version the thunk speaks and the number of bytes
// it wrote. That pair is the entire compatibility contract, so these cases
// are what stands between an incompatible thunk and a published agent record.

// A thunk that wrote fewer bytes than this build expects is an older revision:
// the trailing fields were never written, so the record is refused rather than
// published with whatever was in the caller's buffer.
TEST(wsl_dxg_thunk, a_short_record_is_refused)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(1);
    fake.state.nodes.at(0).StructSize = sizeof(DxgNodeTopology) - 8;

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(fake.state.count("release_snapshot"), 1);
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
}

// The mirror image: a thunk claiming to have written more than the buffer it
// was handed is describing something this build cannot lay out either.
TEST(wsl_dxg_thunk, an_overlong_record_is_refused)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(1);
    fake.state.nodes.at(0).StructSize = sizeof(DxgNodeTopology) + 8;

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(fake.state.count("release_snapshot"), 1);
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
}

TEST(wsl_dxg_thunk, a_record_from_a_different_abi_version_is_refused)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(2);
    fake.state.nodes.at(0).AbiVersion = kDxgNodeTopologyAbiVersion - 1;
    fake.state.nodes.at(1).AbiVersion = kDxgNodeTopologyAbiVersion + 1;

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
}

// Something exporting the right name that answers the call without writing
// anything. The caller's record is zero-initialized before every call, so the
// same StructSize/AbiVersion pair catches this: a silent stub is refused
// instead of being published as a GPU with no compute units.
TEST(wsl_dxg_thunk, a_record_the_thunk_never_wrote_is_refused)
{
    auto fake                          = FakeThunk{};
    fake.state.get_node_writes_nothing = true;
    fake.publish_gpu_nodes(2);

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(fake.state.count("get_node"), 2);
    EXPECT_EQ(fake.state.count("release_snapshot"), 1);
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
}

// The gate is per record, not per thunk: one bad reply does not condemn the
// nodes around it, and one good reply does not vouch for them.
TEST(wsl_dxg_thunk, a_single_incompatible_record_does_not_discard_the_others)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(3);
    fake.state.nodes.at(1).AbiVersion = kDxgNodeTopologyAbiVersion + 1;

    const auto nodes = read_dxg_gpu_topology(fake.table());

    ASSERT_EQ(nodes.size(), 2);
    EXPECT_EQ(nodes.at(0).NodeId, 0);
    EXPECT_EQ(nodes.at(1).NodeId, 2);
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

// A thunk predating the topology ABI exports the hsaKmt* entry points but not
// DxgGetNodeTopology. Missing any one of the five must stop the read before it
// calls anything at all - not crash on a null pointer, and not abort.
TEST(wsl_dxg_thunk, a_thunk_missing_any_entry_point_is_never_called)
{
    for(int missing = 0; missing < 5; ++missing)
    {
        auto fake  = FakeThunk{};
        auto thunk = fake.table();

        switch(missing)
        {
            case 0: thunk.get_node = nullptr; break;
            case 1: thunk.acquire_snapshot = nullptr; break;
            case 2: thunk.release_snapshot = nullptr; break;
            case 3: thunk.open_kfd = nullptr; break;
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

// An old thunk resolves some symbols and not others. The handle it did open
// still has to be closed, and none of the entry points may be called.
TEST(wsl_dxg_thunk, an_old_thunk_missing_a_symbol_is_closed_without_being_called)
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

// The librocdxg that ships today: it opens, it exports the entry points the
// HSA runtime needs - snapshot ownership among them - and it has no
// DxgGetNodeTopology. This is the configuration the WSL path actually meets in
// the field, so the refusal has to be a clean one - the handle given back,
// nothing invoked, no GPU agents and no abort.
TEST(wsl_dxg_thunk, the_shipped_thunk_without_topology_exports_is_refused_cleanly)
{
    auto fake               = FakeThunk{};
    auto loader             = FakeLoader{};
    loader.exports_topology = false;
    fake.publish_gpu_nodes(2);

    const auto nodes = read_dxg_gpu_topology(loader.ops());

    EXPECT_TRUE(nodes.empty());
    EXPECT_EQ(loader.close_count, 1);
    EXPECT_TRUE(fake.state.calls.empty());
}

// resolve_dxg_thunk() reports exactly which entry points it found, so the
// warning naming the missing symbol is driven by the same data the caller
// checks.
TEST(wsl_dxg_thunk, resolve_reports_a_complete_table_only_when_all_five_resolve)
{
    auto loader = FakeLoader{};

    EXPECT_TRUE(resolve_dxg_thunk(loader.handle, loader.ops()).complete());

    loader.missing_symbol = "DxgGetNodeTopology";
    const auto partial    = resolve_dxg_thunk(loader.handle, loader.ops());
    EXPECT_FALSE(partial.complete());
    EXPECT_EQ(partial.get_node, nullptr);
    EXPECT_NE(partial.acquire_snapshot, nullptr) << "the other entry points still resolve";
}

// The resolved set is the whole of what this build asks a thunk for. Anything
// beyond it would be a second, unnecessary coupling to librocdxg's ABI - and
// the entry point most obviously missing from this list, the process-global
// handshake, is one whose side effect on the thunk's own state makes calling
// it from a second in-process consumer actively unsafe.
TEST(wsl_dxg_thunk, resolve_asks_for_exactly_the_entry_points_it_needs)
{
    auto loader = FakeLoader{};

    resolve_dxg_thunk(loader.handle, loader.ops());

    EXPECT_EQ(loader.requested_symbols, kResolvedSymbols);
}
