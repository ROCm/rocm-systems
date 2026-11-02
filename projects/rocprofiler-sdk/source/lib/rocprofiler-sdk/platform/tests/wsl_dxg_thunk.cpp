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
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace
{
using rocprofiler::platform::wsl::DxgLoaderOps;
using rocprofiler::platform::wsl::DxgThunk;
using rocprofiler::platform::wsl::read_dxg_gpu_topology;
using rocprofiler::platform::wsl::resolve_dxg_thunk;

constexpr HSAKMT_STATUS kFailure = HSAKMT_STATUS_ERROR;

class ToolAttachedEnv
{
public:
    explicit ToolAttachedEnv(bool enabled)
    {
        if(const char* value = ::getenv("ROCPROFILER_REGISTER_TOOL_ATTACHED")) previous_ = value;

        if(enabled)
            ::setenv("ROCPROFILER_REGISTER_TOOL_ATTACHED", "1", /*overwrite=*/1);
        else
            ::unsetenv("ROCPROFILER_REGISTER_TOOL_ATTACHED");
    }

    ~ToolAttachedEnv()
    {
        if(previous_)
            ::setenv("ROCPROFILER_REGISTER_TOOL_ATTACHED", previous_->c_str(), /*overwrite=*/1);
        else
            ::unsetenv("ROCPROFILER_REGISTER_TOOL_ATTACHED");
    }

    ToolAttachedEnv(const ToolAttachedEnv&) = delete;
    ToolAttachedEnv& operator=(const ToolAttachedEnv&) = delete;

private:
    std::optional<std::string> previous_;
};

// The fake thunk's answers and its call log. A DxgThunk is a table of plain
// function pointers, so the trampolines below reach the fixture through this
// single pointer, set up by FakeThunk's constructor.
struct FakeThunkState
{
    std::vector<std::string> calls;

    HSAKMT_STATUS open_status    = HSAKMT_STATUS_SUCCESS;
    HSAKMT_STATUS acquire_status = HSAKMT_STATUS_SUCCESS;
    uint32_t      num_nodes      = 0;

    // per-node answers, indexed by node id
    std::vector<HSAKMT_STATUS>     node_status;
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

    FakeThunk(const FakeThunk&) = delete;
    FakeThunk& operator=(const FakeThunk&) = delete;

    static HSAKMT_STATUS OpenKfd()
    {
        g_state->calls.emplace_back("open_kfd");
        return g_state->open_status;
    }

    static HSAKMT_STATUS CloseKfd()
    {
        g_state->calls.emplace_back("close_kfd");
        return HSAKMT_STATUS_SUCCESS;
    }

    static HSAKMT_STATUS AcquireSnapshot(HsaSystemProperties* props)
    {
        g_state->calls.emplace_back("acquire_snapshot");
        // The node count is read back out of this, so a caller that passed
        // nothing would walk zero nodes and look like an empty machine.
        EXPECT_NE(props, nullptr) << "acquire must be handed somewhere to write";
        if(props != nullptr) props->NumNodes = g_state->num_nodes;
        return g_state->acquire_status;
    }

    static HSAKMT_STATUS ReleaseSnapshot()
    {
        g_state->calls.emplace_back("release_snapshot");
        return HSAKMT_STATUS_SUCCESS;
    }

    static HSAKMT_STATUS GetNode(HSAuint32 node_id, HsaNodeProperties* out)
    {
        g_state->calls.emplace_back("get_node");
        EXPECT_NE(out, nullptr) << "the node read must be handed somewhere to write";

        if(node_id >= g_state->node_status.size()) return kFailure;
        if(g_state->node_status.at(node_id) != HSAKMT_STATUS_SUCCESS)
            return g_state->node_status.at(node_id);

        *out = g_state->nodes.at(node_id);
        return HSAKMT_STATUS_SUCCESS;
    }

    // A thunk built against the same hsakmt revision as this test.
    static HSAKMT_STATUS AbiCheck(HsaStructureSizes* sizes)
    {
        g_state->calls.emplace_back("abi_check");
        EXPECT_NE(sizes, nullptr) << "the handshake must advertise something to negotiate over";
        if(sizes == nullptr) return kFailure;

        EXPECT_EQ(sizes->StructureSizes, static_cast<uint16_t>(sizeof(HsaStructureSizes)));
        EXPECT_EQ(sizes->SizeOfHsaNodeProperties, static_cast<uint16_t>(sizeof(HsaNodeProperties)))
            << "the caller must advertise the record size it actually reads";
        return HSAKMT_STATUS_SUCCESS;
    }

    // A thunk whose HsaNodeProperties is not the one this build reads.
    // HSAKMT_STATUS_DRIVER_MISMATCH is what librocdxg answers with.
    static HSAKMT_STATUS AbiCheckMismatch(HsaStructureSizes*)
    {
        g_state->calls.emplace_back("abi_check");
        return HSAKMT_STATUS_DRIVER_MISMATCH;
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
            state.node_status.emplace_back(HSAKMT_STATUS_SUCCESS);
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
            if(std::strcmp(name, "DxgAbiCheck") == 0)
                return reinterpret_cast<void*>(&FakeThunk::AbiCheck);
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
const std::vector<std::string> kRequiredSymbols = {"hsaKmtOpenKFD",
                                                   "hsaKmtAcquireSystemProperties",
                                                   "hsaKmtGetNodeProperties",
                                                   "hsaKmtReleaseSystemProperties",
                                                   "hsaKmtCloseKFD"};

// Everything resolve_dxg_thunk() asks for: the five above plus the optional
// structure-size handshake, asked for last.
const std::vector<std::string> kResolvedSymbols = {"hsaKmtOpenKFD",
                                                   "hsaKmtAcquireSystemProperties",
                                                   "hsaKmtGetNodeProperties",
                                                   "hsaKmtReleaseSystemProperties",
                                                   "hsaKmtCloseKFD",
                                                   "DxgAbiCheck"};
}  // namespace

// --- the happy path --------------------------------------------------------

TEST(wsl_dxg_thunk, constructor_ordered_enumeration_calls_the_thunk_and_leaves_it_balanced)
{
    auto mode = ToolAttachedEnv{false};
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

TEST(wsl_dxg_thunk, explicit_late_attach_never_loads_or_calls_the_thunk)
{
    auto mode   = ToolAttachedEnv{true};
    auto fake   = FakeThunk{};
    auto loader = FakeLoader{};
    fake.publish_gpu_nodes(2);

    EXPECT_TRUE(read_dxg_gpu_topology(loader.ops()).empty());
    EXPECT_TRUE(loader.open_flags.empty()) << "late attach must be rejected before dlopen";
    EXPECT_TRUE(loader.requested_symbols.empty());
    EXPECT_EQ(loader.close_count, 0);
    EXPECT_TRUE(fake.state.calls.empty());

    // rocprofiler-register owns marker presence. A copied helper environment
    // must not turn a different value into permission to touch the thunk.
    ::setenv("ROCPROFILER_REGISTER_TOOL_ATTACHED", "0", /*overwrite=*/1);
    EXPECT_TRUE(read_dxg_gpu_topology(fake.table()).empty());
    EXPECT_TRUE(fake.state.calls.empty())
        << "late attach must not call ABI/open/acquire/get/release/close";
}

// KERNEL_ALREADY_OPENED means another consumer - normally the HSA runtime -
// had the thunk open and we took an additional reference. It is a success and
// it still owes a close.
TEST(wsl_dxg_thunk, an_already_open_thunk_is_used_and_still_closed)
{
    auto fake              = FakeThunk{};
    fake.state.open_status = HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
    fake.publish_gpu_nodes(1);

    const auto nodes = read_dxg_gpu_topology(fake.table());

    EXPECT_EQ(nodes.size(), 1);
    EXPECT_EQ(fake.state.count("close_kfd"), 1);
    EXPECT_EQ(fake.state.count("release_snapshot"), 1);
    EXPECT_EQ(fake.state.calls.back(), "close_kfd");
}

// --- the snapshot lifetime prerequisite ------------------------------------
//
// Released external thunks do not refcount the snapshot pair:
// hsaKmtReleaseSystemProperties() drops their one global snapshot and deletes
// every WDDMDevice with it, for all consumers. read_dxg_gpu_topology() is safe
// with one of those packages only because rocprofv3 runs it from a library
// constructor, before hsa_init(). The prerequisite in-tree runtime refcounts
// the snapshot, but does not expose a run-time capability bit for that behavior.
//
// Late attach is gated above by ROCPROFILER_REGISTER_TOOL_ATTACHED, which
// rocprofiler-register sets explicitly before tool initialization. The two
// tests below pin why inferred thunk state must never replace that marker.
// Once constructor-ordered enumeration has been admitted, the sequence stays
// balanced and unconditional with either ownership model.

// hsaKmtOpenKFD() reports whether its open count was already non-zero, not
// whether a snapshot already exists - any second consumer of the thunk produces
// KERNEL_ALREADY_OPENED, and hsaKmtAcquireSystemProperties() answers SUCCESS
// whether it took a fresh snapshot or handed back the live one. So the status
// cannot select a different teardown, and the sequence does not branch on it.
TEST(wsl_dxg_thunk, the_open_status_does_not_select_a_different_teardown)
{
    auto observe = [](HSAKMT_STATUS open_status) {
        auto fake              = FakeThunk{};
        fake.state.open_status = open_status;
        fake.publish_gpu_nodes(2);
        read_dxg_gpu_topology(fake.table());
        return fake.state.calls;
    };

    const auto fresh   = observe(HSAKMT_STATUS_SUCCESS);
    const auto already = observe(HSAKMT_STATUS_KERNEL_ALREADY_OPENED);

    EXPECT_EQ(fresh, already) << "KERNEL_ALREADY_OPENED cannot distinguish a late attach, so "
                                 "nothing here may be made conditional on it";
    EXPECT_EQ(already.back(), "close_kfd");
}

// DxgAbiCheck negotiates sizeof(HsaNodeProperties) and nothing else, and which
// thunks export it runs the wrong way round to stand in for a snapshot-lifetime
// capability: the released librocdxg packages do export it, while a thunk built
// from the in-tree sources does not. Gating on it would mark exactly the shipped
// thunks as capable. Whether it is present must therefore change nothing about
// the open/acquire/release/close sequence.
TEST(wsl_dxg_thunk, the_abi_handshake_is_not_a_snapshot_lifetime_capability_bit)
{
    auto observe = [](bool exports_handshake) {
        auto fake = FakeThunk{};
        fake.publish_gpu_nodes(2);
        auto thunk = fake.table();
        if(exports_handshake) thunk.abi_check = &FakeThunk::AbiCheck;
        read_dxg_gpu_topology(thunk);

        auto calls = fake.state.calls;
        // The handshake call itself is the one permitted difference.
        if(!calls.empty() && calls.front() == "abi_check") calls.erase(calls.begin());
        return calls;
    };

    EXPECT_EQ(observe(false), observe(true))
        << "the structure-size handshake must not gate the snapshot lifetime";
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

// --- the structure-size handshake ------------------------------------------

// A thunk that answers the handshake with a different sizeof(HsaNodeProperties)
// writes records this build cannot interpret, so the read is abandoned before
// the thunk is opened. Reading it anyway is the silent corruption the handshake
// exists to prevent.
TEST(wsl_dxg_thunk, a_thunk_reporting_a_different_record_size_is_never_read)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(1);

    auto thunk      = fake.table();
    thunk.abi_check = &FakeThunk::AbiCheckMismatch;

    EXPECT_TRUE(read_dxg_gpu_topology(thunk).empty());
    EXPECT_EQ(fake.state.calls, (std::vector<std::string>{"abi_check"}))
        << "nothing may be opened or read once the layouts are known to differ";
}

// A thunk that agrees is read normally, and the handshake happens before
// anything else so a mismatch cannot leave a half-open thunk behind.
TEST(wsl_dxg_thunk, a_thunk_agreeing_on_the_record_size_is_read)
{
    auto fake = FakeThunk{};
    fake.publish_gpu_nodes(1);

    auto thunk      = fake.table();
    thunk.abi_check = &FakeThunk::AbiCheck;

    EXPECT_EQ(read_dxg_gpu_topology(thunk).size(), 1);
    EXPECT_EQ(fake.state.calls.front(), "abi_check");
    EXPECT_EQ(fake.state.calls.back(), "close_kfd");
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
    for(const auto& symbol : kRequiredSymbols)
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

    // The handshake is the one entry point whose absence is not a defect, so it
    // must not count towards completeness.
    loader.missing_symbol = "DxgAbiCheck";
    const auto older      = resolve_dxg_thunk(loader.handle, loader.ops());
    EXPECT_TRUE(older.complete()) << "a thunk predating the handshake is still usable";
    EXPECT_EQ(older.abi_check, nullptr);
}

// The resolved set is the whole KMT interface this topology read requires.
TEST(wsl_dxg_thunk, resolve_asks_for_exactly_the_entry_points_it_needs)
{
    auto loader = FakeLoader{};

    resolve_dxg_thunk(loader.handle, loader.ops());

    EXPECT_EQ(loader.requested_symbols, kResolvedSymbols);
}
