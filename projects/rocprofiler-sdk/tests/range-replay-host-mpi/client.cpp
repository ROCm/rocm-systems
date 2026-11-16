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

/**
 * @file tests/range-replay-host-mpi/client.cpp
 *
 * @brief LD_PRELOAD tool that host-replays main.cpp's range on every rank of an MPI job and
 * coordinates the ranks through peer_sync_cb, implemented with MPI collectives on MPI_COMM_WORLD.
 *
 * Environment (see common.hpp for the scenarios):
 *   RR_MPI_MODE, RR_MPI_SPECIAL_RANK   which rank misbehaves, and how
 *   RR_MPI_PASSES                      total passes including the application's own (default 4)
 *   RR_MPI_PASSES_SPECIAL              pass count the special rank asks for, when set
 *   RR_MPI_OUT                         output prefix; writes <prefix>.tool.rank<R>.txt
 *   RR_MPI_EXPECT_STATUS               status every rank's CLOSE must report (default 0)
 *   RR_MPI_SPECIAL_STATUS              status the special rank's CLOSE must report, when set
 *   RR_MPI_EXPECT_PASSES               passes the job runs: PASS callbacks, peer_sync_cb calls
 */

#include "common.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <mpi.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#define RC(call)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        rocprofiler_status_t _s = (call);                                                          \
        if(_s != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[rr-mpi] error '%s' @%d: %s\n",                                               \
                    #call,                                                                         \
                    __LINE__,                                                                      \
                    rocprofiler_get_status_string(_s));                                            \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

namespace
{
rocprofiler_context_id_t g_ctx{0};

int g_rank = -1;

std::atomic<int> g_closes{0};
std::atomic<int> g_close_status{-1};
std::atomic<int> g_pass_enters{0};
std::atomic<int> g_pass_exits{0};
std::atomic<int> g_sync_end{0};
std::atomic<int> g_sync_restored{0};
std::atomic<int> g_pass_mismatches{0};
std::atomic<int> g_order_errors{0};

uint64_t pass_count(uint64_t, rocprofiler_user_data_t)
{
    const int special_rank = env_int("RR_MPI_SPECIAL_RANK", 1);
    const int passes       = env_int("RR_MPI_PASSES", 4);
    return static_cast<uint64_t>((g_rank == special_rank) ? env_int("RR_MPI_PASSES_SPECIAL", passes)
                                                          : passes);
}

// The collective the SDK asks every rank for at the end of every pass. PASS_END reduces the flags
// with a bitwise AND, which doubles as the barrier the SDK needs before it restores memory;
// RESTORED is a plain barrier. The pass index is reduced as well and compared across ranks: the
// n-th call on every rank must be for the same pass.
uint32_t
peer_sync(uint64_t,
          uint64_t                                   pass,
          rocprofiler_range_replay_peer_sync_phase_t phase,
          uint32_t                                   local_flags,
          rocprofiler_user_data_t)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if(initialized == 0) return 0;

    if(phase == ROCPROFILER_RANGE_REPLAY_PEER_SYNC_PASS_END)
    {
        g_sync_end.fetch_add(1);

        uint64_t lowest  = 0;
        uint64_t highest = 0;
        MPI_Allreduce(&pass, &lowest, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&pass, &highest, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
        if(lowest != highest) g_pass_mismatches.fetch_add(1);

        uint32_t global = 0;
        MPI_Allreduce(&local_flags, &global, 1, MPI_UINT32_T, MPI_BAND, MPI_COMM_WORLD);
        return global;
    }

    g_sync_restored.fetch_add(1);
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

void
range_replay_cb(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void*)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;
    auto* p = static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);
    if(p == nullptr || p->range_id != kRangeId) return;

    if(record.operation == ROCPROFILER_RANGE_REPLAY_PASS)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            if(g_pass_enters.load() != g_pass_exits.load()) g_order_errors.fetch_add(1);
            g_pass_enters.fetch_add(1);
        }
        else
        {
            g_pass_exits.fetch_add(1);
        }
        return;
    }

    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) return;

    switch(record.operation)
    {
        case ROCPROFILER_RANGE_REPLAY_CONFIG:
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if(initialized != 0) MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

            p->pass_count_cb = pass_count;
            p->host_replay   = 1;
            p->peer_sync_cb  = peer_sync;
            break;
        }
        case ROCPROFILER_RANGE_REPLAY_CLOSE:
            g_closes.fetch_add(1);
            g_close_status.store(static_cast<int>(p->status));
            break;
        default: break;
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    RC(rocprofiler_create_context(&g_ctx));
    RC(rocprofiler_configure_callback_tracing_service(
        g_ctx, ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, nullptr, 0, range_replay_cb, nullptr));
    RC(rocprofiler_start_context(g_ctx));
    return 0;
}

void
tool_fini(void*)
{
    const int expect_passes = env_int("RR_MPI_EXPECT_PASSES", 4);
    const int special_rank  = env_int("RR_MPI_SPECIAL_RANK", 1);
    const int default_state = env_int("RR_MPI_EXPECT_STATUS", 0);
    const int expect_status =
        (g_rank == special_rank) ? env_int("RR_MPI_SPECIAL_STATUS", default_state) : default_state;
    const auto out_prefix = env_str("RR_MPI_OUT", "");

    const int status = g_close_status.load();
    fprintf(stderr,
            "[rr-mpi] rank %d closes=%d status=%d (expected %d) pass_enters=%d pass_exits=%d "
            "sync_end=%d sync_restored=%d\n",
            g_rank,
            g_closes.load(),
            status,
            expect_status,
            g_pass_enters.load(),
            g_pass_exits.load(),
            g_sync_end.load(),
            g_sync_restored.load());

    auto ok   = true;
    auto fail = [&](const char* what) {
        fprintf(stderr, "[rr-mpi] FAIL: rank %d: %s\n", g_rank, what);
        ok = false;
    };

    if(g_closes.load() != 1) fail("expected exactly one CLOSE for the range");
    if(status != expect_status) fail("CLOSE reported an unexpected status");
    if(g_pass_enters.load() != expect_passes - 1 || g_pass_exits.load() != expect_passes - 1)
        fail("unexpected number of replayed passes");
    // One PASS_END exchange per pass boundary, including the application's own run, and one
    // RESTORED barrier per restore.
    if(g_sync_end.load() != expect_passes) fail("unexpected number of PASS_END exchanges");
    if(g_sync_restored.load() != expect_passes - 1) fail("unexpected number of RESTORED barriers");
    if(g_pass_mismatches.load() != 0) fail("ranks were at different passes in one exchange");
    if(g_order_errors.load() != 0) fail("PASS callbacks out of order");

    if(!out_prefix.empty())
    {
        std::ofstream out{out_prefix + ".tool.rank" + std::to_string(g_rank) + ".txt"};
        out << "status=" << status << "\n";
        out << "pass_enters=" << g_pass_enters.load() << "\n";
        out << "sync_end=" << g_sync_end.load() << "\n";
        out << "sync_restored=" << g_sync_restored.load() << "\n";
        out << "result=" << (ok ? "PASS" : "FAIL") << "\n";
    }
    fprintf(stderr, "[rr-mpi] rank %d %s\n", g_rank, ok ? "PASS" : "FAIL");
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name        = "rr-mpi";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
